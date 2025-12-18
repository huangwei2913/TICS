import torch
import torch.nn as nn
import copy
from torch.nn.utils.rnn import pad_sequence
from .modules import * # 导入所有基础组件
from .encoder import EnhancedSegmentEncoder
import torch.nn.functional as F
from util.utils import load_large_weights_tolerant
from transformers import XLMRobertaModel


class TICS_MoCo(nn.Module):
    def __init__(self, 
                 backbone_path: str,
                 teacher_config: dict,
                 m: float = 0.996, 
                 temp: float = 0.1,
                 xlmr_path=None, 
                 is_stage2=False
                 ):
        super().__init__()
        self.is_stage2 = is_stage2
        self.m = m
        self.temp = temp
        self.fusion_layers = [2, 5, 9, 10]

        # 1. 共享基础组件 (Backbone, Fusion, Student)
        # 这里遵循你的直觉：基础特征提取部分不分 Q/K，节省显存
        self.backbone = FrozenHubertBackbone(backbone_path)
        self.fusion = FeatureFusion(dim=768, layers_to_use=self.fusion_layers)
        self.student = TICSBoundaryStudent(input_dim=768)
        self.hardener = SCPCBoundaryHardener()

        # 2. 区分 Q 和 K 的增强编码器 (EnhancedSegmentEncoder)
        #self.encoder_q = EnhancedSegmentEncoder(**teacher_config)
        

        self.encoder_q = EnhancedSegmentEncoder(
            input_dim=1024,   # 这里的参数必须显式传 1024
            segment_dim=1024,
            num_layers=12
        )
        self.encoder_k = EnhancedSegmentEncoder(
            input_dim=1024,   # 这里的参数也必须是 1024
            segment_dim=1024,
            num_layers=12
        )

        # --- 关键：必须先加载预训练权重，再进行 deepcopy ---
        self.encoder_q.load_pretrained_weights(
            large_checkpoint_path="/mnt/facebook/hubert-large-ls960-ft",
            start_layer=6
        )
        
        # 影子模型：教师编码器
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # 3. 投影头与预测器
        segment_dim = teacher_config.get('segment_dim', 1024)
        self.predictor = nn.Sequential(
            nn.Linear(segment_dim, segment_dim),
            nn.LayerNorm(segment_dim),
            nn.GELU(),
            nn.Linear(segment_dim, segment_dim)
        )

        # 4. Stage 2 逻辑
        if is_stage2:
            print(f"Loading XLM-R Teacher from {xlmr_path}...")
            self.xlmr_teacher = XLMRobertaModel.from_pretrained(xlmr_path)
            for param in self.xlmr_teacher.parameters():
                param.requires_grad = False
            
            self.llm_projector = nn.Sequential(
                nn.Linear(segment_dim, segment_dim),
                nn.LayerNorm(segment_dim),
                nn.GELU(),
                nn.Linear(segment_dim, segment_dim)
            )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """仅更新经过加工的编码器部分"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward_single_view(self, wav):
        """
        基础处理流程：提取音频并切分为片段（Raw Segments）
        不包含 Encoder 编码，以便在外层控制 Shuffle 策略
        """
        # 基础特征提取
        feature_list = self.backbone(wav, layers_to_extract=self.fusion_layers)
        seq_feat, fused_cls = self.fusion(feature_list)

        # 边界预测
        p_score, E_context = self.student(seq_feat, fused_cls)
        b_hard_ste = self.hardener(p_score)

        # 全向量化池化
        segments_list, durations_list = segment_pooling_with_durations(E_context, b_hard_ste)

        # 准备数据结构
        lengths = torch.tensor([len(s) for s in segments_list], device=wav.device)
        batch_size = len(segments_list)
        
        # 对齐 Tensor (S, B, D)
        padded_segments = pad_sequence(segments_list, batch_first=False) 
        padded_durations = pad_sequence(durations_list, batch_first=False) 

        # 生成掩码
        mask = self.generate_padding_mask(segments_list)

        return padded_segments, padded_durations, mask, lengths, p_score, E_context

    def forward(self, view1_wav, view2_wav, aug_mode='baseline'):
        """MoCo 对称训练逻辑，支持特征重组 (Shuffle)"""
        self._momentum_update_key_encoder()

        # 1. 获取原材料 (此时 view1 和 view2 都是原始顺序)
        z1_raw, d1_raw, m1, _, p1_score, E1 = self.forward_single_view(view1_wav)
        z2_raw, d2_raw, m2, _, p2_score, E2 = self.forward_single_view(view2_wav)

        # 2. 定义局部处理函数：处理 Shuffle -> 编码 -> 还原
        def encode_process(z, d, m, encoder_module, mode):
            if mode == 'shuffle':
                S, B, D = z.shape
                # 打乱索引
                shuffle_idx = torch.stack([torch.randperm(S) for _ in range(B)], dim=1).to(z.device)
                reverse_idx = torch.argsort(shuffle_idx, dim=0)
                
                # 执行打乱
                z_sh = torch.gather(z, 0, shuffle_idx.unsqueeze(-1).expand(-1, -1, D))
                d_sh = torch.gather(d, 0, shuffle_idx.unsqueeze(-1).expand(-1, -1, 1))
                
                # 编码
                z_enc = encoder_module(z_sh, d_sh, m)
                
                # 还原：这一步保证了输出 z 依然能和 p_score1/y_true 对应
                return torch.gather(z_enc, 0, reverse_idx.unsqueeze(-1).expand(-1, -1, D))
            else:
                return encoder_module(z, d, m)

        # 3. 对称分支计算 (MoCo v2 标准)
        # q1 vs k2
        q1 = encode_process(z1_raw, d1_raw, m1, self.encoder_q, aug_mode)
        k2 = encode_process(z2_raw, d2_raw, m2, self.encoder_k, aug_mode)
        
        # q2 vs k1
        q2 = encode_process(z2_raw, d2_raw, m2, self.encoder_q, aug_mode)
        k1 = encode_process(z1_raw, d1_raw, m1, self.encoder_k, aug_mode)

        # 4. Predictor 与 归一化
        p1 = F.normalize(self.predictor(q1), dim=-1)
        p2 = F.normalize(self.predictor(q2), dim=-1)
        z1 = F.normalize(k1, dim=-1)
        z2 = F.normalize(k2, dim=-1)

        return {
            "q1": p1, "k2": z2, "mask1": m1,
            "q2": p2, "k1": z1, "mask2": m2,
            "P_score": p1_score,
            "E_context": E1
        }

    def generate_padding_mask(self, segments_list: list[torch.Tensor]) -> torch.Tensor:
        # 1. 获取 Batch Size 和最大片段长度 (S_max)
        batch_size = len(segments_list)
        lengths = [s.size(0) for s in segments_list]
        max_len = max(lengths)
        device = segments_list[0].device

        # 2. 初始化全为 True 的矩阵 (默认全部屏蔽)
        mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)

        # 3. 为每个样本的有效长度部分填充 False (取消屏蔽)
        for i, l in enumerate(lengths):
            mask[i, :l] = False

        return mask

    #
    def forward_stage2(self, wav, text_input_ids, text_attention_mask):
        """
        Stage 2 核心：语音片段特征与文本特征对齐
        """
        # 1. 语音端流程 (与 Stage 1 一致)
        feature_list = self.backbone(wav)
        seq_feat, fused_cls = self.fusion(feature_list)
        p_score, E_context = self.student(seq_feat, fused_cls)
        
        b_hard_ste = self.hardener(p_score)
        segments_list, durations_list = segment_pooling_with_durations(E_context, b_hard_ste)
        
        # 准备 Padding 和 Mask
        x_padded = pad_sequence(segments_list, batch_first=False) # (S, B, 1024)
        d_padded = pad_sequence(durations_list, batch_first=False) # (S, B, 1)
        audio_mask = self.generate_padding_mask(segments_list)     # (B, S)
        
        # 得到语音片段特征 z_audio: (S, B, 1024)
        z_audio = self.encoder_q(x_padded, d_padded, audio_mask)
        
        # 投影到文本语义空间
        z_audio_proj = self.llm_projector(z_audio.transpose(0, 1)) # (B, S, 1024)

        # 2. 文本端流程 (使用 XLM-R 老师)
        with torch.no_grad():
            xlmr_out = self.xlmr_teacher(
                input_ids=text_input_ids, 
                attention_mask=text_attention_mask
            )
            # 提取最后隐层的特征 (last_hidden_state)
            # z_text: (B, N, 1024)
            z_text = xlmr_out.last_hidden_state

        return {
            "z_audio_proj": z_audio_proj,
            "z_text": z_text,
            "p_score": p_score,
            "audio_mask": audio_mask,
            "text_mask": text_attention_mask
        }