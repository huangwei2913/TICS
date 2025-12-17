import torch
import torch.nn as nn
import copy
from torch.nn.utils.rnn import pad_sequence
from .modules import * # 导入所有基础组件
from .encoder import SegmentEncoder,EnhancedSegmentEncoder
import torch.nn.functional as F
from util.utils import load_large_weights_tolerant
from transformers import XLMRobertaModel

class TICS_MoCo(nn.Module):
    def __init__(self, 
                    backbone_path: str, # HuBERT Base 路径
                    teacher_config: dict, # 教师模型配置 (SegmentEncoder)
                    m: float = 0.996, 
                    temp: float = 0.1,
                    xlmr_path=None, 
                    is_stage2=False
                    ):
        super().__init__()
        

        self.is_stage2 = is_stage2
        # --- A. 共享组件 (Base Encoder) ---
        # 这些部分负责生成片段，既用于 Query 也用于 Key
        self.fusion_layers = [2, 5, 9]
        self.backbone = FrozenHubertBackbone(backbone_path)
        self.fusion = FeatureFusion(dim=768, layers_to_use=self.fusion_layers)
        self.student = TICSBoundaryStudent(input_dim=768)
        self.hardener = SCPCBoundaryHardener()  #STE 硬化
        
        # --- B. 对比学习组件 ---
        self.m = m
        self.temp = temp
        
        # Query Encoder (可训练教师)
        self.encoder_q = EnhancedSegmentEncoder(**teacher_config)

        load_large_weights_tolerant(
            target_model=self.encoder_q , 
            large_checkpoint_path="/mnt/facebook/hubert-large-ls960-ft"
        )
        
        # Key Encoder (动量教师)
        self.encoder_k = copy.deepcopy(self.encoder_q)

        #--- Stage 1 专用 ---
        self.predictor = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )



        for param in self.encoder_k.parameters():
            param.requires_grad = False # 冻结 Key Encoder



        # --- Stage 2 专用：XLM-R 语义对齐 ---
        if is_stage2:
            print(f"Loading XLM-R Teacher from {xlmr_path}...")
            # 仅加载主体，不带预测头
            self.xlmr_teacher = XLMRobertaModel.from_pretrained(xlmr_path)
            # 冻结老师，不参与更新
            for param in self.xlmr_teacher.parameters():
                param.requires_grad = False
            
            # 定义投影层：将语音语义映射到文本语义空间
            # 虽然维度都是 1024，但空间分布不同，需要一个翻译官
            self.llm_projector = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Linear(1024, 1024)
            )            
            
        # 预测头 (仅用于 Query)
#        self.predictor = nn.Linear(teacher_config.get('segment_dim', 1024), 
#                                   teacher_config.get('segment_dim', 1024))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        MoCo 核心：动量更新 Key Encoder。
        DeepSpeed 注意：在分布式训练中无需特殊处理，因为 Key 是本地更新的。
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)



    def forward_single_view(self, wav, encoder_type='q'):
        # 1. 基础特征提取
        feature_list = self.backbone(wav)
        seq_feat, fused_cls = self.fusion(feature_list)

        # 2. 边界预测 (Student)
        p_score, E_context = self.student(seq_feat, fused_cls)

        # 3. 边界硬化与全向量化池化 (带 duration)
        b_hard_ste = self.hardener(p_score)
        segments_list, durations_list = segment_pooling_with_durations(E_context, b_hard_ste)

        # 4. 准备 Padding 数据
        # lengths 是每个 batch 真实的段数
        lengths = torch.tensor([len(s) for s in segments_list], device=wav.device)
        max_len = lengths.max().item()
        batch_size = len(segments_list)
        
        # 将 List[Tensor] 转换为对齐的 Tensor (S, B, D)
        padded_segments = pad_sequence(segments_list, batch_first=False) 
        padded_durations = pad_sequence(durations_list, batch_first=False) 

        # 5. 实现你的循环掩码逻辑 (这就是原本的 generate_padding_mask)
        mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=wav.device)
        for i, l in enumerate(lengths):
            mask[i, :l] = False  # 把真实长度部分设为 False，表示 Transformer 可以看

        # 6. 送入你强悍的 EnhancedSegmentEncoder
        encoder = self.encoder_q if encoder_type == 'q' else self.encoder_k
        
        if encoder_type == 'q':
            # 传入 padded_durations 以适配时长嵌入
            embeddings = encoder(padded_segments, padded_durations, mask)
        else:
            with torch.no_grad():
                embeddings = encoder(padded_segments, padded_durations, mask)

        # 7. 完整返回，一个都不少
        return embeddings, mask, lengths, p_score, E_context, padded_durations

    def forward(self, view1_wav, view2_wav):
        """
        输入: 两个增强视图 (Audio)
        输出: 包含 L_MoCo 和 L_sup 所需项的字典
        """
        # 更新动量编码器 (k) 的权重
        self._momentum_update_key_encoder()
        
        # --- 1. 计算 View 1 (Query) 和 View 2 (Key) ---
        # ⚠️ 现在接收 6 个返回值：增加了一个 _ 表示忽略末尾的 durations
        z_q1, mask_q1, len_q1, p_score_q1, E_context_q1, d_q1 = self.forward_single_view(view1_wav, 'q')
        z_k2, mask_k2, len_k2, _, _, d_k2 = self.forward_single_view(view2_wav, 'k')
        
        # --- 2. 计算 View 2 (Query) 和 View 1 (Key) ---
        z_q2, mask_q2, len_q2, p_score_q2, E_context_q2, d_q2 = self.forward_single_view(view2_wav, 'q')
        z_k1, mask_k1, len_k1, _, _, d_k1 = self.forward_single_view(view1_wav, 'k')
        
        # --- 3. 预测头 (Predictor) ---
        # Predictor 只作用于 Query 分支
        p_q1 = self.predictor(z_q1)
        p_q2 = self.predictor(z_q2)

        # 归一化，确保对比学习在单位超球面上进行
        p_q1 = F.normalize(p_q1, dim=-1)
        p_q2 = F.normalize(p_q2, dim=-1)
        z_k1 = F.normalize(z_k1, dim=-1)
        z_k2 = F.normalize(z_k2, dim=-1)

        # --- 4. 返回嵌入供 Loss 模块使用 ---
        return {
            # MoCo 对比项 (q-k 对应关系)
            "q1": p_q1, 
            "k2": z_k2, 
            "mask_q1": mask_q1, 
            "len_q1": len_q1, 
            "len_k2": len_k2, 
            "dur_q1": d_q1, # 保持数据流完整

            "q2": p_q2, 
            "k1": z_k1, 
            "mask_q2": mask_q2,
            "len_q2": len_q2, 
            "len_k1": len_k1,
            "dur_q2": d_q2,
            
            # L_sup 边界监督项 (主要使用 view1 的预测结果与标签对比)
            "P_score": p_score_q1, 
            
            # 额外的上下文特征
            "E_context": E_context_q1 
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