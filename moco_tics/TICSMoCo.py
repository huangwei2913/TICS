import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import XLMRobertaModel

from .modules import * # 导入 FrozenHubertBackbone, FeatureFusion, TICSBoundaryStudent, SCPCBoundaryHardener
from .AttentionBasedSegmentor import AttentiveSegmentPooling
from .EnhancedSegmentEncoder import EnhancedSegmentEncoder # 确认文件名是 encoder.py 或 EnhancedSegmentEncoder.py
from .SemanticCompressor import ToMeSemanticRefiner
from .projectors import EmotionConsistencyHead  #
from .GlobalLatentSummarizer import GlobalLatentSummarizer
from .XLMRobertaFusionExpert import XLMRobertaFusionExpert

class TICS_MoCo(nn.Module):
    def __init__(self, 
                 backbone_path: str,
                 teacher_config: dict = None,
                 m: float = 0.996, 
                 temp: float = 0.1,
                 xlmr_path="/mnt/conda_data/facebook/xlm-roberta-base", #用于文本编码器，在推理的时候不起作用
                 is_stage2=True,  #默认是端到端的训练，以后进行多个阶段的训练，可以用这里来区分
                 is_traing=True,  #默认我们是训练模式，训练模式的时候使用文本编码器的特征，而当这个值等于False的时候，表示使用的是伪文本特征
                 dim=768,
                 semantic_dim: int = 1024,
                 large_hubert_path="/mnt/conda_data/facebook/hubert-large-ls960-ft" # 用于 segment_encoder 音频编码器段落与段落之间
                 ):
        super().__init__()
        self.is_stage2 = is_stage2
        self.is_traing = is_traing
        self.m = m
        self.temp = temp
        self.fusion_layers = [2, 5, 9, 11]

        # --- 1. 物理层组件 (音频帧处理) ---
        # 基础特征提取 (Hubert-Base)
        self.backbone = FrozenHubertBackbone(backbone_path)
        self.fusion = FeatureFusion(dim=dim, layers_to_use=self.fusion_layers)
        
        # 边界学生：预测 P_score 并将特征升维至 1024 (semantic_dim)
        self.student = TICSBoundaryStudent(input_dim=dim) 
        self.hardener = SCPCBoundaryHardener()

        # --- 2. 物理-语义衔接层 (段落聚合) ---
        # 利用注意力打分寻找语义重心，将帧特征聚合为段落特征
        self.segment_pooler = AttentiveSegmentPooling(input_dim=semantic_dim)

        # --- 3. 语义增强层 (段落关系建模) ---
        # 注入 RoPE 位置信息，并加载 Hubert-Large 权重，整合段落间的逻辑相关性
        self.segment_encoder = EnhancedSegmentEncoder(
            input_dim=semantic_dim, 
            segment_dim=semantic_dim, 
            num_layers=12
        )
        # 跨级初始化：用 Large 模型的灵魂武装段落编码器
        self.segment_encoder.load_pretrained_weights(large_hubert_path)

        #这里设置一个全局语义特征是正确的
        self.global_summarizer = GlobalLatentSummarizer(dim=1024)

        ## [核心] 投影层：把音频摘要投影到 "文本语义空间"
        # 加上 LayerNorm 能大幅稳定 MSE Loss
        self.audio_global_bottleneck = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim) # 输出 1024 维
        )

        # =========================================================
        # Part 4: 文本导师层 (Text Teacher) - 仅在训练时激活
        # =========================================================
        if self.is_stage2 and self.is_traing:
            print(f"🎓 初始化文本导师: {xlmr_path}")
            # 多层融合专家 (768 -> 1024)
            # 注意：这个模块内部已经冻结了 Backbone，只训练融合权重
            self.text_teacher = XLMRobertaFusionExpert(
                xlmr_path=xlmr_path, 
                target_dim=semantic_dim
            )
        else:
            self.text_teacher = None

        # =========================================================
        # Part 5: 动态精炼层 (Dynamic Refinement)
        # =========================================================
        # 接收全局特征，预测 K 值，进行最终语义合并
        self.refiner = ToMeSemanticRefiner(dim=semantic_dim)

  
    
    def generate_padding_mask(self, segments_list: list) -> torch.Tensor:
        """
        生成用于 Transformer 的 Padding Mask
        True 表示该位置是 Padding (需要被忽略), False 表示有效
        """
        batch_size = len(segments_list)
        # 每个音频样本切分出的段落数量 Si
        lengths = [s.size(0) for s in segments_list]
        max_len = max(lengths) if lengths else 0
        device = segments_list[0].device
        # 初始化为 True (全屏蔽)
        mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)
        # 有效区域设为 False
        for i, l in enumerate(lengths):
            mask[i, :l] = False
            
        return mask


    def forward(self, wav, text_input_ids=None, text_mask=None, target_emo=None):
        # 接下来我们将在这个结构基础上实现流式逻辑的 forward
        # --- 1. 物理层 (Physical Layer) ---
        # 提取 Hubert 原始特征 [B, T, 768]
        feat_list = self.backbone(wav, layers_to_extract=self.fusion_layers)
        seq_feat, fused_cls = self.fusion(feat_list)
        p_score, E_context = self.student(seq_feat, fused_cls)
        b_hard_ste = self.hardener(p_score)
        # --- 2. 动态切分 (连接点) ---
        # 我们使用 E_context (1024维) 而不是 seq_feat (768维) 进行池化
        # 这样保留了 Student 对上下文的理解
        # seg_list: List[Tensor(Si, 1024)]
        seg_list, dur_list = self.segment_pooler(E_context, b_hard_ste)
        batch_size = len(seg_list)
        seg_lens = [s.size(0) for s in seg_list]
        max_seg_len = max(seg_lens)
        if max_seg_len == 0:    #全是静音的时候
            return {"p_score": p_score, "loss_skip": True} # 或者做其他兜底处理
        
        padded_segments = pad_sequence(seg_list, batch_first=False)    # [Max_S, B, 1024]
        padded_durations = pad_sequence(dur_list, batch_first=False)   # [Max_S, B, 1]
        # 2. 生成掩码 [B, Max_S]
        # 注意：Transformer 的 src_key_padding_mask 维度通常是 [B, S]
        padding_mask = self.generate_padding_mask(seg_list)
        # --- 3. 增强编码 (全局上下文) ---
        # 这一步内部会用到 RoPE，它能感知各个段落的相对位置
        # context_padded: [Max_S, B, 1024]
        context_padded = self.segment_encoder(padded_segments, padded_durations, padding_mask)
        # 我们利用之前记录的有效长度，把补齐的 Tensor 拆回 List
        # 转回 [B, S, D] 方便切分
        context_padded_trans = context_padded.permute(1, 0, 2) # [B, Max_S, 1024]

        # --- 4. 全局语义脑补 (The "Brain") ---
        # A. 提取 Latent Summary (Summarizer)
        latent_summary = self.global_summarizer(context_padded_trans, mask=padding_mask)
        # B. 投影到文本空间 (Bottleneck)
        # 这是音频侧生成的 "伪文本全局特征"
        audio_global_pred = self.audio_global_bottleneck(latent_summary)

        # --- 5. 获取文本真值 (仅在训练且有文本时) ---
        text_global_target = None
        if self.training and self.is_stage2 and text_input_ids is not None:
            # 这一步不需要梯度传回 Backbone，但 Expert 内部的融合层需要梯度
            # Expert 内部已经处理好了 no_grad 逻辑 (针对 Backbone)
            text_global_target = self.text_teacher(text_input_ids, text_mask)
        
        # --- 6. 动态语义精炼 (ToMe) ---
        # 使用脑补出的 audio_global_pred 来指导切分
        # refined_anchors: 最终用于下游任务的句子特征
        # pred_k: 预测的句子数量 (用于 Loss 回归)

        # 修正：显式传入 padding_mask，确保 refiner 只处理有效词块
        refined_anchors, pred_k = self.refiner(
            context_padded_trans, 
            audio_global_pred, 
            padding_mask=padding_mask  # <-- 必须加上这一句
        )

        # --- 7. 返回结果字典 ---
        return {
            "p_score": p_score,               # 用于边界 Loss
            "audio_global": audio_global_pred,# 用于语义对齐 Loss (Student)
            "text_global": text_global_target,# 用于语义对齐 Loss (Teacher)
            "pred_k": pred_k,                 # 用于数量回归 Loss
            "anchors": refined_anchors,       # 用于对比学习/情感一致性 Loss
            "padding_mask": padding_mask,
            "loss_skip": False
        }