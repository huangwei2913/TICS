import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import XLMRobertaModel
import copy
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

        # 这个就是你的 Audio Projector
        # 它是一个小型的 MLP，负责将特征转换到对比空间，与动量老师之间形成对比学习，要求全局音频特征与当前对应的
        #文本特征对齐，而与其他一个批中的样本的文本特征远离
        self.audio_projector = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512) # 最终对比维度
        )

        ## [核心] 投影层：把音频摘要投影到 "文本语义空间"
        # 加上 LayerNorm 能大幅稳定 MSE Loss
        self.audio_global_bottleneck = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim), # 输出 1024 维
            nn.LayerNorm(semantic_dim)  # 输出前再次归一化，确保 MSE 计算时的数值稳定性
        )
        #两个部分
        self.text_backbone = XLMRobertaFusionExpert(xlmr_path, target_dim=1024)
        # 2. Projector: 刚才拿出来的 global_head 变成了这个
        # 负责 1024 -> 512 (对比空间)
        self.text_projector = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512) # 最终对比维度
        )

        # =========================================================
        # Part 4: 文本导师层 (Text Teacher) - 仅在训练时激活
        # =========================================================
        self.text_backbone_m = copy.deepcopy(self.text_backbone)
        for p in self.text_backbone_m.parameters(): p.requires_grad = False

        # 2. Projector Copy
        self.text_projector_m = copy.deepcopy(self.text_projector)
        for p in self.text_projector_m.parameters(): p.requires_grad = False
        # =========================================================
        # Part 5: 动态精炼层 (Dynamic Refinement)
        # =========================================================
        # 接收全局特征，预测 K 值，进行最终语义合并
        self.refiner = ToMeSemanticRefiner(dim=semantic_dim)

        #注册正负样本的队列管理器
        # 队列长度 (例如 65536)，特征维度 512
        self.register_buffer("queue", torch.randn(512, 2000))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        现在这里逻辑就非常清晰了：
        我们要更新整个 Text 分支（Backbone + Projector）
        """
        # A. 更新 Backbone 里的参数 (layer_weights, dim_alignment)
        # 注意：XLM-R 本身是冻结的，不会变，但 FusionExpert 里那几个可学习层会变
        for param_on, param_mom in zip(self.text_backbone.parameters(), 
                                       self.text_backbone_m.parameters()):
            param_mom.data = param_mom.data * self.m + param_on.data * (1. - self.m)
            
        # B. 更新 Projector 的参数
        for param_on, param_mom in zip(self.text_projector.parameters(), 
                                       self.text_projector_m.parameters()):
            param_mom.data = param_mom.data * self.m + param_on.data * (1. - self.m)  


    # --- 修改前的 TICSMoCo.py ---
    # self.queue[:, ptr:ptr + batch_size] = keys.T

    # --- 修改后的 TICSMoCo.py ---
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # 这里的 keys 是当前 batch 的特征 [B, Dim]
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # --- 核心修复：处理队列末尾溢出问题 ---
        # 如果当前 batch 会超出队列末尾，我们就截断它，或者简单地只更新能放下的部分
        # 更稳健的做法是：
        rem = self.queue.shape[1] - ptr # 队列剩余空间
        if batch_size > rem:
            # 如果放不下，先把能放下的放了，剩下的从头开始放 (或者直接跳过本次溢出部分)
            self.queue[:, ptr:] = keys[:rem].T
            self.queue[:, :batch_size - rem] = keys[rem:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            
        # 更新指针，取模确保永远在队列范围内
        ptr = (ptr + batch_size) % self.queue.shape[1]
        self.queue_ptr[0] = ptr
    
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
        audio_global_pred = self.audio_global_bottleneck(latent_summary) # [B, 1024]
        # 1. 投影到 512 维对比空间 (用于 MoCo)
        q = self.audio_projector(audio_global_pred)
        q = F.normalize(q, dim=-1) # [B, 512]

        k_m = None
        text_global_online = None

        if self.training and text_input_ids is not None:
            # 1. 执行动量更新：让 Teacher 向 Online 靠拢
            self._momentum_update_key_encoder()
            
            # 2. 计算 Online 文本特征 (用于训练 text_backbone 和 text_projector)
            # 这部分计算是有梯度的
            text_feat_online = self.text_backbone(text_input_ids, text_mask)
            text_global_online = text_feat_online # 1024维真值，用于 audio_global_pred 的回归
            
            # 3. 计算 Momentum 文本特征 (MoCo Key)
            # 使用动量分支，不计算梯度，确保 Key 极度稳定
            with torch.no_grad():
                text_feat_m = self.text_backbone_m(text_input_ids, text_mask)
                k_m = self.text_projector_m(text_feat_m)
                k_m = F.normalize(k_m, dim=-1) # [B, 512]
        
        # =========================================================
        # 4. 动态语义精炼 (ToMe Refinement)
        # =========================================================
        # 使用修正后的 audio_global_pred 指导 Refiner
        refined_anchors, pred_k = self.refiner(
            context_padded_trans, 
            audio_global_pred, 
            padding_mask=padding_mask
        )

        # =========================================================
        #audio_global_pred 和 text_global_online 是双向奔赴，也就说彼此靠近
        #希望全局的语音特征（audio_global_pred）能够和文本的全局特征（text_global_online）尽可能接近
        #希望样本中的音频产生的文本特征（q）与该音频对应的文本特征（k_m）无限接近，且远离其他文本
        #精炼后的特征中能够预测出一个段落个数（pred_k），代表在这个音频中会有多少个子句
        #通过监督学习，将 p_score 与每一个词（或子句）对应的时间段匹配起来
        #精炼锚点 (refined_anchors) 解决了**“每一段到底代表了哪个具体的语义点”**。
        #refined_anchors 就是你从长语音中提取出来的**“语义浓缩包”**
        #可以通过 refined_anchors，把 10 秒语音浓缩成了 pred_k 个（比如 4 个）精炼特征。
        #下游模型只需要处理这 4 个向量就能理解整段话。这极大地降低了长语音处理的计算成本。
        #想法：既然 pred_k 预测有 4 个子句，那么 refined_anchors 就有 4 个向量。我们可以要求这 4 
        # 个向量分别去靠近文本中对应的 4 个短句特征。
        return {
            "p_score": p_score,               # 边界得分
            "q": q,                           # MoCo Query (512) 
            "k_m": k_m,                       # MoCo Key (512, 动量)
            "audio_global_pred": audio_global_pred, # 预测的文本向量 (1024)
            "text_global_online": text_global_online, # 真实的文本向量 (1024)
            "refined_anchors": refined_anchors, # 精炼后的锚点
            "pred_k": pred_k,                  # 预测的段落数，应该是一段音频中，逻辑意义完整、且在声学上有明显停顿或转折的片段数量。
            "loss_skip": False
        }