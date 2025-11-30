import torch
import torch.nn as nn
import copy
from torch.nn.utils.rnn import pad_sequence
from .modules import * # 导入所有基础组件
from .encoder import SegmentEncoder
import torch.nn.functional as F
from util.utils import load_large_weights_tolerant

class TICS_MoCo(nn.Module):
    def __init__(self, 
                    backbone_path: str, # HuBERT Base 路径
                    teacher_config: dict, # 教师模型配置 (SegmentEncoder)
                    m: float = 0.996, 
                    temp: float = 0.1):
        super().__init__()
        
        # --- A. 共享组件 (Base Encoder) ---
        # 这些部分负责生成片段，既用于 Query 也用于 Key
        self.fusion_layers = [2, 5, 9]
        self.backbone = FrozenHubertBackbone(backbone_path)
        self.fusion = FeatureFusion(dim=768, layers_to_use=self.fusion_layers)
        self.student = TICSBoundaryStudent(input_dim=768)
        self.hardener = SCPCBoundaryHardener()
        
        # --- B. 对比学习组件 ---
        self.m = m
        self.temp = temp
        
        # Query Encoder (可训练教师)
        self.encoder_q = SegmentEncoder(**teacher_config)
        #self.encoder_q.load_xlarge_weights() # 加载强权重
        load_large_weights_tolerant(
            target_model=self.encoder_q , 
            large_checkpoint_path="/mnt/facebook/hubert-large-ls960-ft"
        )
        
        # Key Encoder (动量教师)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False # 冻结 Key Encoder
            
        # 预测头 (仅用于 Query)
        self.predictor = nn.Linear(teacher_config.get('segment_dim', 1024), 
                                   teacher_config.get('segment_dim', 1024))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        MoCo 核心：动量更新 Key Encoder。
        DeepSpeed 注意：在分布式训练中无需特殊处理，因为 Key 是本地更新的。
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward_single_view(self, wav, encoder_type='q'):
        """执行单视图的前向传播：从波形到片段嵌入"""
        

        feature_list = self.backbone(wav, layers_to_extract=self.fusion_layers)
        feats_dict = {
            layer_idx: feat 
            for layer_idx, feat in zip(self.fusion_layers, feature_list)
        }
        # 1. 提取特征 (Backbone + Fusion)
        #feats_dict = self.backbone(wav,layers_to_extract=self.fusion_layers)


        seq_feat, fused_cls = self.fusion(feats_dict)
        
        # 2. 预测边界 (Student)
        p_score = self.student(seq_feat, fused_cls)
        b_hard_ste = self.hardener(p_score)
        
        # 3. 分段池化 (Pooling)
        # 输出 List[Tensor]
        segments_list = segment_pooling(seq_feat, b_hard_ste)
        
        # 4. 填充准备 (Padding)
        # Pad 默认是 (Max_Seg, B, D)
        padded_segments = pad_sequence(segments_list, batch_first=False)
        
        # 生成 Mask (B, Max_Seg)
        lengths = [s.shape[0] for s in segments_list]
        mask = torch.ones(len(segments_list), padded_segments.shape[0], dtype=torch.bool, device=wav.device)
        for i, l in enumerate(lengths):
            mask[i, :l] = False
            
        # 5. 教师编码
        if encoder_type == 'q':
            embeddings = self.encoder_q(padded_segments, mask)
        else:
            with torch.no_grad():
                embeddings = self.encoder_k(padded_segments, mask)
                
        return embeddings, mask, lengths

    def forward(self, view1_wav, view2_wav):
        """
        输入: 两个增强视图 (Audio)
        输出: 损失值字典
        """
        # 更新动量编码器
        self._momentum_update_key_encoder()
        
        # --- 1. 计算 View 1 (Query) 和 View 2 (Key) ---
        # Q1: 视图1通过 Query Encoder
        z_q1, mask_q1, len_q1 = self.forward_single_view(view1_wav, 'q')
        # K2: 视图2通过 Key Encoder (用于对称性损失)
        z_k2, mask_k2, len_k2 = self.forward_single_view(view2_wav, 'k')
        
        # --- 2. 计算 View 2 (Query) 和 View 1 (Key) ---
        # 为了对称性，通常交换视图再算一次
        z_q2, mask_q2, len_q2 = self.forward_single_view(view2_wav, 'q')
        z_k1, mask_k1, len_k1 = self.forward_single_view(view1_wav, 'k')
        
        # --- 3. 预测头 ---
        p_q1 = self.predictor(z_q1)
        p_q2 = self.predictor(z_q2)

        p_q1 = F.normalize(p_q1, dim=-1)
        p_q2 = F.normalize(p_q2, dim=-1)
        z_k1 = F.normalize(z_k1, dim=-1)
        z_k2 = F.normalize(z_k2, dim=-1)


        # --- 4. 返回嵌入供 Loss 模块使用 ---
        return {
            "q1": p_q1, "k2": z_k2, "mask_q1": mask_q1, 
            "len_q1": len_q1, "len_k2": len_k2, # 第一组长度
            
            "q2": p_q2, "k1": z_k1, "mask_q2": mask_q2,
            "len_q2": len_q2, "len_k1": len_k1  # <--- 必须添加这两项！
        }