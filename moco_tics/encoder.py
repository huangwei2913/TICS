import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import HubertModel

HUGE_XLARGE_PATH = "/mnt/facebook/hubert-xlarge-ls960-ft"

class SegmentEncoder(nn.Module):
    # 匹配 HuBERT XLarge 的核心参数
    HUGO_XLARGE_DIM = 1024
    HUGO_XLARGE_HEADS = 16  # HuBERT XLarge 使用 16 个注意力头
    
    def __init__(self, 
                 input_dim: int = 768,        # Segment Pooling 得到的特征维度 (通常是 HuBERT Base/Large 的 768)
                 segment_dim: int = HUGO_XLARGE_DIM, # 教师模型处理的内部维度 (1024)
                 num_layers: int = 12,        # 使用 12 层 Transformer Encoder
                 num_heads: int = HUGO_XLARGE_HEADS,
                 max_segments: int = 1000,
                 dropout: float = 0.1):
        super().__init__()
        
        # 1. 输入维度投影层 (关键): 
        # 用于将 Segment Pooling 输出的特征 (e.g., 768D) 映射到 HuBERT XLarge 的内部维度 (1024D)。
        # 如果您的 HuBERT Backbone 已经是 1024D，这个层就是 nn.Identity() 或直接跳过。
        self.input_projection = nn.Linear(input_dim, segment_dim)
        
        # 2. 位置编码 (用于捕获片段的顺序信息)
        self.pos_encoder = nn.Embedding(max_segments, segment_dim) 
        self.dropout = nn.Dropout(dropout)
        
        # 3. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=segment_dim, 
            nhead=num_heads, 
            dim_feedforward=segment_dim * 4, # 默认 MLP 扩展系数
            dropout=dropout,
            batch_first=False # (T, B, D) 格式
        )
        
        # 4. Transformer Encoder (堆叠 num_layers 个层)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def load_xlarge_weights(self, xlarge_path: str = HUGE_XLARGE_PATH):
            """
            理由：实现 SegmentEncoder 的初始化，使用 HuBERT XLarge 的深层权重
            （第 6 到 17 层，共 12 层），提供强大的语义编码能力。
            """
            print(f"Initializing SegmentEncoder from HuBERT XLarge layers 6-17 at: {xlarge_path}...")
            
            try:
                # 1. 加载预训练的 HuBERT XLarge 模型 (仅用于提取权重)
                # 使用 local_files_only=True 确保从本地路径加载
                hubert_xlarge = HubertModel.from_pretrained(xlarge_path, local_files_only=True)

                # 2. 定义源层索引
                START_LAYER = 6
                END_LAYER = START_LAYER + self.transformer_encoder.num_layers # 6 + 12 = 18

                # 3. 遍历 SegmentEncoder 的层并复制权重
                for i in range(self.transformer_encoder.num_layers):
                    target_layer = self.transformer_encoder.layers[i]
                    
                    # 源层索引: 从 HuBERT 的第 6 层开始
                    source_layer_index = START_LAYER + i 
                    source_layer = hubert_xlarge.encoder.layer[source_layer_index] # 注意：HubertModel 使用 .layer 属性
                    
                    # 复制状态字典
                    target_layer.load_state_dict(source_layer.state_dict())
                
                print(f"SegmentEncoder 的 {self.transformer_encoder.num_layers} 层已成功使用 HuBERT XLarge 的 {START_LAYER} 到 {END_LAYER-1} 层权重初始化。")
                
                # 4. 删除 HuBERT 模型以释放内存
                del hubert_xlarge
                torch.cuda.empty_cache() # 释放 GPU 显存
                
            except Exception as e:
                print(f"FATAL ERROR: Failed to load XLarge weights from {xlarge_path}. Exception: {e}")
                print("Please ensure the HuggingFace HubertModel can load the local path.")
        
    def forward(self, segments: torch.Tensor, padding_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Args:
            segments: (T, B, D_in) - 填充后的片段序列，D_in 是输入维度 (e.g., 768)
            padding_mask: (B, T) - Transformer 的 Key Padding Mask (True=Masked)

        Returns:
            output: (T, B, D_seg) - 编码后的片段序列表示 (D_seg=1024)
        """
        time_steps, batch_size, dim_in = segments.shape
        device = segments.device
        
        # 1. 投影到教师模型维度 (768 -> 1024)
        x = self.input_projection(segments)
        
        # 2. 添加位置编码
        position_indices = torch.arange(time_steps, device=device) 
        pos_embedding = self.pos_encoder(position_indices).unsqueeze(1).expand(-1, batch_size, -1)
        
        x = x + pos_embedding
        x = self.dropout(x)
        
        # 3. 运行 Transformer 编码器
        # src_key_padding_mask: 告诉 Transformer 忽略哪些元素（填充部分）
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        return output