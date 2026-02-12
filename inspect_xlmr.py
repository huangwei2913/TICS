import torch
from transformers import XLMRobertaTokenizer
# 导入你刚刚定义的类，或者直接把类定义放在脚本里
# from your_module import XLMRobertaFusionExpert
import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class XLMRobertaFusionExpert(nn.Module):
    def __init__(self, xlmr_path, target_dim=1024):
        super().__init__()
        # 加载 768 维的 Base 模型
        self.model = XLMRobertaModel.from_pretrained(xlmr_path, output_hidden_states=True)
        # 冻结
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 挑选 4 层进行融合 (索引对应 hidden_states 的 [4, 7, 10, 13])
        self.layers_to_use = [3, 6, 9, 12] 
        
        # 1. 维度对齐层：从 768 升维到 1024
        # 这样文本导师输出的特征就能直接和语音侧的 1024 维进行 Loss 计算
        self.dim_alignment = nn.Sequential(
            nn.Linear(768, target_dim),
            nn.LayerNorm(target_dim)
        )
        
        # 2. 跨层融合权重
        self.layer_weights = nn.Parameter(torch.ones(len(self.layers_to_use)))
        
        # 3. 最终全局映射 (1024 -> 1024)
        self.global_head = nn.Sequential(
            nn.Linear(target_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # all_layers 长度为 13 (Embedding + 12 layers)
            all_layers = outputs.hidden_states
            
            # 提取目标层的 [CLS] token
            # hidden_states[i] 形状: [B, T, 768] -> [:, 0, :] 得到 [B, 768]
            cls_list = [all_layers[i][:, 0, :] for i in self.layers_to_use]
            stacked_cls = torch.stack(cls_list, dim=0) # [4, B, 768]

        # 层加权融合
        weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1)
        fused_768 = (stacked_cls * weights).sum(dim=0) # [B, 768]
        
        # 升维到 1024
        fused_1024 = self.dim_alignment(fused_768) # [B, 1024]
        
        # 最终全局特征
        text_global_summary = self.global_head(fused_1024) # [B, 1024]
        
        return text_global_summary
def test_xlmr_expert():
    model_path = "/mnt/conda_data/facebook/xlm-roberta-base"
    target_dim = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Initializing XLMRobertaFusionExpert on {device}...")
    
    # 1. 实例化模型
    expert = XLMRobertaFusionExpert(xlmr_path=model_path, target_dim=target_dim).to(device)
    expert.eval()

    # 2. 准备 tokenizer 和模拟数据
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    text_batch = ["Hello, this is a semantic test.", "This is a second longer sentence for testing multi-sentence logic."]
    
    # 对文本进行编码
    inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
    
    print(f"Input IDs shape: {inputs['input_ids'].shape}") # [Batch, Seq_Len]

    # 3. 前向传播
    print("\n--- Running Forward Pass ---")
    try:
        with torch.no_grad():
            global_summary = expert(inputs['input_ids'], inputs['attention_mask'])
        
        print(f"✅ Success! Output shape: {global_summary.shape}")
        
        # 验证维度
        assert global_summary.shape == (len(text_batch), target_dim), \
            f"Dimension mismatch! Expected {(len(text_batch), target_dim)}, got {global_summary.shape}"
        print(f"✅ Dimension Verified: {target_dim}")

    except Exception as e:
        print(f"❌ Forward Pass Failed: {e}")
        return

    # 4. 验证参数冻结状态 (学术严谨性)
    print("\n--- Parameter Freeze Check ---")
    frozen_ok = True
    for name, param in expert.model.named_parameters():
        if param.requires_grad:
            print(f"⚠️ Warning: Parameter {name} is NOT frozen!")
            frozen_ok = False
            break
    if frozen_ok:
        print("✅ XLM-R Backbone is correctly frozen.")

    # 5. 验证可学习参数 (融合权重和投影层)
    print("\n--- Trainable Parameters Check ---")
    trainable_params = [n for n, p in expert.named_parameters() if p.requires_grad]
    print(f"Trainable components: {trainable_params}")
    if 'layer_weights' in trainable_params:
        print("✅ Layer weights are learnable.")
    if 'dim_alignment.0.weight' in trainable_params:
        print("✅ Dimension alignment layer is learnable.")

    print("\n🎉 Test Completed Successfully!")

if __name__ == "__main__":
    # 如果你把类定义在别的文件，记得先 import
    # 这里为了演示，假设类已经在当前命名空间
    test_xlmr_expert()