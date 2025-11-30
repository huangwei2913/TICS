from transformers import HubertModel
import torch
import os

HUGE_XLARGE_PATH = "/mnt/facebook/hubert-xlarge-ls960-ft"

# 确保路径存在
if not os.path.exists(HUGE_XLARGE_PATH):
    print(f"Error: Path {HUGE_XLARGE_PATH} does not exist. Please check the path.")
    
try:
    # 尝试加载模型
    print("Loading HuBERT XLarge model weights...")
    hubert_xlarge = HubertModel.from_pretrained(HUGE_XLARGE_PATH, local_files_only=True)
    
    # 获取模型的状态字典
    state_dict = hubert_xlarge.state_dict()
    
    print("-" * 50)
    print(f"Success: HuBERT XLarge model loaded ({len(state_dict)} weight keys)")
    print("Encoder layers shapes (Key: Shape):")
    print("-" * 50)

    # 遍历并打印所有权重键和其形状
    for name, tensor in state_dict.items():
        # 重点关注 HuBERT 的编码器层
        if "encoder.layers" in name:
            print(f"{name}: {tuple(tensor.shape)}")
    
    print("\n--- Top and bottom layer shapes (reference) ---")
    
    # 打印一些非 Encoder 层的参考信息
    print(f"feature_extractor.conv_layers.0.conv.weight: {state_dict['feature_extractor.conv_layers.0.conv.weight'].shape}")
    print(f"encoder.pos_conv.0.weight: {state_dict['encoder.pos_conv.0.weight'].shape}")
    
except Exception as e:
    print("-" * 50)
    print(f"Warning: Failed to load full model structure with from_pretrained.")
    print(f"Error: {e}")
    print("Trying to load raw weight file...")

    # 如果 from_pretrained 失败，尝试直接加载权重文件
    try:
        weight_file = os.path.join(HUGE_XLARGE_PATH, "pytorch_model.bin")
        if not os.path.exists(weight_file):
            print(f"Error: Weight file not found {weight_file}")
            raise FileNotFoundError

        state_dict_raw = torch.load(weight_file, map_location='cpu')
        
        print("-" * 50)
        print(f"Success: Raw weight file loaded ({len(state_dict_raw)} weight keys)")
        print("Encoder layers shapes (Key: Shape):")
        print("-" * 50)

        for name, tensor in state_dict_raw.items():
            if "encoder.layers" in name:
                print(f"{name}: {tuple(tensor.shape)}")
                
    except Exception as e_raw:
        print(f"Failed to load raw weight file. Error: {e_raw}")

