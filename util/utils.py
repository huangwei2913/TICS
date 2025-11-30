import math
from timm.models.vision_transformer import  Mlp, Block , DropPath
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Type
#使用cross Atttention模块
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ==============================================================================
# 2. CrossAttentionBlock 模块 (核心修改)
# ==============================================================================
class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer: Type[nn.Module] = nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        
        # --- 修正点 1: 确保 LayerNorm 输入类型正确 ---
        # 在 DeepSpeed FP16 模式下，x 和 norm1 的权重应为 torch.float16。
        # 但 LayerNorm 的 F.layer_norm 偶尔会默认期望 FP32。
        # 强制将输入张量 x 转换为 LayerNorm 权重的 dtype (通常为 FP16)
        
        # 步骤 1: 确保 x 的类型与 norm1 的权重类型一致 (通常是 FP16)
        # 获取当前模型运行的 dtype，如果模型在 GPU 上，norm1.weight.dtype 应该是 torch.float16
        current_dtype = self.norm1.weight.dtype
        x_norm_input = x.to(current_dtype) 
        
        # 原计算：x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        
        # 新计算：使用类型转换后的输入进行 LayerNorm
        attn_output = self.drop_path(self.attn(self.norm1(x_norm_input)))
        
        # 由于 CrossAttention 只返回 Bx1xC 的张量，而输入 x 是 BxNxC
        # 这里的残差连接应该使用原始 x 的 CLS token (x[:, 0:1, ...])
        # 注意：attn_output 已经是 Bx1xC
        x = x[:, 0:1, ...] + attn_output
        
        # --- 修正点 2: MLP 块 (如果存在) ---
        if self.has_mlp:
            # 确保 MLP 输入也是正确的类型 (虽然 DeepSpeed 通常会处理)
            x_mlp_input = x.to(self.norm2.weight.dtype)
            x = x + self.drop_path(self.mlp(self.norm2(x_mlp_input)))

        return x


import torch
import os
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

# 定义常量
START_LAYER = 6
END_LAYER = 17

def load_large_weights_tolerant(target_model, large_checkpoint_path, start_layer=START_LAYER, end_layer=END_LAYER):
    """
    容错加载 HuBERT Large (D=1024) 权重的指定层 (6-17) 到 TICS SegmentEncoder (D=1024)。
    此函数使用鲁棒的键匹配逻辑，修复布尔值歧义错误。
    """
    if not os.path.exists(large_checkpoint_path):
        logging.error(f"❌ 错误: HuBERT Large 路径不存在: {large_checkpoint_path}")
        return

    # --- 1. 加载源权重 ---
    try:
        weight_file = os.path.join(large_checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(weight_file):
             weight_file = large_checkpoint_path
             if not os.path.exists(large_checkpoint_path): # 修复: 检查路径是否存在
                 raise FileNotFoundError(f"未找到 HuBERT Large 权重文件: {large_checkpoint_path}")

        source_state_dict = torch.load(weight_file, map_location='cpu')
        if 'model' in source_state_dict:
            source_state_dict = source_state_dict['model']

        logging.info(f"✅ 成功加载 HuBERT Large 检查点。总权重键数: {len(source_state_dict)}")
    except Exception as e:
        logging.error(f"❌ 致命错误: 无法加载或解析 HuBERT Large 权重文件。错误: {e}")
        return

    # --- 2. 遍历并容错复制 ---
    target_state_dict = target_model.state_dict()
    copied_count = 0
    skipped_keys = []
    
    # 临时存储QKV权重，用于连接 (按层存储)
    qkv_weights_map = defaultdict(lambda: {'q_w': None, 'k_w': None, 'v_w': None})
    qkv_biases_map = defaultdict(lambda: {'q_b': None, 'k_b': None, 'v_b': None})
    
    # HuBERT Large 的层索引范围 (6 到 17)
    for source_idx in range(start_layer, end_layer + 1):
        target_idx = source_idx - start_layer  # 目标 SegmentEncoder 的层索引 (0 到 11)
        
        # 定义目标键的前缀
        target_prefix = f"transformer_encoder.layers.{target_idx}."
        layer_idx_str = f"layers.{source_idx}." # 匹配源键中的 layers.X.

        # 遍历源模型中的所有键
        for name, tensor in source_state_dict.items():
            
            # 鲁棒匹配
            if layer_idx_str in name and ('proj' in name or 'dense' in name or 'norm' in name):

                start_index = name.find(layer_idx_str) + len(layer_idx_str)
                suffix = name[start_index:]
                target_key = None
                
                # --- 核心键名映射逻辑 (无需切片) ---
                
                # QKV 权重需要特殊的处理
                if suffix == 'attention.q_proj.weight':
                    qkv_weights_map[target_idx]['q_w'] = tensor
                    continue
                elif suffix == 'attention.k_proj.weight':
                    qkv_weights_map[target_idx]['k_w'] = tensor
                    continue
                elif suffix == 'attention.v_proj.weight':
                    qkv_weights_map[target_idx]['v_w'] = tensor
                    continue
                elif suffix == 'attention.q_proj.bias':
                    qkv_biases_map[target_idx]['q_b'] = tensor
                    continue
                elif suffix == 'attention.k_proj.bias':
                    qkv_biases_map[target_idx]['k_b'] = tensor
                    continue
                elif suffix == 'attention.v_proj.bias':
                    qkv_biases_map[target_idx]['v_b'] = tensor
                    continue

                # FFN 映射
                elif suffix == 'feed_forward.intermediate_dense.weight':
                    target_key = target_prefix + 'linear1.weight'
                elif suffix == 'feed_forward.intermediate_dense.bias':
                    target_key = target_prefix + 'linear1.bias'
                elif suffix == 'feed_forward.output_dense.weight':
                    target_key = target_prefix + 'linear2.weight'
                elif suffix == 'feed_forward.output_dense.bias':
                    target_key = target_prefix + 'linear2.bias'
                
                # Attention 和 Norm 映射
                elif suffix == 'attention.out_proj.weight':
                    target_key = target_prefix + 'self_attn.out_proj.weight'
                elif suffix == 'attention.out_proj.bias':
                    target_key = target_prefix + 'self_attn.out_proj.bias'
                elif suffix == 'layer_norm.weight':
                    target_key = target_prefix + 'norm1.weight'
                elif suffix == 'layer_norm.bias':
                    target_key = target_prefix + 'norm1.bias'
                elif suffix == 'final_layer_norm.weight':
                    target_key = target_prefix + 'norm2.weight'
                elif suffix == 'final_layer_norm.bias':
                    target_key = target_prefix + 'norm2.bias'
                else:
                    skipped_keys.append(f"UNMAPPED_{name}")
                    continue

                # 执行直接复制 (维度已匹配)
                if target_key in target_state_dict:
                    try:
                        target_state_dict[target_key].copy_(tensor)
                        copied_count += 1
                    except Exception as e:
                        skipped_keys.append(f"RUNTIME_ERROR_{name} -> {target_key}: {e}")
                        continue
                
    # --- 3. 处理 QKV 权重连接 (HuBERT Q,K,V -> TICS in_proj) ---
    qkv_copied_count = 0
    for target_idx in range(0, end_layer - start_layer + 1):
        qkv_w = qkv_weights_map[target_idx]
        qkv_b = qkv_biases_map[target_idx]
        
        target_w_key = f"transformer_encoder.layers.{target_idx}.self_attn.in_proj_weight"
        target_b_key = f"transformer_encoder.layers.{target_idx}.self_attn.in_proj_bias"
        
        # ⚠️ 修复后的检查: 显式检查所有六个张量是否都不是 None
        if (qkv_w['q_w'] is not None and qkv_w['k_w'] is not None and qkv_w['v_w'] is not None and 
            qkv_b['q_b'] is not None and qkv_b['k_b'] is not None and qkv_b['v_b'] is not None):
            
            # 权重和偏置的连接顺序必须是 Q, K, V
            combined_weight = torch.cat([qkv_w['q_w'], qkv_w['k_w'], qkv_w['v_w']], dim=0)
            combined_bias = torch.cat([qkv_b['q_b'], qkv_b['k_b'], qkv_b['v_b']], dim=0)

            if target_w_key in target_state_dict and target_b_key in target_state_dict:
                target_state_dict[target_w_key].copy_(combined_weight)
                target_state_dict[target_b_key].copy_(combined_bias)
                qkv_copied_count += 2
            else:
                logging.error(f"❌ 目标键 {target_w_key} 或 {target_b_key} 在 SegmentEncoder 中未找到。")
        else:
             # 如果 QKV 缺失，打印警告信息
             logging.warning(f"⚠️ Layer {target_idx} (Source Layer {target_idx + start_layer}): 缺少 QKV 权重，无法连接。")


    # --- 4. 最终加载到 SegmentEncoder ---
    total_copied = copied_count + qkv_copied_count
    
    # 路径更新为 HuBERT Large 的新路径
    target_model.load_state_dict(target_state_dict, strict=False)

    logging.info("-" * 50)
    logging.info(f"✅ SegmentEncoder 权重加载完成（使用 HuBERT Large）。")
    logging.info(f"   成功复制非QKV键数量: {copied_count}")
    logging.info(f"   成功复制QKV键数量: {qkv_copied_count}")
    logging.info(f"   总计复制键数量: {total_copied}")
    logging.info("-" * 50)

# 确保在调用该函数时，将 LARGE_PATH 传入：
# load_large_weights_tolerant(segment_encoder, '/mnt/facebook/hubert-large-ls960-ft')
# --- 示例调用（需要在 TICS 源码中替换原有的加载逻辑） ---
# 
# if __name__ == '__main__':
#     # 假设 SegmentEncoder 类已定义，且实例 segment_encoder 已创建
#     # from models.segment_encoder import SegmentEncoder 
#     # segment_encoder = SegmentEncoder(...)
#     
#     # 替换为您的 HuBERT XLarge 路径
#     HUGE_XLARGE_PATH = "/mnt/facebook/hubert-xlarge-ls960-ft" 
#     
#     # 假设 target_model 已经被实例化
#     # load_xlarge_weights_tolerant(segment_encoder, HUGE_XLARGE_PATH)
#     pass