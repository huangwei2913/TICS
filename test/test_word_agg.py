import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# --------------------------------------------------------------------------
# 1. 待验证的辅助函数：Token 到词级特征聚合
# --------------------------------------------------------------------------

def aggregate_tokens_to_words(E_token, word_ids):
    """
    基于 tokenizer.word_ids() 将 XLM-R 的 Subword Token 特征聚合到词级特征（求平均）。
    """
    
    # 确保 E_token 维度是 [N_token, D_feat]
    if E_token.ndim == 3:
        E_token = E_token.squeeze(0) 
    elif E_token.ndim != 2:
        raise ValueError("E_token 维度必须是 [N_token, D_feat]")
        
    word_features = []
    current_tokens = []
    current_word_index = None

    # 从第一个 Token 开始遍历
    for idx, word_idx in enumerate(word_ids):
        
        # 1. 忽略 [CLS], [SEP] 等特殊 Token (word_idx == None)
        if word_idx is None:
            # 如果遇到 None 且 current_tokens 有积累，说明是词语的结束，需要聚合
            if current_tokens:
                word_features.append(torch.stack(current_tokens).mean(dim=0))
                current_tokens = [] 
            current_word_index = None
            continue
            
        # 2. 遇到了新词的开始
        if word_idx != current_word_index and current_word_index is not None:
            # 聚合并保存上一个词的特征
            if current_tokens:
                word_features.append(torch.stack(current_tokens).mean(dim=0))
            
            # 开始积累新词的 Tokens
            current_tokens = [E_token[idx]]
            current_word_index = word_idx
            
        # 3. 仍是同一个词，或新序列的第一个词 (积累 Tokens)
        else:
            current_tokens.append(E_token[idx])
            current_word_index = word_idx

    # 循环结束后，处理最后一个词
    if current_tokens:
        word_features.append(torch.stack(current_tokens).mean(dim=0))
        
    # 返回聚合后的张量
    if word_features:
        return torch.stack(word_features)
    else:
        return torch.empty(0, E_token.size(1), dtype=E_token.dtype)

# --------------------------------------------------------------------------
# 2. 模型和数据设置 (使用用户提供的本地路径)
# --------------------------------------------------------------------------

LOCAL_PATH = "/mnt/facebook/xlm-roberta-large" 
TEST_TEXT = "Hello, XLM-RoBERTa 很强大! 人工智能 (AI) 正在改变世界。"
# XLM-RoBERTa Large 的隐藏维度
D_FEAT = 1024 

print(f"--- 验证 XLM-RoBERTa 词级聚合函数 ---")
print(f"使用的本地路径：{LOCAL_PATH}")
print(f"测试文本：{TEST_TEXT}\n")

# 加载 Tokenizer 和 Model，使用 local_files_only=True
try:
    # 强制不训练，冻结参数
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(LOCAL_PATH, local_files_only=True).eval() 
    for param in model.parameters():
        param.requires_grad = False
    model_loaded = True
except Exception as e:
    print(f"❌ 错误：无法从本地路径加载模型。请确认路径和文件完整性。错误信息：{e}")
    model_loaded = False
    # 如果加载失败，设置模拟数据进行维度检查
    # 无法进行特征值精确对比，但可检查维度和数量
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large") # 确保至少能tokenize
    
    
# --------------------------------------------------------------------------
# 3. 执行测试
# --------------------------------------------------------------------------

# 步骤 1: Tokenize 并获取 word_ids
inputs = tokenizer(TEST_TEXT, return_tensors="pt", truncation=True, return_offsets_mapping=True)
input_ids = inputs['input_ids']
word_ids = inputs.word_ids(0) # 获取第一个 batch item 的 word_ids

print(f"原始 Word IDs 序列长度 (N_token): {len(word_ids)}")  #24
print(f"Word IDs 序列: {word_ids}") # 词对应的ID


# 步骤 2: 运行模型，获取 Token 特征
if model_loaded:
    with torch.no_grad():
        outputs = model(input_ids)
        E_token = outputs.last_hidden_state.squeeze(0) # [N_token, D_feat]  
else:
    # 使用随机张量模拟特征
    N_token = len(word_ids)
    E_token = torch.randn(N_token, D_FEAT)
    print("注意：模型未加载，使用随机特征进行维度检查。")


# 步骤 3: 确定期望的词汇数量 M (最大 word_id + 1)
valid_word_ids = [w for w in word_ids if w is not None]
M_expected = max(valid_word_ids) + 1 if valid_word_ids else 0
print(f"预期聚合后的词汇数量 (M): {M_expected}")  #词汇总数量 6


# 步骤 4: 调用聚合函数
E_text_true = aggregate_tokens_to_words(E_token, word_ids)


# --------------------------------------------------------------------------
# 4. 验证结果
# --------------------------------------------------------------------------

print(f"\n--- 验证结果 ---")

# 验证 1: 检查最终的词汇数量 M
M_actual = E_text_true.size(0)
print(f"实际聚合后的词汇数量 (M_actual): {M_actual}")
print(f"聚合后的特征形状 (M_actual, D_feat): {E_text_true.shape}")

# 验证 2: 检查词汇数量是否匹配
try:
    assert M_actual == M_expected
    print(f"✅ 验证成功：词汇数量 M_actual ({M_actual}) 匹配 M_expected ({M_expected})。")
except AssertionError:
    print(f"❌ 验证失败：期望词数 {M_expected}，实际词数 {M_actual}")


# 验证 3: 检查特征维度是否匹配
try:
    assert E_text_true.size(1) == D_FEAT
    print(f"✅ 验证成功：特征维度 ({E_text_true.size(1)}) 匹配模型维度 ({D_FEAT})。")
except AssertionError:
    print(f"❌ 验证失败：期望特征维度 {D_FEAT}，实际维度 {E_text_true.size(1)}")


# 验证 4: (高级检查) 验证聚合均值的精确性 (仅在模型成功加载时运行)
if model_loaded and M_expected > 1:
    print("\n--- 聚合精度检查 ---")
    
    # 遍历所有词汇索引 (0 到 M_expected-1)
    for word_idx_to_check in range(M_expected):
        # 1. 找到该词对应的所有 Token 索引
        idx_tokens = [i for i, w in enumerate(word_ids) if w == word_idx_to_check]
        
        # 2. 手动计算这些 Token 的均值
        manual_mean = E_token[idx_tokens].mean(dim=0)
        
        # 3. 比较聚合函数的结果和手动计算结果
        if torch.allclose(E_text_true[word_idx_to_check], manual_mean, atol=1e-6):
            print(f"✅ 词汇索引 {word_idx_to_check} 的特征聚合结果正确。")
        else:
            print(f"❌ 词汇索引 {word_idx_to_check} 的聚合结果不匹配手动计算均值。")


print("\n辅助函数验证完毕。")