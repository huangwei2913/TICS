import torch
from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------------------------------
# 1. 待验证的辅助函数：Token 到词级特征聚合
# --------------------------------------------------------------------------

def aggregate_tokens_to_words(E_token, word_ids):
    """
    基于 tokenizer.word_ids() 将 XLM-R 的 Subword Token 特征聚合到词级特征（求平均）。
    
    Args:
        E_token (torch.Tensor): 单一样本的 Token 特征序列，形状 [N_token, D_feat]。
        word_ids (list): 由 tokenizer.word_ids() 返回的列表，[None, 0, 0, 1, None, ...]
        
    Returns:
        torch.Tensor: 聚合后的词级特征序列，形状 [M, D_feat]。
    """
    
    if not E_token.ndim == 2:
        E_token = E_token.squeeze(0) # 确保处理 [1, N_token, D] 或 [N_token, D]
        
    word_features = []
    current_tokens = []
    current_word_index = None

    # 从第一个 Token 开始遍历，idx 是 Token 的索引
    for idx, word_idx in enumerate(word_ids):
        
        # 1. 忽略 [CLS], [SEP] 等特殊 Token (word_idx == None)
        if word_idx is None:
            # 如果遇到 None 且 current_tokens 有积累，说明是词语的结束，需要聚合
            if current_tokens:
                word_features.append(torch.stack(current_tokens).mean(dim=0))
                current_tokens = [] 
            current_word_index = None
            continue
            
        # 2. 如果是新的词语开始 (Word Index 发生变化)
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
        # 如果句子为空或只包含特殊 Token
        return torch.empty(0, E_token.size(1), dtype=E_token.dtype)

# --------------------------------------------------------------------------
# 2. 模型和数据设置
# --------------------------------------------------------------------------

# 使用 XLM-RoBERTa Large 作为测试模型
MODEL_NAME = "FacebookAI/xlm-roberta-large" 
TEST_TEXT = "Hello, XLM-RoBERTa 很强大! 人工智能 (AI) 正在改变世界。"
D_FEAT = 1024 # XLM-RoBERTa Large 的隐藏维度

print(f"--- 验证 XLM-RoBERTa 词级聚合函数 ---")
print(f"模型：{MODEL_NAME}")
print(f"测试文本：{TEST_TEXT}\n")

# 加载 Tokenizer 和 Model
try:
    # 强制不训练，冻结参数
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval() 
    for param in model.parameters():
        param.requires_grad = False
except Exception as e:
    print(f"无法加载模型，请检查网络或路径。错误：{e}")
    # 创建模拟数据进行测试（如果模型加载失败）
    tokenizer = None
    model = None


# --------------------------------------------------------------------------
# 3. 执行测试
# --------------------------------------------------------------------------

# 步骤 1: Tokenize 并获取 word_ids
inputs = tokenizer(TEST_TEXT, return_tensors="pt", truncation=True, return_offsets_mapping=True)
input_ids = inputs['input_ids']
word_ids = inputs.word_ids(0) # 获取第一个 batch item 的 word_ids

print(f"原始 Word IDs 序列长度 (N_token): {len(word_ids)}")

# 步骤 2: 运行模型，获取 Token 特征
if model:
    with torch.no_grad():
        outputs = model(input_ids)
        E_token = outputs.last_hidden_state.squeeze(0) # [N_token, D_feat]
else:
    # 如果模型加载失败，使用随机张量模拟特征
    N_token = len(word_ids)
    E_token = torch.randn(N_token, D_FEAT)
    print("注意：模型加载失败，使用随机特征进行维度检查。")


# 步骤 3: 确定期望的词汇数量 M
# 期望的词汇数量 M = 最大的 word_id + 1
valid_word_ids = [w for w in word_ids if w is not None]
M_expected = max(valid_word_ids) + 1 if valid_word_ids else 0
print(f"预期聚合后的词汇数量 (M): {M_expected}")


# 步骤 4: 调用聚合函数
E_text_true = aggregate_tokens_to_words(E_token, word_ids)


# --------------------------------------------------------------------------
# 4. 验证结果
# --------------------------------------------------------------------------

# 验证 1: 检查最终的词汇数量 M
M_actual = E_text_true.size(0)
print(f"\n--- 验证结果 ---")
print(f"实际聚合后的词汇数量 (M_actual): {M_actual}")
print(f"聚合后的特征形状 (M_actual, D_feat): {E_text_true.shape}")

# 验证 2: 检查词汇数量是否匹配
assert M_actual == M_expected, f"失败：期望词数 {M_expected}，实际词数 {M_actual}"
print(f"✅ 验证成功：词汇数量 M_actual ({M_actual}) 匹配 M_expected ({M_expected})。")

# 验证 3: 检查特征维度是否匹配
assert E_text_true.size(1) == D_FEAT, f"失败：期望特征维度 {D_FEAT}，实际维度 {E_text_true.size(1)}"
print(f"✅ 验证成功：特征维度 ({E_text_true.size(1)}) 匹配模型维度 ({D_FEAT})。")

# 验证 4: (高级检查) 验证 XLM-RoBERTa 的 subword 是否被正确聚合
# XLM-RoBERTa 会将 'XLM-RoBERTa' 切分成多个 token (如 XL, M, -, Ro, BER, Ta) 
# 我们检查第一个词（Hello）和第二个词（XLM-RoBERTa）的特征是否一致
word_ids_clean = [w for w in word_ids if w is not None]

if M_expected > 1 and model:
    # 找到第一个词 'Hello' 对应的 token indices
    idx_word_0_tokens = [i for i, w in enumerate(word_ids) if w == 0]
    # 找到第二个词 'XLM-RoBERTa' 对应的所有 token indices (它被切成了很多 subwords)
    idx_word_1_tokens = [i for i, w in enumerate(word_ids) if w == 1]
    
    # 手动计算聚合均值 (第一个词)
    manual_mean_0 = E_token[idx_word_0_tokens].mean(dim=0)
    # 比较聚合函数的结果和手动计算结果
    if torch.allclose(E_text_true[0], manual_mean_0, atol=1e-6):
        print(f"✅ 验证成功：第 0 个词 'Hello' 的特征聚合结果正确。")
    else:
        print(f"❌ 验证失败：第 0 个词的聚合结果不匹配手动计算均值。")

    # 手动计算聚合均值 (第二个词，即 XLM-RoBERTa)
    manual_mean_1 = E_token[idx_word_1_tokens].mean(dim=0)
    if torch.allclose(E_text_true[1], manual_mean_1, atol=1e-6):
        print(f"✅ 验证成功：第 1 个词 'XLM-RoBERTa' 的多 Token 聚合结果正确。")
    else:
        print(f"❌ 验证失败：第 1 个词的聚合结果不匹配手动计算均值。")

print("\n辅助函数验证完毕。如果所有 '✅' 通过，该函数可以安全地用于 Collate Function。")
