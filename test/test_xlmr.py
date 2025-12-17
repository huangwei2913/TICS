import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer

def test_xlmr_features(model_path: str):
    print(f"正在从本地加载 XLM-R 模型: {model_path} ...")
    
    # 1. 加载 Tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = XLMRobertaModel.from_pretrained(model_path)
    model.eval()

    # --- 新增：确认 Padding ID 的逻辑 ---
    print("\n" + "="*30)
    print("--- Tokenizer 特殊 Token 确认 ---")
    print(f"Padding Token: '{tokenizer.pad_token}'")
    print(f"Padding Token ID: {tokenizer.pad_token_id}")
    print(f"BOS Token ID: {tokenizer.bos_token_id}")
    print(f"EOS Token ID: {tokenizer.eos_token_id}")
    print("="*30 + "\n")

    # 2. 准备一段典型的测试文本
    test_text = "他方今天给纯爱家是暂时加了很多条件。只要你对"
    
    # 3. 分词测试
    inputs = tokenizer(test_text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print("--- Token 序列测试 ---")
    print(f"Tokens: {tokens}")
    print(f"IDs: {inputs['input_ids'][0].tolist()}")

    # 4. 模拟 Padding 行为
    print("\n--- Padding 行为验证 ---")
    # 假设我们有两条长度不一的句子
    sentences = ["短句子", "这是一个非常长的句子测试"]
    batch_inputs = tokenizer(sentences, padding=True, return_tensors="pt")
    print(f"Batch IDs 矩阵:\n{batch_inputs['input_ids']}")
    # 观察矩阵中填充的数字是否就是上面的 pad_token_id

if __name__ == "__main__":
    LOCAL_XLMR_PATH = "/mnt/facebook/xlm-roberta-large"
    test_xlmr_features(LOCAL_XLMR_PATH)