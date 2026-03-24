import os
import json
import re
from tqdm import tqdm
from openai import OpenAI
import pdfplumber

# ==========================================
# 1. 配置 API (推荐使用极度便宜/免费)
# ==========================================
# 这里以兼容 OpenAI 格式的任意国产大模型为例
API_KEY = "sk-你的API_KEY"
BASE_URL = "..." # 填写接口
MODEL_NAME = "qwen-plus" # 可以换成 deepseek-chat 等

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==========================================
# 2. 文本提取与切块 (Chunking) 模块
# ==========================================
def read_document(file_path):
    """读取 TXT 或 PDF 文件内容"""
    text = ""
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError("目前仅支持 .txt 或 .pdf 文件")
    return text

def chunk_text(text, chunk_size=500):
    """将长文本切分成适合大模型阅读的小块"""
    # 简单按字数切块，实际企业应用中可按段落或 \n\n 切割
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ==========================================
# 3. 核心：调用 LLM 自动生成 QA 对
# ==========================================
def generate_qa_pairs(text_chunk):
    """利用大模型从文本块中抽取 QA 对"""
    prompt = f"""你是一个专业的数据集构建专家。
请阅读以下[参考文本]，从中提取出核心知识点，并据此生成 3 到 5 个高质量的问答对。
要求：
1. 问题要符合人类提问的真实口吻。
2. 答案必须基于[参考文本]，准确无误。
3. 必须严格以下面的 JSON 数组格式输出，不要包含任何其他多余的解释文字或 Markdown 标记！

输出格式示例：
[
    {{"instruction": "这里写问题", "input": "", "output": "这里写答案"}},
    {{"instruction": "这里写另一个问题", "input": "", "output": "这里写另一个答案"}}
]

[参考文本]：
{text_chunk}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3 # 降低发散性，保证数据忠于原文
        )
        result_text = response.choices[0].message.content
        
        # 容错处理：用正则提取出被 LLM 乱加的 ```json 和 ``` 包裹的内容
        match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if match:
            qa_list = json.loads(match.group(0))
            return qa_list
        else:
            return []
    except Exception as e:
        print(f"\n生成失败: {e}")
        return []

# ==========================================
# 4. 主干流水线 (Pipeline)
# ==========================================
def main():
    input_file = "sample_rules.txt" # 你的原始文档路径
    output_file = "my_custom_dataset.json" # LLaMA-Factory 要求的输出文件
    
    # 为了测试，如果你没有 sample_rules.txt，我们自动创建一个
    if not os.path.exists(input_file):
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write("星辰科技有限公司报销管理办法：\n1. 员工差旅费报销需在出差结束后 7 个工作日内通过 OA 系统提交。\n2. 报销单必须附带增值税专用发票，普通发票不予报销。\n3. 乘坐高铁二等座及以下标准实报实销，一等座需总监级别以上审批。")
    
    print(f"📄 正在读取文档: {input_file}")
    raw_text = read_document(input_file)
    chunks = chunk_text(raw_text, chunk_size=300)
    print(f"✂️ 文档已切分为 {len(chunks)} 个文本块。")
    
    all_qa_data = []
    
    print("🤖 正在调用大模型生成问答数据...")
    for chunk in tqdm(chunks):
        if len(chunk.strip()) < 20: continue # 忽略太短的无意义文本块
        
        qa_pairs = generate_qa_pairs(chunk)
        all_qa_data.extend(qa_pairs)
        
    print(f"✅ 数据集生成完毕！共生成 {len(all_qa_data)} 条高质量 QA 数据。")
    
    # 保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_data, f, ensure_ascii=False, indent=4)
        
    print(f"💾 已保存至: {output_file}，可以直接丢进 LLaMA-Factory 训练啦！")

if __name__ == "__main__":
    main()
