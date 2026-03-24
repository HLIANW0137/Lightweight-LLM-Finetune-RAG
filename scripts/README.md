# 🛠️ 自动化数据构建流水线 (Auto QA Data Pipeline)

在垂直领域的大模型微调（Fine-tuning）中，最耗时、成本最高的环节往往是**高质量指令数据的构建**。

本目录下的脚本提供了一套极简的自动化解决方案，能够利用云端大模型（如 Qwen, DeepSeek 等），将企业内部冗长、非结构化的规章制度（TXT/PDF），全自动清洗、抽取并转换为 LLaMA-Factory 所需的标准 `JSON` 问答对格式。

## ✨ 核心特性

* **多格式支持**：原生支持 `.txt` 和 `.pdf` 文件的文本抽取。
* **智能滑动切块 (Chunking)**：自动将长文档切分为适合 LLM 理解的小文本块，防止上下文溢出。
* **Prompt 工程**：内置经过优化的 Prompt 模板，有效控制生成数据的发散性，确保问答严格忠于原文（降低幻觉）。
* **格式容错提取**：自动使用正则表达式捕获大模型输出的 JSON 结构，防止由于 Markdown 标记导致的解析崩溃。

## 📦 依赖安装

在运行脚本前，请确保安装了以下轻量级依赖：

```bash
pip install openai tqdm pdfplumber
```

## 🚀 快速上手

### 1. 配置 API Key
打开 `auto_qa_generator.py`，在顶部配置你的大模型 API 密钥。推荐使用阿里云百炼平台（兼容 OpenAI 格式）或 DeepSeek，成本极低。
```python
API_KEY = "sk-你的API_KEY"
BASE_URL = "[https://dashscope.aliyuncs.com/compatible-mode/v1](https://dashscope.aliyuncs.com/compatible-mode/v1)" 
MODEL_NAME = "qwen-plus"
```

### 2. 准备原始数据
将你的企业文档重命名为 `sample_rules.txt`（或在代码中修改输入路径），放置在与脚本同级的目录下。

### 3. 一键生成数据集
在项目根目录（或本目录下）运行脚本：
```bash
python auto_qa_generator.py
```

终端会显示炫酷的进度条。运行结束后，当前目录下会生成一个 `my_custom_dataset.json` 文件。

## 📊 数据转换效果演示

**📥 输入 (非结构化文本)：**
> "报销制度：员工差旅费报销需在出差结束后 7 个工作日内通过 OA 系统提交，并附带增值税专用发票。"

**📤 输出 (微调标准 JSON)：**
```json
[
  {
    "instruction": "出差回来的报销流程有时间限制吗？",
    "input": "",
    "output": "有的。根据公司报销制度，员工差旅费报销必须在出差结束后的 7 个工作日内通过 OA 系统提交报销申请。"
  },
  {
    "instruction": "报销的时候对发票有什么具体要求？",
    "input": "",
    "output": "报销单必须附带合规的增值税专用发票。"
  }
]
```

生成的数据集可直接放入
