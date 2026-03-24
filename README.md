# Lightweight-LLM-Finetune-RAG
A Complete Guide to Fine-Tuning and RAG Deployment Based on a Single  GPU
# 🚀 本地轻量级大模型全链路实战：从微调到 RAG 部署的保姆级指南

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![LLaMA-Factory](https://img.shields.io/badge/LLaMA--Factory-WebUI-blue?style=flat-square)](https://github.com/hiyouga/LLaMA-Factory)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

本项目提供了一套完整的、基于单张消费级显卡的大模型本地落地解决方案。涵盖了从底层环境配置、LoRA 指令微调，到最终结合 LangChain + FAISS + FastAPI 实现 RAG（检索增强生成）问答系统的全流程。

无论你是手持 8GB 显存的入门级游戏本，还是拥有 24GB 显存的 RTX 4090，亦或是刚刚入手最新 **RTX 50 系 (Blackwell 架构)** 的极客玩家，都能在这里找到最适合你的“黄金配置”。

## 🌟 核心亮点

* **🔥 RTX 50 系避坑指南**：独家解决 Blackwell 架构运行 PyTorch 时的 `CUDA error: no kernel image` 崩溃问题。
* **💻 全主流显卡制霸**：针对 8GB / 12GB / 16GB / 24GB 不同显存梯度，提供最优的微调参数组合。
* **🧠 参数白话文解析**：拒绝黑盒调参，详细拆解 Batch Size、Gradient Accumulation、LoRA Rank 等晦涩概念的实际作用。
* **🛠️ 纯净版防爆显存方案**：不依赖容易报错的第三方插件，用最原生的方式榨干显卡性能。

---

## 💻 硬件与系统要求

* **操作系统**: Windows 11 + WSL2 (Ubuntu 22.04 / 24.04) 或 原生 Linux。强烈建议在 WSL2/Linux 中进行，避免 Windows 下复杂的 C++ 编译冲突。
* **内存 (RAM)**: 32GB 推荐（最低 16GB）。
* **Python 版本**: 3.10 或 3.11。

### 🎮 显卡分级与模型推荐 (最优解)

| 显存大小 | 代表显卡 | 推荐模型规模 | 微调策略 (最优解) |
| :--- | :--- | :--- | :--- |
| **8GB** | RTX 3060 / 4060 | 0.5B - 1.5B | 必须开启 4-bit / 8-bit 量化，极小 Batch Size。 |
| **12GB** | RTX 4070 / 5070 Ti | 1.5B - 3B | 纯净版 bf16 微调，无需量化，截断长度控制在 1024 内。 |
| **16GB** | RTX 4080 | 7B - 8B | 开启 4-bit 量化可微调 8B (如 Llama3-8B)，或纯净微调 3B。 |
| **24GB** | RTX 3090 / 4090 | 7B - 14B | 全量 bf16 微调 8B，开启 Flash Attention 2 提速，支持长文本。 |

---

## 🛠️ 第一阶段：环境搭建与破除“新显卡魔咒”

### 1. 配置基础 Linux 环境 (WSL2)

# 1. 下载并静默安装 Miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```
# 2. 刷新环境变量并创建独立 Python 3.11 环境
```
source ~/.bashrc
```
```
conda create -n llm_env python=3.11 -y
```
```
conda activate llm_env
```

### 2. 安装 PyTorch 引擎 (⚠️ 关键)

请根据你的显卡架构，选择**其中一种**方式安装：

#### 方案 A：主流显卡 (RTX 30系 / 40系)
稳定、成熟，直接使用官方正式版即可：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 方案 B：最新显卡 (RTX 50系 Blackwell 架构) 🛑 避坑必看！
如果你使用的是最新架构的显卡（sm_120），官方稳定版 (cu121/cu124) 尚未内置 50 系显卡的计算图纸，强制运行会导致底层算子崩溃。
* **前置条件**：必须在 Windows 物理机更新最新版 **NVIDIA Studio 驱动**，以支持底层的 PTX JIT 即时编译。
* **安装指令**：直接使用 PyTorch Nightly 版本的 cu128 完美点亮新显卡，并跳过容易引起依赖冲突的 vision/audio 库。
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
```

✅ **验证 GPU 是否点亮（测试）**：
```bash
python -c "import torch; a = torch.tensor([1.0]).cuda(); print('\n🎉 成了！GPU能算数了:', a, '\n')"
```
如果成功打印出 `tensor([1.], device='cuda:0')`，则说明底层封印彻底解除！

### 3. 安装 LLaMA-Factory
目前开源界最好用、最成熟的零代码微调框架。
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[metrics]
```

---

## 🔥 第二阶段：LoRA 微调与核心参数解密

启动 LLaMA-Factory 的 WebUI 界面：
```bash
llamafactory-cli webui
```
打开浏览器访问 `http://localhost:7860`。

### 🧠 核心微调参数大解密（知其然，知其所以然）

在 WebUI 中，你会看到大量参数。不要慌，理解以下几个核心参数，你就能驾驭任何显卡：

1. **截断长度 (Cutoff Length)**
   * **作用**：模型一次能处理的最大文本长度（Token数）。
   * **影响**：**显存杀手！** 长度翻倍，显存占用呈指数级增长。12GB 显存建议设为 `512` 或 `1024`；24GB 显存可尝试 `4096`。
2. **批处理大小 (Batch Size)**
   * **作用**：显卡一次性并行处理的数据条数。
   * **影响**：设得越大，训练越快，但也越容易爆显存（OOM）。小显存通常只能设为 `1` 或 `2`。
3. **梯度累加 (Gradient Accumulation)**
   * **作用**：显存不够时的“作弊神器”。如果 Batch Size 只能开到 1，但你想达到 Batch Size=8 的训练效果，就可以把梯度累加设为 8。模型会计算 8 次再更新一次权重。
   * **公式**：`真实 Batch Size = Batch Size × Gradient Accumulation`。
4. **计算类型 (Compute Type: fp16 / bf16)**
   * **作用**：数据在显卡中的精度格式。
   * **建议**：RTX 30/40/50 系显卡**强烈建议选择 `bf16`**。它比传统的 `fp16` 更不容易出现数值溢出（Loss 变成 NaN），且能省下一半显存。
5. **量化等级 (Quantization: 4-bit / 8-bit / none)**
   * **作用**：通过压缩模型权重来大幅降低显存占用（代价是轻微的智商下降）。
   * **避坑**：**RTX 50 系新显卡目前跑量化极易报错，请保持 `none`！** 8GB 显存的老显卡如果想跑 7B 模型，则必须选 `4-bit`。
6. **加速方式 (Flash Attention 2)**
   * **作用**：NVIDIA 推出的一种底层加速算法，能大幅降低长文本的显存占用并提速。
   * **避坑**：RTX 30/40 系推荐开启（需额外 `pip install flash-attn`）；**RTX 50 系请选 `auto` 或禁用**，防止新显卡缺少底层库导致崩溃。
7. **LoRA 秩 (LoRA Rank / Alpha)**
   * **作用**：Rank 决定了微调时新增参数的多少。Rank 越大，模型能学到的新知识细节越多，但也更容易过拟合。
   * **建议**：常规微调设为 `8` 或 `16`；如果是复杂的逻辑注入，可设为 `32` 或 `64`。Alpha 通常设为 Rank 的 2 倍。


---

### 🏆 各显卡极限微调“黄金配置表”

#### 方案一：12GB 显存极限保命流 (以 RTX 5070 Ti / 4070 为例)
* **目标模型**：Qwen2.5-1.5B-Instruct 或 3B 模型
* **核心思路**：纯净版微调，不借用量化，靠参数控制显存。

| 配置项 | 推荐设定值 | 核心原因与避坑说明 |
| :--- | :--- | :--- |
| **截断长度** | `1024` 或 `512` | 防爆显存核心。长文本是吃显存的大户。 |
| **Batch Size** | `2` (或 `1`) | 配合梯度累加使用。 |
| **梯度累加** | `4` (或 `8`) | 变相实现大 Batch Size (如 2*4=8)，保证收敛。 |
| **计算类型** | `bf16` | 防爆显存核心，比 fp16 更稳。 |
| **加速方式** | `auto` | 50系禁用 flash_attn2，30/40系可开启。 |
| **量化等级** | `none` | 50系跑 4/8-bit 极易报错，保持 none。 |

#### 方案二：8GB 显存丐帮逆袭流 (以 RTX 4060 / 3060 为例)
* **目标模型**：Llama-3-8B 或 Qwen2.5-7B
* **核心思路**：必须依赖 4-bit 量化（需安装 `bitsandbytes`），牺牲极小精度换取大模型运行权。

| 配置项 | 推荐设定值 | 核心原因与避坑说明 |
| :--- | :--- | :--- |
| **量化等级** | `4-bit` | 8GB 跑 7B 模型的唯一出路。 |
| **截断长度** | `512` | 显存极度紧张，文本必须短。 |
| **Batch Size** | `1` | 只能设为 1。 |
| **梯度累加** | `16` | 弥补 BS=1 带来的梯度震荡。 |

#### 方案三：24GB 显存土豪满血流 (以 RTX 4090 / 3090 为例)
* **目标模型**：Qwen2.5-14B 或 Llama-3-8B
* **核心思路**：火力全开，拉长上下文，开启 Flash Attention 2 极速狂飙。

| 配置项 | 推荐设定值 | 核心原因与避坑说明 |
| :--- | :--- | :--- |
| **截断长度** | `4096` 或 `8192` | 显存大就是任性，支持长文档微调。 |
| **Batch Size** | `4` 或 `8` | 充分利用 CUDA 核心并行计算。 |
| **加速方式** | `flashattn2` | 极大提升训练速度，降低长文本显存占用。 |
| **计算类型** | `bf16` | 满血精度。 |

---

## 🧮 进阶必读：大模型微调显存 (VRAM) 消耗估算与速查表

在大模型微调中，显存占用不仅仅是“把模型塞进显卡”那么简单。实际上，你的显存是被切分成三块大蛋糕吃掉的。

### 1. 核心显存估算公式

大模型微调的总显存占用 $V_{total}$ 主要由三部分组成：

$$V_{total} \approx V_{weights} + V_{optimizer} + V_{activations}$$

* **$V_{weights}$**：模型本身的静态权重大小。
* **$V_{optimizer}$**：训练时优化器（如 AdamW）和梯度记录的状态大小。
* **$V_{activations}$**：前向传播时产生的中间激活值（也就是“吃显存的无底洞”）。

### 2. 7 大核心参数如何影响显存？

| 参数名称 | 影响哪个变量 | 显存计算逻辑与数学关系 |
| :--- | :--- | :--- |
| **量化 (Quant) & 计算类型 (Compute Type)** | 模型权重 $V_{weights}$ | $V_{weights} = Params \times Q$。其中 $Params$ 为模型参数量（如 1.5B 即 15亿）。$Q$ 为精度系数：bf16/fp16 为 2 字节，8-bit 为 1 字节，4-bit 为 0.5 字节。 |
| **LoRA 秩 (Rank)** | 优化器 $V_{optimizer}$ | Rank 越高，新增的可训练参数越多。标准 AdamW 会为每个可训练参数额外占用约 8 字节（动量和方差）加 4 字节（梯度）。通常 LoRA 占用极小（< 500MB）。 |
| **批处理大小 (Batch Size)** | 激活值 $V_{activations}$ | **呈绝对正比例增长**。Batch Size 从 1 变 2，激活值显存直接翻倍。 |
| **截断长度 (Cutoff Length)** | 激活值 $V_{activations}$ | **显存核弹**！在标准注意力机制下，空间复杂度为 $O(L^2)$，即长度翻倍，显存消耗涨 4 倍！ |
| **加速方式 (Flash Attention)** | 激活值 $V_{activations}$ | **救命解药**！开启后，它通过底层硬件优化，强行把注意力的空间复杂度从 $O(L^2)$ 降维打击到 $O(L)$，让长文本微调成为可能。 |
| **梯度累加 (Gradient Acc)** | **不增加显存** | 它是用时间换空间的魔法。无论累加多少步，单次前向传播的显存占用只取决于你设置的基础 Batch Size。 |

### 3. 12GB 显存极限生存速查表 (单卡 LoRA 微调)

基于上述计算逻辑，如果是你的单卡 **12GB** 显存，面对不同的模型参数量，生存边界如下：

| 模型规模 | 参数设置方案 (12GB 显存不爆法则) | 估算显存占用 | 训练可行性 |
| :--- | :--- | :--- | :--- |
| **0.5B / 1.5B**<br>(如 Qwen 1.5B) | `bf16`, Quant:`none`, Batch:`2`, Cutoff:`1024` | 约 5GB - 7GB | **极度舒适**。甚至能把 Cutoff 开到 2048。 |
| **3B / 4B**<br>(如 Qwen 3B) | `bf16`, Quant:`none`, Batch:`1`, Cutoff:`1024` | 约 9GB - 11GB | **极限拉扯**。显卡风扇狂转，但能跑通原生精度。 |
| **7B / 8B**<br>(如 Llama3 8B) | `bf16`, Quant:`4-bit`, Batch:`1`, Cutoff:`512` | 约 8GB - 10GB | **必须妥协**。必须上 4-bit 量化，且文本必须截断到极短。 |
| **14B 及以上** | 任何本地单卡配置 | 远超 12GB | **直接 OOM**。请租用云端算力。 |

> **💡 极客避坑提示：** > 全新的 RTX 50 系显卡 (Blackwell 架构) 因为目前开源生态尚未完全支持 4-bit 量化 (`bitsandbytes`) 和 Flash Attention，所以现阶段你的极限就是纯净版 `bf16` 跑 3B 模型。等社区底层库完成适配后，你就可以轻松跑 7B 模型了！

## 🚀 第三阶段：模型融合导出与企业级 RAG 部署实战

*(注：本部分为进阶核心操作，以下提供了完整的架构设计与单文件极简版实现代码。)*

当大模型通过 LoRA 学习到了你的业务知识后，它实际上只是产生了一个“外挂权重补丁”。为了让它能像 ChatGPT 一样在企业的 Web 系统中稳定运行，并且彻底消除胡说八道的“幻觉”，我们需要进行**模型融合**与 **RAG（检索增强生成）** 部署。

### 1. 模型融合与导出 (脱离 LLaMA-Factory)

在 LLaMA-Factory 的 WebUI 界面中，切换到 **`Export (导出)`** 选项卡，进行以下操作：
1. **模型路径 (Model Path)**: 保持为你最初选择的底座模型（如 `Qwen/Qwen2.5-1.5B-Instruct`）。
2. **检查点路径 (Checkpoint Path)**: 选择你刚刚微调完成并保存的 LoRA 权重文件夹（如 `train_2026-xxx`）。
3. **导出路径 (Export Directory)**: 填入一个全新的本地文件夹名称，例如 `export_models/Qwen-1.5B-MyDomain`。
4. **最大分块大小 (Max Chunk Size)**: 保持默认的 `2GB`。
5. 点击 **“导出 (Export)”**。
   *系统会在后台将底座模型与你的 LoRA 补丁完美融合成一个全新的、可独立运行的模型文件夹。以后我们在代码中直接调用这个新文件夹即可。*

### 2. 企业级落地技术栈

为了将微调后的模型正式接入业务线，我们将组合使用以下业界标准的开源组件：
* **推理引擎 (HuggingFace / vLLM)**：负责将沉睡在硬盘里的模型加载到显存中。小显存推荐直接使用 HuggingFace 原生加载；如果是 24G 以上显存且追求高并发，推荐使用 vLLM 部署。
* **向量数据库 (FAISS / ChromaDB)**：用于本地化存储你的企业私有知识库（如 PDF、Word、员工手册），支持毫秒级的语义检索。
* **编排框架 (LangChain)**：作为粘合剂，将大模型、向量库、历史对话记忆无缝串联。
* **Web 框架 (FastAPI)**：对外提供稳定、高并发的 RESTful API 接口，方便前端界面或微信机器人调用。

### 3. RAG 核心运行流程 (防幻觉利器)

RAG（Retrieval-Augmented Generation）是目前解决大模型幻觉的最优解。它的运行流水线如下：
1. **用户提问**：用户输入“请问公司的出差报销流程是什么？”
2. **向量检索**：系统将问题转化为多维向量，在本地 FAISS 数据库中闪电匹配并抽出最相关的 3 条公司制度原话。
3. **上下文拼接**：将这 3 条原话与用户的问题，组装成一个结构化的 Prompt。
4. **模型生成**：微调后的专属大模型基于这些准确的参考资料，用符合公司专业语气的口吻生成最终回答。

### 4. 极简版 RAG + FastAPI 部署代码

在项目根目录下新建 `rag_api.py`。
**安装依赖**：`pip install fastapi uvicorn sentence-transformers faiss-cpu pydantic transformers accelerate`

将以下代码复制进 `rag_api.py`，它完美适配 12GB 显存单卡环境：

```
python
import torch
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
```
```
app = FastAPI(title="轻量级企业大模型 RAG 接口")

```
1. 加载模型 (BGE 向量检索模型 + 你的微调大模型)
```
print("正在加载 BGE 向量模型 (约 0.5GB 显存)...")
embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

print("正在加载微调后的专属 Qwen 模型...")
# 这里替换为你刚才导出的融合模型路径
MODEL_PATH = "export_models/Qwen-1.5B-MyDomain" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16, # 保持 bf16 防止爆显存
    device_map="auto"
)

```
2. 构建本地 FAISS 向量知识库 (模拟企业数据)
```
print("正在构建 FAISS 向量知识库...")
knowledge_base = [
    "报销制度：员工差旅费报销需在出差结束后 7 个工作日内通过 OA 系统提交，并附带增值税专用发票。",
    "IT 支持：如果电脑出现蓝屏或死机，请先尝试长按电源键强制重启，若无法解决请拨打分机号 8080。",
    "病假规定：员工每人每月享有 1 天带薪病假，需提供三甲医院的病假条即可不扣底薪。"
]

# 将文字转化为向量并存入高速 FAISS 索引
embeddings = embed_model.encode(knowledge_base)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embeddings).astype('float32'))

```
3. 定义 FastAPI 接口逻辑
```
class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_with_rag(request: ChatRequest):
    user_query = request.query
    
    # [检索阶段]：把用户的问题变成向量，去数据库里找最相似的 1 条知识
    query_vector = embed_model.encode([user_query])
    distances, indices = faiss_index.search(np.array(query_vector).astype('float32'), k=1)
    retrieved_doc = knowledge_base[indices[0][0]]
    
    # [增强阶段]：把找出来的企业知识和用户的问题拼在一起
    prompt = f"""你是一个专业的企业内部助手。请根据以下[参考资料]准确回答用户的问题。如果资料中没有相关信息，请直接回答“我不知道，请联系 HR”。
    
[参考资料]：{retrieved_doc}

[用户问题]：{user_query}
"""
    
    # [生成阶段]：交给大模型推理
    messages = [
        {"role": "system", "content": "你是公司专属 AI 助手。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    
    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=256,
        temperature=0.3 # 低温度确保回答严谨不发散
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return {
        "query": user_query,
        "retrieved_context": retrieved_doc, # 返回检索到的原话，方便前端做来源溯源
        "answer": response
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 服务已启动！请用 Postman 或前端调用 http://localhost:8000/api/chat")
    uvicorn.run(app, host="0.0.0.0", port=8000)

```

## 🤝 贡献与支持

如果你在使用不同型号的显卡时探索出了更优的参数组合，欢迎提交 Pull Request 补充到“黄金配置表”中！

如果本教程帮助你成功点亮了新显卡或完成了第一次微调，请点亮右上角的 ⭐️ **Star** 支持一下本项目！

