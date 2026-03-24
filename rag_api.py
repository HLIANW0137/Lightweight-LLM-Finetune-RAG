import torch
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# 允许跨域请求，不需要可忽略
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有网页访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app = FastAPI(title="轻量级企业大模型 RAG 接口")

# ==========================================
# 1. 加载模型 (BGE 向量检索模型 + 你的微调大模型)
# ==========================================
print("正在加载 BGE 向量模型 (约 0.5GB 显存)...")
embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

print("正在加载微调后的专属 Qwen 模型...")
# 这里替换为你刚才导出的融合模型路径
MODEL_PATH = "export_models/Qwen-1.5B-MyDomain"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 保持 bf16 防止爆显存
    device_map="auto"
)

# ==========================================
# 2. 构建本地 FAISS 向量知识库 (模拟企业数据)
# ==========================================
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


# ==========================================
# 3. 定义 FastAPI 接口逻辑
# ==========================================
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
        temperature=0.3  # 低温度确保回答严谨不发散
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {
        "query": user_query,
        "retrieved_context": retrieved_doc,  # 返回检索到的原话，方便前端做来源溯源
        "answer": response
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 服务已启动！请用 Postman 或前端调用 http://localhost:8000/api/chat")
    uvicorn.run(app, host="0.0.0.0", port=8000)
