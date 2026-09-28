import os
import json
import re
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 复用你在 agent 里创建的 client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def load_and_split_docs(docs_dir="docs", chunk_size=300, overlap=60):
    """
    按自然段落 + 句子边界切分文档。
    优先在 \n\n（段落）或 。！？\n（句子结尾）切分，
    避免把一句话从中间切断。
    """
    chunks = []
    if not os.path.exists(docs_dir):
        return chunks

    for filename in os.listdir(docs_dir):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # 先按段落切
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            if len(para) <= chunk_size:
                chunks.append(para)
                continue

            # 段落太长时，按句子切分后合并
            sentences = re.split(r"(?<=[。！？])", para)
            buffer = ""
            for sent in sentences:
                if len(buffer) + len(sent) <= chunk_size:
                    buffer += sent
                else:
                    if buffer:
                        chunks.append(buffer.strip())
                    buffer = sent
            if buffer.strip():
                chunks.append(buffer.strip())

    return chunks

def get_embedding(text: str) -> list:
    """调用百炼的 Embedding API 获取向量"""
    # 注意：百炼的 text-embedding-v2 是常用的免费/低价模型
    resp = client.embeddings.create(
        input=text,
        model="text-embedding-v2"
    )
    return resp.data[0].embedding

def build_index(chunks: list):
    """为所有文本块生成向量，并保存到本地"""
    print(f"正在为 {len(chunks)} 个文本块生成向量...")
    vectors = []
    for chunk in chunks:
        vec = get_embedding(chunk)
        vectors.append(vec)
    
    vectors = np.array(vectors)
    # 将向量和对应的文本块保存到本地，避免每次启动都重新计算
    np.save("docs/vectors.npy", vectors)
    with open("docs/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print("向量索引构建完成！")
    return vectors, chunks

def load_index():
    """加载本地的向量索引"""
    if os.path.exists("docs/vectors.npy") and os.path.exists("docs/chunks.json"):
        vectors = np.load("docs/vectors.npy")
        with open("docs/chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return vectors, chunks
    return None, None

def search_knowledge_base(query: str, top_k: int = 3, score_threshold: float = 0.45) -> str:
    vectors, chunks = load_index()
    if vectors is None:
        return "知识库尚未建立，请先运行 build_index。"

    query_vec = np.array(get_embedding(query))
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    scores = np.dot(vectors_norm, query_norm)

    top_indices = np.argsort(scores)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if scores[idx] >= score_threshold:
            results.append(chunks[idx])

    # 如果没有任何片段达到阈值 → 明确告知"知识边界"
    if not results:
        # 粗略列出知识库主题（从前 500 字里提取几个关键词，或人工维护）
        topics_hint = "Python、Agent、机器学习、过拟合/欠拟合、深度学习（部分）"
        return (
            f"[知识库无相关内容] 查询：{query}。"
            f"当前知识库主要覆盖：{topics_hint}。"
            f"请明确告知用户此信息缺失，不要编造或补全。"
        )

    return "\n---\n".join(results)

if __name__ == "__main__":
    # 运行此文件时，执行一次索引构建
    chunks = load_and_split_docs()
    if chunks:
        build_index(chunks)
    else:
        print("没有找到 docs 目录或目录下没有 txt 文件。")