import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 复用你在 agent 里创建的 client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def load_and_split_docs(docs_dir="docs", chunk_size=200, overlap=50):
    """读取 docs 目录下的所有 .txt 文件，按字符数切分成块"""
    chunks = []
    if not os.path.exists(docs_dir):
        return chunks
    
    for filename in os.listdir(docs_dir):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        # 简单的滑动窗口切分
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i+chunk_size]
            if chunk.strip(): # 过滤空块
                chunks.append(chunk)
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

def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    核心检索函数：将用户问题向量化，与本地向量库计算余弦相似度，
    返回最相似的 top_k 个文本块。
    """
    vectors, chunks = load_index()
    if vectors is None:
        return "知识库尚未建立，请先运行 build_index。"

    # 获取查询向量
    query_vec = np.array(get_embedding(query))
    
    # 计算余弦相似度
    # 归一化
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    
    # 点积得到相似度分数
    scores = np.dot(vectors_norm, query_norm)
    
    # 获取分数最高的 top_k 索引
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    # 拼接结果
    results = []
    for idx in top_indices:
        results.append(chunks[idx])
    
    return "\n---\n".join(results)

if __name__ == "__main__":
    # 运行此文件时，执行一次索引构建
    chunks = load_and_split_docs()
    if chunks:
        build_index(chunks)
    else:
        print("没有找到 docs 目录或目录下没有 txt 文件。")