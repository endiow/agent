"""
RAG 工具模块：加载文档、切分、向量化、构建检索器
支持文档变化自动检测，内容变动时自动重建向量库。
"""

import os
import shutil
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

DOCS_DIR = "docs"                        # 原始文档目录
CHROMA_DIR = "chroma_db"                 # 向量库持久化目录
HASH_FILE = os.path.join(CHROMA_DIR, ".docs_hash")   # 哈希记录文件

# 切分参数（同时参与哈希计算，改动参数也会触发重建）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ============================================================
# 1. 文档加载与切分
# ============================================================

def load_documents():
    """加载 docs/ 目录下的所有 txt 和 pdf 文档"""
    docs = []
    if not os.path.exists(DOCS_DIR):
        return docs

    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            docs.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
    return docs


def split_documents(docs):
    """把长文档切成小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    return splitter.split_documents(docs)


# ============================================================
# 2. 文档变化检测
# ============================================================

def compute_docs_hash() -> str:
    """
    计算 docs/ 目录下所有文档内容的哈希。
    同时把切分参数纳入哈希，确保改了参数也会触发重建。
    """
    hasher = hashlib.md5()

    # 把切分参数拼进去
    hasher.update(f"chunk_size={CHUNK_SIZE},overlap={CHUNK_OVERLAP}".encode())

    if not os.path.exists(DOCS_DIR):
        return hasher.hexdigest()

    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith((".txt", ".pdf")):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "rb") as f:
            hasher.update(filename.encode())
            hasher.update(f.read())

    return hasher.hexdigest()


def needs_rebuild() -> bool:
    """判断向量库是否需要重建"""
    if not os.path.exists(CHROMA_DIR):
        return True
    if not os.path.exists(HASH_FILE):
        return True
    try:
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            old_hash = f.read().strip()
        return old_hash != compute_docs_hash()
    except Exception:
        return True


def save_docs_hash():
    """保存当前文档哈希"""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        f.write(compute_docs_hash())


# ============================================================
# 3. 向量库构建
# ============================================================

def build_vectorstore():
    """构建向量库并持久化"""
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    docs = load_documents()
    if not docs:
        print("[RAG] docs/ 目录为空，跳过向量库构建")
        return None

    splits = split_documents(docs)
    print(f"[RAG] 加载 {len(docs)} 个文档，切分为 {len(splits)} 个片段")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"[RAG] 向量库已构建，持久化到 {CHROMA_DIR}")
    return vectorstore


# ============================================================
# 4. 检索器（带自动重建）
# ============================================================

def get_retriever(k: int = 3):
    """
    获取检索器。
    - 向量库不存在 → 构建
    - 文档内容有变化 → 删除旧库，重建
    - 无变化 → 直接加载
    """
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    if needs_rebuild():
        print("[RAG] 检测到文档变化（或首次运行），重建向量库...")
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        build_vectorstore()
        save_docs_hash()
    else:
        print("[RAG] 向量库无变化，直接加载")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ============================================================
# 5. 手动重建入口（可选）
# ============================================================

if __name__ == "__main__":
    """直接运行此文件时，强制重建向量库"""
    print("=" * 50)
    print("手动重建向量库")
    print("=" * 50)

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"已删除旧向量库：{CHROMA_DIR}")

    result = build_vectorstore()
    if result:
        save_docs_hash()
        print("重建完成")
    else:
        print("docs/ 目录为空，未构建向量库")