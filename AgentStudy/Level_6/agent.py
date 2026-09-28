import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── 1. 模型 ──────────────────────────────────────────────
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1,   # 事实性问答用低温，减少随机性
)

# ── 2. 提示词模板 ────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的助手，用简洁的中文回答用户的问题。"
               "如果不知道答案，直接说'我不知道'，不要编造。"),
    ("human", "{user_input}"),
])

# ── 3. 链 ────────────────────────────────────────────────
chain = prompt | llm | StrOutputParser()

# ── 4. 基础调用 ──────────────────────────────────────────
def ask(question: str) -> str:
    return chain.invoke({"user_input": question})

# ── 5. 流式输出 ──────────────────────────────────────────
def ask_stream(question: str):
    for chunk in chain.stream({"user_input": question}):
        print(chunk, end="", flush=True)
    print()

if __name__ == "__main__":
    # 基础调用
    print(ask("什么是向量数据库？"))

    # 流式输出
    ask_stream("用三句话解释什么是Agent。")