"""
阶段一：基础问答 Agent（支持普通 / 流式 / 批量）
使用阿里云百炼（OpenAI 兼容模式）+ LangChain LCEL
"""

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── 1. 模型 ──────────────────────────────────────────
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1,
).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

# ── 2. 提示词模板 ────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个知识渊博的助手，用简洁的中文回答用户的问题。"
     "如果不知道答案，直接说'我不知道'，不要编造。"),
    ("human", "{user_input}"),
])

# ── 3. LCEL 链 ───────────────────────────────────────
chain = prompt | llm | StrOutputParser()


# ── 4. 三种调用方式 ──────────────────────────────────
def ask(question: str) -> str:
    """非流式：等全部生成完一次性返回"""
    return chain.invoke({"user_input": question})


def ask_stream(question: str) -> None:
    """流式：逐块打印"""
    for chunk in chain.stream({"user_input": question}):
        print(chunk, end="", flush=True)
    print()


def ask_batch(questions: list[str]) -> list[str]:
    """批量：并行处理多个问题，返回结果列表"""
    inputs = [{"user_input": q} for q in questions]
    return chain.batch(inputs)


# ── 5. 批量测试 ──────────────────────────────────────
def run_batch_test():
    """跑一组预置问题，对比串行和并行的耗时"""
    questions = [
        "什么是向量数据库？",
        "用一句话解释什么是 Agent。",
        "Python 的列表和元组有什么区别？",
        "什么是 RESTful API？",
        "解释一下什么是过拟合。",
    ]

    print(f"\n📦 批量测试：共 {len(questions)} 个问题")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. {q}")

    # ── 并行调用（batch）──────────────────────────────
    start = time.time()
    results = ask_batch(questions)
    elapsed_batch = time.time() - start

    # ── 串行调用（逐个 invoke，用于对比）────────────────
    start = time.time()
    for q in questions:
        ask(q)
    elapsed_serial = time.time() - start

    # ── 输出结果与耗时对比 ─────────────────────────────
    print("\n" + "=" * 50)
    print("📋 批量结果")
    print("=" * 50)
    for i, (q, a) in enumerate(zip(questions, results), 1):
        print(f"\n【{i}】{q}")
        print(f"     {a[:100]}{'...' if len(a) > 100 else ''}")

    print("\n" + "=" * 50)
    print("⏱️ 耗时对比")
    print("=" * 50)
    print(f"  batch 并行：{elapsed_batch:.2f} 秒")
    print(f"  invoke 串行：{elapsed_serial:.2f} 秒")
    print(f"  加速比：{elapsed_serial / elapsed_batch:.2f}x")


# ── 6. 交互主循环 ────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("基础问答 Agent（阿里云百炼 qwen-plus）")
    print("指令：")
    print("  quit / exit / q  → 退出")
    print("  stream           → 切换到流式模式")
    print("  normal           → 切换到普通模式")
    print("  batch            → 运行批量测试")
    print("=" * 50)

    current_mode = "normal"

    while True:
        user_input = input(f"\n👤 你（当前模式：{current_mode}）：").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if user_input.lower() == "stream":
            current_mode = "stream"
            print("✅ 已切换到流式模式")
            continue

        if user_input.lower() == "normal":
            current_mode = "normal"
            print("✅ 已切换到普通模式")
            continue

        if user_input.lower() == "batch":
            run_batch_test()
            continue

        if not user_input:
            continue

        try:
            if current_mode == "stream":
                print("\n🤖 助手：", end="", flush=True)
                ask_stream(user_input)
            else:
                answer = ask(user_input)
                print(f"\n🤖 助手：{answer}")
        except Exception as e:
            print(f"❌ 出错：{e}")