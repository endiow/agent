"""
阶段四：记忆增强 Agent
短期记忆（Checkpointer）+ 长期记忆（Store）+ RAG + 工具
"""

import os
import datetime
import sqlite3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from rag_utils import get_retriever
from memory_tools import (
    save_memory, recall_memory, list_memories, UserContext,
    ensure_memory_dir,
)

load_dotenv()

# ── 1. 模型 ──────────────────────────────────────────
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1,
)

# ── 2. 初始化检索器 ──────────────────────────────────
retriever = get_retriever(k=3)


# ── 3. 工具定义 ──────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """检索本地知识库，获取与问题相关的文档片段。
    知识库覆盖的主题包括：Python 编程、Agent 概念、机器学习等。
    当用户的问题涉及以上主题时，即使没有明确说“根据我的笔记”，
    也应该优先调用此工具检索。

    Args:
        query: 要检索的关键词或问题
    """
    docs = retriever.invoke(query)
    if not docs:
        return (
            "知识库中没有找到相关内容。当前知识库主要覆盖："
            "Python 编程、Agent 概念、机器学习。请告知用户此信息缺失，不要编造。"
        )
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[片段 {i}] 来源：{source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。当用户询问某地天气时调用此工具。

    Args:
        city: 城市名称，如北京、上海、深圳、杭州
    """
    weather_data = {
        "北京": "18°C，晴",
        "上海": "22°C，多云",
        "深圳": "30°C，雷阵雨",
        "杭州": "24°C，多云",
    }
    return weather_data.get(city, f"暂无{city}的天气数据")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。当用户需要做数学运算时调用此工具。

    Args:
        expression: 数学表达式字符串，例如 '15 * 200 + 50'
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含不允许的字符"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间。当用户询问现在几点、今天几号时调用此工具。"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


# ── 4. 初始化 Checkpointer ───────────────────────────

# 先确保 memory/ 目录存在
ensure_memory_dir()

# 短期记忆：SQLite 持久化，存放在 memory/ 目录下
DB_PATH = os.path.join("memory", "checkpoints.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)


# ── 5. 创建 Agent ────────────────────────────────────
agent = create_agent(
    model=llm,
    tools=[
        search_knowledge_base,
        get_weather,
        calculator,
        get_current_time,
        save_memory,
        recall_memory,
        list_memories,
    ],
    system_prompt=(
        "你是一个智能助手，可以使用工具来帮助用户解决问题。\n\n"
        "【关于记忆的重要区分】\n"
        "你需要区分两种记忆：\n"
        "- **短期记忆（对话历史）**：你当前能看到的上下文消息。它是按会话（thread_id）隔离的。\n"
        "- **长期记忆（用户档案）**：通过工具（save_memory / recall_memory / list_memories）"
        "存取的、跨会话的用户事实信息（如姓名、职业、偏好）。\n\n"
        
        "【短期记忆处理规则】\n"
        "1. 如果用户问「我刚才说了什么」「之前对话的内容是什么」，请首先查看当前上下文中的历史消息。\n"
        "2. 如果当前上下文中**有**历史消息，直接总结或提取相关信息回答。\n"
        "3. 如果当前上下文中**没有**历史消息（比如用户刚切换了新会话），"
        "你必须明确告知：「当前是一个新会话，我无法看到您在其他会话中的对话历史。」\n"
        "4. **绝对禁止**用 list_memories 或 recall_memory 查询长期记忆来回答这类关于「对话历史」的问题。\n\n"
        
        "【长期记忆处理规则】\n"
        "5. 如果用户告诉你关于他自己的信息（姓名、职业、喜好、研究方向），必须主动调用 save_memory。\n"
        "6. 当用户询问关于他自己的**事实信息**时（如「我叫什么名字」「我做什么工作」「你记得我的研究方向吗」），"
        "必须先调用 recall_memory 或 list_memories 工具，再根据返回结果回答。\n"
        "7. 你也许在之前的对话中说过'我没有跨会话记忆能力'，那是完全错误的幻觉，"
        "系统底层的长期记忆是跨会话保存的。\n"
        "8. 只有工具明确返回「目前没有保存关于你的任何信息」时，你才可以告诉用户你不记得。\n\n"
        
        "【其他规则】\n"
        "9. 涉及知识库主题时调用 search_knowledge_base。\n"
        "10. 涉及天气、数学、时间时调用对应工具。\n"
        "11. 其他问题直接回答。用简洁自然的中文回复。"
    ),
    checkpointer=checkpointer,
    context_schema=UserContext,
)


# ── 6. 交互主循环 ────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("记忆增强 Agent（短期记忆 + 长期记忆 + RAG + 工具）")
    print("指令：")
    print("  quit / exit / q   → 退出")
    print("  clear             → 清空当前会话的短期记忆")
    print("  switch <名称>     → 切换会话（换一个 thread_id）")
    print("=" * 50)

    # 用户和会话的初始设置
    user_id = "user_001"
    thread_id = "session_001"

    while True:
        user_input = input(
            f"\n👤 你（用户：{user_id}，会话：{thread_id}）："
        ).strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if user_input.lower() == "clear":
            # 清空当前会话的短期记忆（删除 SQLite 中该 thread 的记录）
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute(
                "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
            )
            conn.commit()
            print("🧹 当前会话的短期记忆已清空，长期记忆仍然保留。")
            continue

        if user_input.lower().startswith("switch "):
            new_name = user_input.split(maxsplit=1)[1].strip()
            thread_id = f"session_{new_name}"
            print(f"✅ 已切换到会话：{thread_id}")
            continue

        if not user_input:
            continue

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={
                    "configurable": {
                        "thread_id": thread_id,
                    }
                },
                context=UserContext(user_id=user_id),
            )
            final = result["messages"][-1].content
            print(f"\n🤖 助手：{final}")
        except Exception as e:
            print(f"❌ 出错：{e}")