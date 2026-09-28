"""
阶段三：RAG + Agent
把知识库检索封装成 @tool，让 Agent 自主决定何时检索
"""

import os
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.documents import Document

from rag_utils import get_retriever

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
    当用户询问关于本地文档、个人笔记、内部资料的内容时调用此工具。

    Args:
        query: 要检索的关键词或问题
    """
    docs = retriever.invoke(query)
    if not docs:
        return "知识库中没有找到相关内容。当前知识库主要覆盖：[请根据你的文档内容填写]。请告知用户此信息缺失，不要编造。"

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


# ── 4. 创建 Agent ────────────────────────────────────
agent = create_agent(
    model=llm,
    tools=[search_knowledge_base, get_weather, calculator, get_current_time],
    system_prompt=(
        "你是一个智能助手，可以使用工具来帮助用户解决问题。\n\n"
        "重要规则：\n"
        "1. 如果用户询问的是本地知识库包含的内容（如个人笔记、内部文档），"
        "必须优先调用 search_knowledge_base 工具。\n"
        "2. 如果用户的问题涉及天气、数学计算或时间查询，调用对应的工具。\n"
        "3. 如果知识库返回'没有找到相关内容'，明确告知用户信息缺失，不要编造。\n"
        "4. 如果问题不涉及工具能力，直接回答。\n"
        "5. 给出最终答案时，用简洁自然的中文回复。"
    ),
)


# ── 5. 交互主循环 ────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("RAG Agent（知识库检索 + 工具调用）")
    print("指令：quit / exit / q  → 退出")
    print("=" * 50)

    while True:
        user_input = input("\n👤 你：").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        if not user_input:
            continue

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            final = result["messages"][-1].content
            print(f"\n🤖 助手：{final}")
        except Exception as e:
            print(f"❌ 出错：{e}")