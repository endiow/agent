"""
agent.py — MCP 版工具增强 Agent
通过 MCPAdapter 连接 MCP Server，动态发现工具
"""

import asyncio
import os
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.tools import tool

load_dotenv()

# ── 1. 模型 ──────────────────────────────────────────
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1,
)


# ── 2. 本地工具（不需要 MCP 的保留为 @tool）───────────

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


# ── 3. MCP 配置 ──────────────────────────────────────

MCP_SERVER_CONFIG = {
    "weather": {
        "transport": "stdio",
        "command": "python",
        "args": ["weather_mcp_server.py"],
    },
    "hefeng_qweather": {
        "transport": "http",
        "url": "http://127.0.0.1:8000/mcp",
    },
    "caiyun_weather": {
        "transport": "http",
        "url": "https://mcp-weather.caiyunapp.com/mcp",
        "headers": {
            "X-Caiyun-API-Key": os.getenv("CAIYUN_API_KEY"),
        },
    }
}


# ── 4. 主逻辑 ────────────────────────────────────────

async def main():
    print("=" * 50)
    print("MCP 工具增强 Agent")
    print("指令：quit / exit / q  → 退出")
    print("=" * 50)

    # 连接 MCP Server，自动发现工具
    async with MCPAdapter(MCP_SERVER_CONFIG) as adapter:
        mcp_tools = await adapter.list_tools()
        print(f"\n✅ 已连接 MCP Server，发现 {len(mcp_tools)} 个工具：")
        for t in mcp_tools:
            print(f"   - {t.name}")

        # 合并 MCP 工具 + 本地 @tool
        all_tools = mcp_tools + [calculator, get_current_time]

        agent = create_agent(
            model=llm,
            tools=all_tools,
            system_prompt=(
                "你是一个智能助手，可以使用工具来帮助用户解决问题。\n"
                "天气工具选择规则：\n"
                "1. 用户没有指定城市，说“我这里/本地/当前所在地”时，"
                "调用 local_weather。\n"
                "2. 用户明确指定城市，如“深圳天气”“北京热不热”时，"
                "调用 weather_by_city。\n"
                "3. 不要凭空编造天气，必须根据工具返回结果回答。\n"
                "数学计算调用 calculator；时间日期调用 get_current_time。\n"
                "给出最终答案时，用简洁自然的中文回复。"
            ),
        )

        # 交互循环
        while True:
            user_input = input("\n👤 你：").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            if not user_input:
                continue

            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]}
                )
                final = result["messages"][-1].content
                print(f"\n🤖 助手：{final}")
            except Exception as e:
                print(f"❌ 出错：{e}")


if __name__ == "__main__":
    asyncio.run(main())