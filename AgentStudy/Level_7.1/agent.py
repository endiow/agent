"""
阶段二：工具增强 Agent
使用 create_agent + @tool 装饰器，让 Agent 自主决定是否调用工具
"""

import os
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from weather_tool import get_local_weather, get_weather_by_city

load_dotenv()

# ── 1. 模型 ──────────────────────────────────────────
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1,
)


# ── 2. 定义工具（用 @tool 装饰器）─────────────────────

# @tool
# def get_weather(city: str) -> str:
#     """查询指定城市的当前天气。当用户询问某地天气时调用此工具。

#     Args:
#         city: 城市名称，如北京、上海、深圳、杭州
#     """
#     weather_data = {
#         "北京": "18°C，晴",
#         "上海": "22°C，多云",
#         "深圳": "30°C，雷阵雨",
#         "杭州": "24°C，多云",
#     }
#     return weather_data.get(city, f"暂无{city}的天气数据")


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


# ── 3. 创建 Agent ────────────────────────────────────
agent = create_agent(
    model=llm,
    tools=[
        get_local_weather,      # 本地/当前位置天气，IP 定位
        get_weather_by_city,    # 指定城市天气
        calculator,
        get_current_time,
    ],
    system_prompt=(
        "你是一个智能助手，可以使用工具来帮助用户解决问题。\n"
        "天气工具选择规则：\n"
        "1. 用户没有指定城市，说“我这里/本地/当前所在地/实时实地天气”时，"
        "调用 get_local_weather。\n"
        "2. 用户明确指定城市，如“深圳天气”“北京热不热”时，"
        "调用 get_weather_by_city。\n"
        "3. 不要凭空编造天气，必须根据工具返回结果回答。\n"
        "数学计算调用 calculator；时间日期调用 get_current_time。\n"
        "给出最终答案时，用简洁自然的中文回复。"
    ),
)


# ── 4. 交互主循环 ────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("工具增强 Agent（create_agent + @tool）")
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
            # 最终回答在最后一条消息
            final = result["messages"][-1].content
            print(f"\n🤖 助手：{final}")
        except Exception as e:
            print(f"❌ 出错：{e}")