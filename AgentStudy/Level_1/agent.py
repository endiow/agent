"""
ReAct + Tools 最小 Agent 应用
功能：Agent 能自主推理，决定是否调用工具（计算器、时间查询），
     观察结果后继续推理，直到给出最终答案。
"""

import os
import json
import time
import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 第 1 步：初始化 LLM 客户端（阿里云百炼，兼容 OpenAI 格式）
# ============================================================
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取，不要硬编码
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 使用的模型（百炼免费额度覆盖 qwen 系列）
MODEL = "qwen-plus"


# ============================================================
# 第 2 步：定义工具（Tools）
# ============================================================

def calculator(expression: str) -> str:
    """
    计算器工具：输入一个数学表达式，返回计算结果。
    例如：calculator("15 * 200 + 50") -> "3050"
    """
    try:
        # 安全限制：只允许数字和基本运算符
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


def get_current_time() -> str:
    """
    时间查询工具：返回当前日期和时间。
    """
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

def get_weather(city: str) -> str:
    """天气查询工具：输入城市名，返回当前天气。"""
    weather_data = {
        "北京": "18°C，晴",
        "上海": "22°C，多云",
        "深圳": "30°C，雷阵雨",
        "杭州": "24°C，多云"
    }
    return weather_data.get(city, f"暂无{city}的天气数据")


def write_file(filename: str, content: str) -> str:
    """文件写入工具：将内容写入本地文件。"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功写入 {filename}"
    except Exception as e:
        return f"写入失败：{e}"


def search_web(query: str) -> str:
    """网页搜索工具（模拟）：返回模拟的搜索结果。"""
    mock_results = {
        "Python": "Python 是一种广泛使用的高级编程语言。",
        "Agent": "Agent 是能够自主感知环境并采取行动的系统。",
    }
    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value
    return f"关于「{query}」的模拟搜索结果：未找到直接匹配的内容。"


# 工具注册表：将工具名称映射到实际函数
TOOL_REGISTRY = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "write_file": write_file,
    "search_web": search_web,
}

# 日志文件路径（与 agent.py 同目录）
LOG_FILE = "tool_log.jsonl"


def log_tool_call(tool_name: str, tool_args: dict, result: str,
                  elapsed: float, success: bool):
    """
    将一次工具调用追加写入 JSONL 日志。
    每行是一个独立的 JSON 对象，方便后续分析。
    """
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool": tool_name,
        "args": tool_args,
        "result": result,
        "elapsed_sec": round(elapsed, 4),
        "success": success,
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[日志写入失败] {e}")

# 工具描述（告诉 LLM 有哪些工具可用、怎么用）
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式。当用户需要做数学运算时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 '15 * 200 + 50'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。当用户询问现在几点、今天日期时调用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气。当用户询问某地天气时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如北京、上海、深圳、杭州"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将文本内容写入本地文件。当用户要求保存、记录、写入文件时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名，例如 'result.txt'"
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入文件的文本内容"
                    }
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索网页获取信息。当用户询问你不了解的知识、概念或实时信息时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，例如 'Python'、'Agent'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ============================================================
# 第 3 步：执行工具调用
# ============================================================

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    根据 LLM 返回的工具名称和参数，执行对应的工具函数。
    并自动将调用记录写入 tool_log.jsonl。
    """
    start = time.time()

    if tool_name not in TOOL_REGISTRY:
        result = f"错误：未知工具 '{tool_name}'"
        success = False
    else:
        func = TOOL_REGISTRY[tool_name]
        try:
            if tool_args:
                result = func(**tool_args)
            else:
                result = func()
            result = str(result)
            success = True
        except Exception as e:
            result = f"工具执行错误：{e}"
            success = False

    elapsed = time.time() - start

    # 写入日志（无论成功失败都记录）
    log_tool_call(tool_name, tool_args, result, elapsed, success)

    return result


# ============================================================
# 第 4 步：ReAct 核心循环
# ============================================================

SYSTEM_PROMPT = """你是一个智能助手，可以使用工具来帮助用户解决问题。

工作方式（ReAct 模式）：
1. 先推理（Reason）：分析用户的问题，判断是否需要调用工具。
2. 再行动（Act）：如果需要工具，调用对应的工具函数。
3. 观察结果（Observe）：查看工具返回的结果。
4. 继续推理：根据结果判断是否还需要更多信息，或给出最终答案。

可用工具：
- calculator：计算数学表达式
- get_current_time：获取当前日期和时间
- get_weather：查询指定城市的天气
- write_file：将内容写入本地文件
- search_web：搜索网页获取信息

重要规则：
- 如果问题不涉及工具能力，直接回答，不要调用工具。
- 调用工具时，确保参数格式正确。
- 如果多个任务之间没有依赖关系，你可以在一次回复中同时调用多个工具，以提高效率。
- 给出最终答案时，用简洁自然的中文回复。"""


def run_agent(user_input: str, max_steps: int = 5, verbose: bool = True):
    """
    ReAct Agent 主循环。

    参数：
        user_input: 用户输入的问题
        max_steps: 最大循环步数（防止无限循环）
        verbose: 是否打印中间推理过程
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for step in range(max_steps):
        if verbose:
            print(f"\n{'='*50}")
            print(f"🔄 第 {step + 1} 轮推理")
            print(f"{'='*50}")

        # ---- Reason: 调用 LLM，让它决定下一步 ----
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.1,
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        # ---- 判断：LLM 是否要调用工具？ ----
        if assistant_message.tool_calls:
            # 有工具调用 → Act + Observe
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(f"🔧 调用工具: {tool_name}")
                    print(f"📥 参数: {tool_args}")

                # 执行工具（Act）
                result = execute_tool(tool_name, tool_args)

                if verbose:
                    print(f"📤 结果: {result}")

                # 将工具结果作为 Observation 追加到消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            # 继续循环，让 LLM 根据工具结果决定下一步
        else:
            # 没有工具调用 → 这是最终答案
            final_answer = assistant_message.content
            if verbose:
                print(f"\n✅ 最终答案：{final_answer}")
            return final_answer

    # 超过最大步数，强制结束
    return "已达到最大推理步数，无法继续。"


# ============================================================
# 第 5 步：交互式运行
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 ReAct Agent 已启动（输入 quit 退出）")
    print("=" * 50)

    while True:
        user_input = input("\n👤 你：").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        if not user_input:
            continue

        try:
            run_agent(user_input, max_steps=5, verbose=True)
        except Exception as e:
            print(f"❌ 运行出错：{e}")