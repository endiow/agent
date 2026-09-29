"""
阶段五：Plan-and-Execute Agent（含短期记忆）
规划 → 执行 → 反思 → 推进 → 汇总
复用阶段四的 RAG、记忆工具和用户上下文
支持 switch 切换用户 / clear 清空会话

短期记忆机制：
  - State 里声明 messages: Annotated[list, add_messages]
  - thread_id 固定为 {user_id}_session
  - finalizer 把 AI 回复写回 messages
"""

import os
import json
import re
import datetime
import sqlite3
from typing import TypedDict, Optional, Annotated
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
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

# ── 2. 检索器 ─────────────────────────────────────────
retriever = get_retriever(k=3)


# ── 3. 业务工具 ──────────────────────────────────────

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


# ── 4. 内层执行器 ─────────────────────────────────────
INNER_SYSTEM_PROMPT = """你是一个任务执行器，只专注完成给定的这一个步骤。

规则：
1. 如果这一步需要工具，调用必要的工具完成任务。
2. 如果这一步不需要工具，直接给出结果。
3. 如果工具调用失败或信息不足，在 result 中明确标注
   "[信息不足：具体缺失什么]"，不得编造。
4. 如果用户询问关于他自己的信息（姓名、职业、偏好等），
   必须调用 recall_memory 或 list_memories 工具查询。
5. 如果用户告诉你关于他自己的信息，主动调用 save_memory。

完成后，严格按以下 JSON 格式输出，不要任何解释、不要 markdown 代码块：
{{"status": "success 或 failed", "result": "这一步的执行结果"}}
"""


@lru_cache(maxsize=1)
def get_inner_agent():
    return create_agent(
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
        system_prompt=INNER_SYSTEM_PROMPT,
        context_schema=UserContext,
    )


# ── 5. State（★ 新增 messages） ──────────────────────
class PlanExecuteState(TypedDict):
    task: str
    messages: Annotated[list, add_messages]     # ★ 跨轮对话历史
    plan: list
    current_step: str
    past_steps: list
    last_result: str
    last_tool_log: list
    reflection_feedback: Optional[str]
    reflection_count: int
    final_response: str


# ── 6. 提示词（★ Planner / Summarizer 加入 conversation） ──
PLANNER_PROMPT = """你是一个任务规划器。请将用户的复杂任务拆解成一系列清晰、有序、可独立执行的步骤。

【历史对话】
{conversation}

【本轮任务】
{task}

要求：
1. 每个步骤只做一件事，描述要具体、可执行。
2. 步骤之间要有明确的先后顺序。
3. 如果本轮任务里出现了指代（"这个""刚才""上面那个""再算一次"），
   必须结合历史对话把它消解成自包含的表述，不要让执行器去猜。
   例：历史里问过"深圳天气"，本轮说"那北京呢" → 步骤应写成"查询北京天气"。
4. 只输出 JSON，不要任何解释、不要 markdown 代码块。
5. JSON 格式严格如下：
{{"steps": ["步骤1", "步骤2", "步骤3"]}}

你可以使用的工具（供参考，不要写进步骤描述）：
- search_knowledge_base：检索本地知识库
- get_weather：查询城市天气
- calculator：数学计算
- get_current_time：获取当前时间
- save_memory / recall_memory / list_memories：长期记忆

如果任务很简单（1-2 步可以完成），不要强行拆成更多步骤。
简单的事实查询、概念解释，通常 1 步就够了。
"""

EXECUTOR_USER_TEMPLATE = """原始总任务：{task}

已完成的步骤及结果：
{context}

当前需要执行的步骤：{step}

请调用必要的工具完成这一步，然后按 JSON 格式输出结果。
"""

REFLECTOR_PROMPT = """你是一个严格的审查者，负责检查任务执行器对某一步的执行结果。

原始任务：{task}
当前步骤：{step}

已完成的步骤及结果：
{context}

【执行器的工具调用证据】
{tool_evidence}

【执行器的最终输出】
{result}

审查维度：
1. 忠实度：输出中的每个事实，是否能在"工具调用证据"或"已完成步骤"中找到依据？
   执行器有没有在证据之外添加工具没返回的内容？
2. 完整性：这一步的任务是否真正完成？
3. 准确性：有没有事实错误或逻辑矛盾？
4. 诚实性：如果信息不足，执行器有没有明确标注 "[信息不足]"？

审查标准：
- 如果执行器有工具调用，且最终输出的事实与工具返回一致 → 通过。
- 如果工具本身返回的是模拟数据，只要执行器忠实引用了工具返回的内容 → 通过。
- 只有执行器在工具返回之外添加了未提供的事实 → 不通过。
- 如果最终输出只是"描述了打算做什么"而没有实际产出 → 不通过。
- 如果执行器对信息不足的情况正确标注了 "[信息不足]" → 通过。

请严格按以下 JSON 输出，不要任何解释：
{{"passed": true 或 false, "critique": "如果不通过，写具体问题；如果通过，写'无问题'"}}
"""

SUMMARIZER_PROMPT = """请根据历史对话和本轮执行结果，为用户生成一段完整、自然的最终回答。

【历史对话】
{conversation}

【本轮任务】
{task}

【本轮各步骤结果】
{results}

要求：
1. 用简洁的中文回答，直接说结论，像正常对话一样流畅。
2. 不要罗列"步骤1、步骤2"，也不要说"根据专家"、"根据工具"。
3. 如果结果里有 "[信息不足]" 之类的占位，如实告知用户，不要编造。
"""


# ── 7. 辅助函数 ──────────────────────────────────────
def _parse_json(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        return {}


def _extract_tool_log(messages: list) -> list:
    """从内层 agent 返回的消息列表里提取工具调用记录"""
    tool_log = []
    for i, msg in enumerate(messages):
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            tool_result = None
            for later in messages[i + 1:]:
                if getattr(later, "tool_call_id", None) == tc["id"]:
                    tool_result = later.content
                    break
            tool_log.append({
                "tool": tc["name"],
                "args": tc["args"],
                "result": tool_result,
            })
    return tool_log


def _format_conversation(messages: list) -> str:
    """把 messages 格式化成对话文本"""
    if not messages:
        return "（无）"
    label = {"human": "用户", "ai": "助手", "system": "系统"}
    lines = []
    for m in messages:
        role = getattr(m, "type", "?")
        content = getattr(m, "content", "")
        lines.append(f"{label.get(role, role)}：{content}")
    return "\n".join(lines)


# ── 8. 节点 ──────────────────────────────────────────
def planner_node(state: PlanExecuteState) -> dict:
    # ★ 只看历史（去掉本轮那条 HumanMessage）
    conv = _format_conversation(state["messages"][:-1]) \
        if len(state["messages"]) > 1 else "（无）"

    prompt = PLANNER_PROMPT.format(task=state["task"], conversation=conv)
    response = llm.invoke([HumanMessage(content=prompt)])
    data = _parse_json(response.content)

    if data and isinstance(data.get("steps"), list) and data["steps"]:
        steps = data["steps"]
    else:
        steps = [state["task"]]

    print(f"\n📋 规划完成，共 {len(steps)} 步：")
    for i, s in enumerate(steps, 1):
        print(f"   {i}. {s}")

    return {
        "plan": steps,
        "current_step": steps[0],
        "past_steps": [],
        "last_result": "",
        "last_tool_log": [],
        "reflection_feedback": None,
        "reflection_count": 0,
    }


def executor_node(state: PlanExecuteState, config) -> dict:
    user_id = config.get("configurable", {}).get("user_id", "default")
    current_step = state["current_step"]

    print(f"\n   ▶️ 执行：{current_step}")

    if state["past_steps"]:
        context = "\n".join(f"- {s} → {r}" for s, r in state["past_steps"])
    else:
        context = "（无）"

    user_msg = EXECUTOR_USER_TEMPLATE.format(
        task=state["task"],
        context=context,
        step=current_step,
    )

    if state.get("reflection_feedback"):
        user_msg += (
            f"\n\n【重要】上一次执行被审查者拒绝了，反馈如下：\n"
            f"{state['reflection_feedback']}\n"
            f"请针对这些问题重新执行，务必避免相同的错误。"
        )
        print(f"   🔁 重做（第 {state['reflection_count']} 次）")

    inner_agent = get_inner_agent()
    result = inner_agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        context=UserContext(user_id=user_id),
    )

    messages = result["messages"]
    final_content = messages[-1].content
    data = _parse_json(final_content)
    exec_result = data.get("result", final_content) if data else final_content

    tool_log = _extract_tool_log(messages)

    for t in tool_log:
        print(f"      🔧 {t['tool']}({t['args']})")

    preview = exec_result[:120] + ("..." if len(exec_result) > 120 else "")
    print(f"   ✅ 结果：{preview}")

    return {"last_result": exec_result, "last_tool_log": tool_log}


def reflector_node(state: PlanExecuteState) -> dict:
    if state["past_steps"]:
        context = "\n".join(f"- {s} → {r}" for s, r in state["past_steps"])
    else:
        context = "（无）"

    if state.get("last_tool_log"):
        lines = []
        for i, t in enumerate(state["last_tool_log"], 1):
            lines.append(
                f"[工具 {i}] {t['tool']}({t['args']})\n"
                f"[返回 {i}] {t['result']}"
            )
        tool_evidence = "\n\n".join(lines)
    else:
        tool_evidence = "（执行器没有调用任何工具）"

    prompt = REFLECTOR_PROMPT.format(
        task=state["task"],
        step=state["current_step"],
        context=context,
        tool_evidence=tool_evidence,
        result=state["last_result"],
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    data = _parse_json(response.content)

    passed = bool(data.get("passed", True)) if data else True
    critique = data.get("critique", "无问题") if data else "无问题"

    if passed:
        print(f"   ✅ 审查通过")
        return {"reflection_feedback": None}
    else:
        print(f"   ⚠️ 审查未通过：{critique[:120]}")
        return {
            "reflection_feedback": critique,
            "reflection_count": state["reflection_count"] + 1,
        }


def advance_node(state: PlanExecuteState) -> dict:
    new_past = state["past_steps"] + [[state["current_step"], state["last_result"]]]
    new_plan = state["plan"][1:]
    next_step = new_plan[0] if new_plan else ""

    return {
        "past_steps": new_past,
        "plan": new_plan,
        "current_step": next_step,
        "last_result": "",
        "last_tool_log": [],
        "reflection_feedback": None,
        "reflection_count": 0,
    }


def finalizer_node(state: PlanExecuteState) -> dict:
    print(f"\n📝 正在汇总最终答案...")

    # ★ 只看历史
    conv = _format_conversation(state["messages"][:-1]) \
        if len(state["messages"]) > 1 else "（无）"

    results = "\n".join(f"- {s}：{r}" for s, r in state["past_steps"])
    prompt = SUMMARIZER_PROMPT.format(
        conversation=conv,
        task=state["task"],
        results=results,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    final_text = response.content

    # ★ 把 AI 回复写回 messages
    return {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }


# ── 9. 路由 ──────────────────────────────────────────
def after_planner(state: PlanExecuteState) -> str:
    return "executor" if state["plan"] else "finalizer"


def after_reflector(state: PlanExecuteState) -> str:
    if state["reflection_feedback"] is None:
        return "advance"
    if state["reflection_count"] >= 3:
        print(f"   ⚠️ 已达最大反思次数（2），保留当前结果")
        return "advance"
    return "redo"


def after_advance(state: PlanExecuteState) -> str:
    return "executor" if state["plan"] else "finalizer"


# ── 10. 构建图 ───────────────────────────────────────
def build_graph(checkpointer=None):
    builder = StateGraph(PlanExecuteState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("reflector", reflector_node)
    builder.add_node("advance", advance_node)
    builder.add_node("finalizer", finalizer_node)

    builder.add_edge(START, "planner")

    builder.add_conditional_edges("planner", after_planner, {
        "executor": "executor",
        "finalizer": "finalizer",
    })

    builder.add_edge("executor", "reflector")

    builder.add_conditional_edges("reflector", after_reflector, {
        "redo": "executor",
        "advance": "advance",
    })

    builder.add_conditional_edges("advance", after_advance, {
        "executor": "executor",
        "finalizer": "finalizer",
    })

    builder.add_edge("finalizer", END)

    agent = builder.compile(checkpointer=checkpointer) if checkpointer \
        else builder.compile()

    print("\n===== Mermaid 图 =====")
    print(agent.get_graph().draw_mermaid())
    print("======================\n")

    return agent


# ── 11. 主循环 ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Plan-and-Execute Agent（规划 + 执行 + 反思 + 短期/长期记忆）")
    print("指令：")
    print("  quit / exit / q   → 退出")
    print("  switch <名称>     → 切换用户（隔离长期记忆 + 短期会话）")
    print("  clear             → 清空当前会话的短期记忆")
    print("=" * 60)

    ensure_memory_dir()
    conn = sqlite3.connect("memory/plan_checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    agent = build_graph(checkpointer)

    user_id = "user_001"

    while True:
        user_input = input(f"\n👤 你（用户：{user_id}）：").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if user_input.lower() == "clear":
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM checkpoints WHERE thread_id LIKE ?",
                (f"{user_id}_%",),
            )
            cursor.execute(
                "DELETE FROM writes WHERE thread_id LIKE ?",
                (f"{user_id}_%",),
            )
            conn.commit()
            print("🧹 当前用户的会话状态已清空，长期记忆仍保留。")
            continue

        if user_input.lower().startswith("switch "):
            new_name = user_input.split(maxsplit=1)[1].strip()
            user_id = new_name if new_name.startswith("user_") else f"user_{new_name}"
            print(f"✅ 已切换到用户：{user_id}")
            continue

        if not user_input:
            continue

        # ★ 固定 thread_id，让 checkpointer 累积同一会话
        thread_id = f"{user_id}_session"

        try:
            result = agent.invoke(
                {
                    "task": user_input,
                    "messages": [HumanMessage(content=user_input)],   # ★ 追加本轮
                    "plan": [],
                    "current_step": "",
                    "past_steps": [],
                    "last_result": "",
                    "last_tool_log": [],
                    "reflection_feedback": None,
                    "reflection_count": 0,
                    "final_response": "",
                },
                config={
                    "configurable": {
                        "user_id": user_id,
                        "thread_id": thread_id,
                    }
                },
            )
            print(f"\n🤖 助手：{result['final_response']}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ 出错：{e}")