"""
阶段六：多 Agent 协作系统（LLM Supervisor + 短期记忆）
Supervisor 每步思考 → 派一个专家 → 看结果 → 再思考 → ... → 汇总
复用阶段五的 RAG、记忆工具和用户上下文
支持 switch 切换用户 / clear 清空会话

架构：
        ┌───────────────┐
        │  Supervisor   │  ←── LLM 每轮思考：看历史轨迹，决定下一步
        └───────┬───────┘
        ┌───────┼───────┬──────────┐
        ▼       ▼       ▼          ▼
    research compute  info      memory
        └───────┴───┬───┴──────────┘
                    ▼
                 finalizer

短期记忆机制：
  - State 里声明 messages: Annotated[list, add_messages]
  - thread_id 固定为 {user_id}_session
  - finalizer 把 AI 回复写回 messages
  - Supervisor / Finalizer 的 prompt 里带 conversation
"""

import os
import json
import re
import sqlite3
from typing import TypedDict, Annotated
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


# ── 3. 基础工具 ──────────────────────────────────────

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
    now = __import__("datetime").datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


# ── 4. 专家 Agent 系统提示词 ─────────────────────────
RESEARCH_AGENT_PROMPT = """你是一个知识检索专家，只负责从本地知识库检索与任务相关的信息。

规则：
1. 必须调用 search_knowledge_base 工具进行检索，不要凭记忆回答。
2. 只报告工具实际返回的内容，不得编造、不得添加工具未提供的事实。
3. 如果知识库没有覆盖该主题，明确回复"知识库未覆盖该主题"。
4. 用简洁的中文总结检索到的要点，保留关键细节和来源。
5. 如果任务与知识库主题（Python / Agent / 机器学习）无关，直接回复
   "本任务无需知识库检索"。

输出：直接给出结果，不要 JSON，不要 markdown 代码块。
"""

COMPUTE_AGENT_PROMPT = """你是一个数学计算专家，只负责处理数值计算问题。

规则：
1. 调用 calculator 工具完成计算，不要心算。
2. 报告计算表达式和结果。
3. 如果任务中没有数值计算需求，直接回复"本任务无计算需求"。

输出：直接给出结果，不要 JSON，不要 markdown 代码块。
"""

INFO_AGENT_PROMPT = """你是一个实时信息查询专家，负责查询天气和当前时间。

规则：
1. 查询天气用 get_weather，查询时间用 get_current_time。
2. 忠实返回工具返回的内容，不得修饰、不得补充工具未提供的信息。
3. 如果任务与天气/时间无关，直接回复"本任务无天气/时间查询需求"。

输出：直接给出结果，不要 JSON，不要 markdown 代码块。
"""

MEMORY_AGENT_PROMPT = """你是一个用户长期记忆管理专家，负责读写用户的长期记忆（用户画像）。

规则：
1. 如果任务涉及用户个人信息（姓名、职业、偏好、研究方向、习惯等），
   调用 recall_memory 或 list_memories 查询。
2. 如果任务中用户告诉了你关于他自己的新信息，主动调用 save_memory 保存。
3. 只报告工具实际返回的内容，不得编造用户信息。
4. 如果任务与用户个人信息无关，直接回复"本任务无记忆读写需求"。
注意：你管理的是**用户画像**，不是**对话历史**。查询"上几轮问了什么"不属于你的职责。

输出：直接给出结果，不要 JSON，不要 markdown 代码块。
"""


# ── 5. 专家 Agent 工厂 ────────────────────────────────
@lru_cache(maxsize=1)
def get_research_agent():
    return create_agent(
        model=llm,
        tools=[search_knowledge_base],
        system_prompt=RESEARCH_AGENT_PROMPT,
        context_schema=UserContext,
    )


@lru_cache(maxsize=1)
def get_compute_agent():
    return create_agent(
        model=llm,
        tools=[calculator],
        system_prompt=COMPUTE_AGENT_PROMPT,
        context_schema=UserContext,
    )


@lru_cache(maxsize=1)
def get_info_agent():
    return create_agent(
        model=llm,
        tools=[get_weather, get_current_time],
        system_prompt=INFO_AGENT_PROMPT,
        context_schema=UserContext,
    )


@lru_cache(maxsize=1)
def get_memory_agent():
    return create_agent(
        model=llm,
        tools=[save_memory, recall_memory, list_memories],
        system_prompt=MEMORY_AGENT_PROMPT,
        context_schema=UserContext,
    )


EXPERTS = {
    "research_agent": {
        "getter": get_research_agent,
        "desc": "从本地知识库检索（覆盖 Python、Agent 概念、机器学习等）",
    },
    "compute_agent": {
        "getter": get_compute_agent,
        "desc": "数学计算、表达式求值",
    },
    "info_agent": {
        "getter": get_info_agent,
        "desc": "查询实时天气、当前时间",
    },
    "memory_agent": {
        "getter": get_memory_agent,
        "desc": "读写用户长期记忆（姓名、职业、偏好等）",
    },
}


# ── 6. State（★ 新增 messages） ──────────────────────
class HistoryItem(TypedDict):
    expert: str
    instruction: str
    result: str


class MultiAgentState(TypedDict):
    task: str
    messages: Annotated[list, add_messages]     # ★ 跨轮对话历史
    history: list                                # 本轮执行轨迹
    next: str
    next_instruction: str
    step_count: int
    final_response: str


MAX_STEPS = 10


# ── 7. 提示词（★ Supervisor / Finalizer 加入 conversation） ──
SUPERVISOR_PROMPT = """你是一个多 Agent 协作主管，采用「观察 → 思考 → 行动」的循环工作。

【历史对话】（跨轮上下文，帮助你理解"刚才""上面""之前"等指代）
{conversation}

【本轮用户任务】
{task}

【本轮已执行的步骤轨迹】
{history}

可用专家：
- research_agent：从本地知识库检索（Python / Agent / 机器学习等）
- compute_agent：数学计算、表达式求值
- info_agent：查询实时天气、当前时间
- memory_agent：读写用户长期记忆（姓名、职业、偏好等用户画像）

你的工作：
1. 结合历史对话理解本轮任务（注意"刚才""上面""之前"等指代）。
2. 观察轨迹：哪些信息已经拿到了？哪些还缺？
3. 判断下一步：
   - 需要动作 → 输出 next=某专家 + instruction=给该专家的具体可执行指令
   - 全部齐备 → 输出 next=FINISH

关键规则（防止重复与死循环）：
- 如果某个专家已经成功完成了某项任务，不要再派它做同一件事。
- 只有当上次执行失败/信息不完整时，才可以重派——且要换个角度、换种表达。
- instruction 要具体、自包含，把上下文写清楚，不要让专家猜：
    反例（太模糊）："处理用户问题"
    正例："从知识库检索：RAG 检索增强生成的定义和原理"
- 一次只派一个专家，只做一件事。
- FINISH 的时机：所有子诉求都被处理过，或明确判断无法再获得信息。
- **特别提醒**：如果任务本身可以从"历史对话"直接回答（如"我上两个问题是什么"、
  "你刚才为什么没查北京"），不要派任何专家，直接 FINISH，由汇总器从历史对话里提取答案。

严格输出 JSON（不要 markdown 代码块、不要解释）：
{{"next": "专家名 或 FINISH", "instruction": "给该专家的具体指令（FINISH 时留空）", "reason": "简短理由"}}
"""

EXPERT_USER_TEMPLATE = """你被分配到的具体任务：
{instruction}

【背景】用户原始诉求：{task}

【之前的步骤轨迹】
{history}

【要求】
- 只完成上面分配的这一个任务，不要越界做其他事。
- 如果需要工具，调用你手上的工具。
- 如果任务与你的职责无关，直接回复"本任务无 xxx 需求"。
- 简洁汇报结果，不要寒暄。

输出：直接给出结果，不要 JSON，不要 markdown 代码块。
"""

FINALIZER_PROMPT = """你是最终回答生成器。请根据历史对话和专家执行轨迹，为用户生成一段完整、自然的回答。

【历史对话】
{conversation}

【本轮用户任务】
{task}

【本轮专家执行轨迹】
{history}

要求：
1. 用流畅的中文回答，像正常对话一样。
2. 整合各专家的信息，不要罗列"专家X说"、"步骤1、步骤2"。
3. 忽略"本任务无 xxx 需求"这类占位输出。
4. 如果某专家明确表示"信息不足 / 知识库未覆盖"，如实告知，不要编造。
5. **如果任务本身是关于历史对话的**（如"我上两个问题是什么"），
   直接从【历史对话】里提取答案，不要提"专家"这个词。
6. 简洁明了，直接给结论。
"""


# ── 8. 辅助函数 ──────────────────────────────────────
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


def _format_history_for_supervisor(history: list) -> str:
    if not history:
        return "（尚未开始）"
    lines = []
    for i, h in enumerate(history, 1):
        r = h["result"]
        if len(r) > 300:
            r = r[:300] + "..."
        lines.append(
            f"[步骤 {i}] 专家：{h['expert']}\n"
            f"   指令：{h['instruction']}\n"
            f"   返回：{r}"
        )
    return "\n\n".join(lines)


def _format_history_for_expert(history: list) -> str:
    if not history:
        return "（无）"
    return "\n".join(
        f"- {h['expert']}：{h['instruction']} → {h['result'][:100]}"
        for h in history
    )


def _similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """粗糙的相似度判断（Jaccard），用于硬去重"""
    if not a or not b:
        return False
    if a == b:
        return True
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) > threshold


# ── 9. 节点 ──────────────────────────────────────────
def supervisor_node(state: MultiAgentState) -> dict:
    if state["step_count"] >= MAX_STEPS:
        print(f"\n🎯 主管：已达最大步数（{MAX_STEPS}），强制收尾")
        return {"next": "FINISH", "next_instruction": ""}

    # ★ 只看历史（去掉本轮那条 HumanMessage）
    conv = _format_conversation(state["messages"][:-1]) \
        if len(state["messages"]) > 1 else "（无历史）"

    prompt = SUPERVISOR_PROMPT.format(
        conversation=conv,
        task=state["task"],
        history=_format_history_for_supervisor(state["history"]),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    data = _parse_json(response.content)

    next_agent = data.get("next", "FINISH")
    instruction = (data.get("instruction") or "").strip()
    reason = data.get("reason", "")

    # 合法性校验
    if next_agent != "FINISH" and next_agent not in EXPERTS:
        print(f"   ⚠️ 非法专家名 {next_agent!r}，改为 FINISH")
        next_agent = "FINISH"

    # 硬去重：和上一步完全相同 → 强制结束
    if state["history"] and next_agent != "FINISH":
        last = state["history"][-1]
        if last["expert"] == next_agent and _similar(last["instruction"], instruction):
            print(f"   ⚠️ 检测到重复调用 [{next_agent}]，强制 FINISH")
            next_agent = "FINISH"

    # 派活必须带指令
    if next_agent != "FINISH" and not instruction:
        print(f"   ⚠️ 缺少 instruction，改为 FINISH")
        next_agent = "FINISH"

    if next_agent == "FINISH":
        print(f"\n🎯 主管：FINISH  （{reason or '信息已齐备'}）")
        return {"next": "FINISH", "next_instruction": ""}

    print(f"\n🎯 主管决策：{next_agent}")
    print(f"   指令：{instruction}")
    print(f"   理由：{reason}")

    return {
        "next": next_agent,
        "next_instruction": instruction,
        "step_count": state["step_count"] + 1,
    }


def make_expert_node(name: str):
    def node(state: MultiAgentState, config) -> dict:
        user_id = config.get("configurable", {}).get("user_id", "default")
        instruction = state["next_instruction"]

        if not instruction:
            print(f"   ⚠️ [{name}] 收到空指令，跳过")
            return {}

        print(f"\n   ▶️ [{name}] 执行：{instruction}")

        user_msg = EXPERT_USER_TEMPLATE.format(
            instruction=instruction,
            task=state["task"],
            history=_format_history_for_expert(state["history"]),
        )

        agent = EXPERTS[name]["getter"]()
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_msg}]},
                context=UserContext(user_id=user_id),
            )
            content = result["messages"][-1].content or ""
        except Exception as e:
            content = f"[执行失败] {e}"

        preview = content[:120] + ("..." if len(content) > 120 else "")
        print(f"   ✅ [{name}] 返回：{preview}")

        new_history = state["history"] + [{
            "expert": name,
            "instruction": instruction,
            "result": content,
        }]

        return {"history": new_history}

    return node


def finalizer_node(state: MultiAgentState) -> dict:
    print(f"\n📝 正在汇总最终答案...")

    # ★ 只看历史
    conv = _format_conversation(state["messages"][:-1]) \
        if len(state["messages"]) > 1 else "（无历史）"

    if state["history"]:
        history_text = "\n\n".join(
            f"【{h['expert']}｜{h['instruction']}】\n{h['result']}"
            for h in state["history"]
        )
    else:
        history_text = "（无）"

    prompt = FINALIZER_PROMPT.format(
        conversation=conv,
        task=state["task"],
        history=history_text,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    final_text = response.content

    # ★ 把 AI 回复写回 messages
    return {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }


# ── 10. 路由 ─────────────────────────────────────────
def after_supervisor(state: MultiAgentState) -> str:
    return "finalizer" if state["next"] == "FINISH" else state["next"]


# ── 11. 构建图 ───────────────────────────────────────
def build_graph(checkpointer=None):
    builder = StateGraph(MultiAgentState)

    builder.add_node("supervisor", supervisor_node)
    for name in EXPERTS:
        builder.add_node(name, make_expert_node(name))
    builder.add_node("finalizer", finalizer_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        after_supervisor,
        {**{n: n for n in EXPERTS}, "finalizer": "finalizer"},
    )

    for name in EXPERTS:
        builder.add_edge(name, "supervisor")

    builder.add_edge("finalizer", END)

    agent = builder.compile(checkpointer=checkpointer) if checkpointer \
        else builder.compile()

    print("\n===== Mermaid 图 =====")
    print(agent.get_graph().draw_mermaid())
    print("======================\n")

    return agent


# ── 12. 主循环 ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent 协作系统（LLM Supervisor + 短期/长期记忆）")
    print("指令：")
    print("  quit / exit / q   → 退出")
    print("  switch <名称>     → 切换用户（隔离长期记忆 + 短期会话）")
    print("  clear             → 清空当前会话的短期记忆")
    print("=" * 60)

    ensure_memory_dir()
    conn = sqlite3.connect(
        "memory/multi_agent_checkpoints.db", check_same_thread=False
    )
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
            print("🧹 当前会话的短期记忆已清空，长期记忆仍保留。")
            continue

        if user_input.lower().startswith("switch "):
            new_name = user_input.split(maxsplit=1)[1].strip()
            user_id = new_name if new_name.startswith("user_") else f"user_{new_name}"
            print(f"✅ 已切换到用户：{user_id}")
            continue

        if not user_input:
            continue

        # ★ 固定 thread_id
        thread_id = f"{user_id}_session"

        try:
            result = agent.invoke(
                {
                    "task": user_input,
                    "messages": [HumanMessage(content=user_input)],   # ★ 追加本轮
                    "history": [],
                    "next": "",
                    "next_instruction": "",
                    "step_count": 0,
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