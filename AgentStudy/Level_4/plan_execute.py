"""
Level 4: Plan-and-Execute Agent
复用 agent.py 中的工具、execute_tool、TOOLS_SCHEMA、client、MODEL
支持三种汇总模式：concise / faithful / detailed
"""

import json
import re
from agent import (
    client, MODEL, TOOLS_SCHEMA, execute_tool, TOOL_REGISTRY
)


# ============================================================
# 1. 规划器：让 LLM 输出 JSON 步骤列表
# ============================================================

PLANNER_PROMPT = """你是一个任务规划器。请将用户的复杂任务拆解成一系列清晰、有序、可独立执行的步骤。

要求：
1. 每个步骤只做一件事，描述要具体、可执行。
2. 步骤之间要有明确的先后顺序。
3. 只输出 JSON，不要任何解释、不要 markdown 代码块。
4. JSON 格式严格如下：
{{"steps": ["步骤1", "步骤2", "步骤3"]}}

你可以使用的工具（供参考，不要写进步骤描述）：
- calculator：计算数学表达式
- get_current_time：获取当前日期时间
- get_weather：查询城市天气
- write_file：写入文件
- search_web：搜索网页
- search_knowledge_base：检索本地知识库
- save_memory / recall_memory：长期记忆

用户任务：{task}

如果任务很简单（1-2 步可以完成），不要强行拆成更多步骤。
简单的事实查询、概念解释，通常 1 步就够了。
"""


def plan(task: str, verbose: bool = True) -> list:
    """调用 LLM 生成步骤列表"""
    prompt = PLANNER_PROMPT.format(task=task)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    text = response.choices[0].message.content.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        steps = data.get("steps", [])
        if verbose:
            print(f"\n📋 规划完成，共 {len(steps)} 步：")
            for i, s in enumerate(steps, 1):
                print(f"   {i}. {s}")
        return steps
    except Exception as e:
        print(f"[规划失败] JSON 解析错误：{e}\n原始输出：{text}")
        return []


# ============================================================
# 2. 执行器：复用 ReAct 循环，执行单个步骤
# ============================================================

EXECUTOR_PROMPT = """你是一个任务执行器。请专注完成当前这一步的任务。

原始总任务：{task}

已完成的步骤及结果：
{context}

当前需要执行的步骤：{step}

请调用必要的工具完成任务，然后给出这一步的执行结果。
如果这一步不需要工具，直接给出结果。

完成这一步后，请严格按以下 JSON 输出：
{{"status": "success 或 failed", "result": "这一步的执行结果"}}

执行完成后，你必须给出这一步的【实际产出】，而不是描述你打算怎么做。
如果任务是“整合信息”，你必须直接输出整合后的完整内容。

你的整合内容必须严格来自前序步骤提供的信息。如果信息不完整，
请在结果中明确标注"[信息不足]"，而不是用通用知识补全。
"""


def execute_step(step: str, task: str, completed: list,
                 max_steps: int = 5, verbose: bool = True):
    """
    用 ReAct 循环执行单个步骤。
    返回 (status, result)：status 是 "success" 或 "failed"，result 是纯文本结果。
    """
    if completed:
        context = "\n".join(
            f"- {c['step']} → {c['result']}" for c in completed
        )
    else:
        context = "（无）"

    messages = [
        {"role": "system",
         "content": EXECUTOR_PROMPT.format(task=task, context=context, step=step)},
        {"role": "user", "content": f"请执行：{step}"},
    ]

    final_content = "（执行步数超限）"

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                if verbose:
                    print(f"      🔧 {tool_name}({tool_args})")
                result = execute_tool(tool_name, tool_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            final_content = msg.content or "（本步无输出）"
            break

    # ---- 解析执行器输出的 JSON ----
    text = final_content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        status = data.get("status", "success").lower()
        result = data.get("result", text)
        return status, result
    except Exception:
        return "success", final_content


# ============================================================
# 3. 重规划器：步骤失败时，重新规划剩余步骤
# ============================================================

REPLANNER_PROMPT = """你是一个任务重规划器。之前计划中的某一步执行失败了。

原始任务：{task}

已完成的步骤及结果：
{completed}

失败的步骤：{failed_step}
失败原因：{error}

请根据当前情况，重新规划剩余的步骤（不要重复已完成的步骤）。
只输出 JSON，格式：{{"steps": ["...", "..."]}}
"""


def replan(task: str, completed: list, failed_step: str,
           error: str, verbose: bool = True) -> list:
    """重新规划剩余步骤"""
    completed_str = "\n".join(
        f"- {c['step']} → {c['result']}" for c in completed
    ) or "（无）"

    prompt = REPLANNER_PROMPT.format(
        task=task, completed=completed_str,
        failed_step=failed_step, error=error
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    text = response.choices[0].message.content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        steps = json.loads(text).get("steps", [])
        if verbose:
            print(f"\n🔁 重新规划，共 {len(steps)} 步：")
            for i, s in enumerate(steps, 1):
                print(f"   {i}. {s}")
        return steps
    except Exception as e:
        print(f"[重规划失败] {e}")
        return []


# ============================================================
# 4. 汇总器：三种模式（concise / faithful / detailed）
# ============================================================

# ---- 4a. concise 模式：默认，自然流畅，允许信息压缩 ----

SUMMARIZER_PROMPT = """请根据以下已完成的任务步骤和结果，为用户生成一段完整、自然的最终回答。

原始任务：{task}

各步骤结果：
{results}

要求：用简洁的中文回答，直接说结论，不要罗列"步骤1、步骤2"，要像正常对话一样流畅。
"""


def summarize(task: str, completed: list) -> str:
    """concise 模式：LLM 汇总，自然流畅，有信息压缩"""
    results = "\n".join(
        f"- {c['step']}：{c['result']}" for c in completed
    )
    prompt = SUMMARIZER_PROMPT.format(task=task, results=results)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---- 4b. faithful 模式：直接拼接，零 LLM 加工 ----

def summarize_faithful(task: str, completed: list) -> str:
    """faithful 模式：直接拼接各步骤结果，完全不做 LLM 加工"""
    parts = []
    for c in completed:
        parts.append(f"【{c['step']}】\n{c['result']}")
    return "\n\n".join(parts)


# ---- 4c. detailed 模式：保留全部要点，只润色语言 ----

DETAILED_PROMPT = """请把以下步骤结果改写成一段流畅自然的中文回答。

原始任务：{task}

各步骤结果：
{results}

严格要求：
1. 不得删减或压缩任何信息点。原结果里提到的所有事实、数字、名词、例子，
   都必须在最终回答中体现。
2. 不得用你自己的知识补充任何新信息。如果原结果里没有，就不要写。
3. 只做"表达润色"——把要点连成通顺的句子，不做"内容取舍"。
4. 如果原结果里有"[信息不足]"之类的标注，必须保留这个标注，不能省略。
"""


def summarize_detailed(task: str, completed: list) -> str:
    """detailed 模式：保留全部要点，只做语言润色"""
    results = "\n".join(
        f"- {c['step']}：{c['result']}" for c in completed
    )
    prompt = DETAILED_PROMPT.format(task=task, results=results)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,   # 降低随机性，减少自行发挥
    )
    return response.choices[0].message.content


# ============================================================
# 5. 主流程：规划 → 执行 → 重规划 → 汇总
# ============================================================

def run_plan_execute(task: str, max_replans: int = 2,
                     summary_mode: str = "concise", verbose: bool = True):
    """
    Plan-and-Execute 主流程。

    参数：
        summary_mode:
            - "concise"  : LLM 汇总，自然流畅，有信息压缩（默认）
            - "faithful" : 直接拼接，零失真，格式简单
            - "detailed" : LLM 润色，保留全部要点，只改语言不改内容
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"🎯 任务：{task}")
        print(f"📝 汇总模式：{summary_mode}")
        print(f"{'='*50}")

    # 1. 规划
    steps = plan(task, verbose=verbose)
    if not steps:
        return "规划失败，无法执行。"

    completed = []
    replan_count = 0

    # 2. 逐步执行
    while steps:
        step = steps[0]
        if verbose:
            print(f"\n▶️ 执行：{step}")

        status, result = execute_step(step, task, completed, verbose=verbose)

        # 3. 判断失败：直接看执行器返回的 status
        if status == "failed":
            if verbose:
                print(f"   ⚠️ 步骤失败：{result[:80]}")
            if replan_count >= max_replans:
                return f"任务失败（超过最大重规划次数）：{result}"
            steps = replan(task, completed, step, result, verbose=verbose)
            replan_count += 1
            continue

        # 4. 记录成功（存纯净的 result，不含 status 字段）
        if verbose:
            print(f"   ✅ 结果：{result}")
            print(f"   （执行器输出长度：{len(result)} 字符）")
        completed.append({"step": step, "result": result})
        steps = steps[1:]

    # 5. 汇总：根据模式分发
    if verbose:
        print(f"\n{'='*50}")
        print(f"📝 正在汇总最终答案（模式：{summary_mode}）...")

    if summary_mode == "faithful":
        final = summarize_faithful(task, completed)
    elif summary_mode == "detailed":
        final = summarize_detailed(task, completed)
    else:  # concise（默认）
        final = summarize(task, completed)

    if verbose:
        print(f"\n✅ 最终答案：\n{final}")
    return final


# ============================================================
# 6. 交互式运行
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("🧠 Plan-and-Execute Agent 已启动")
    print("指令：")
    print("  quit / exit / q  → 退出")
    print("  mode concise     → LLM 汇总（默认，流畅）")
    print("  mode faithful    → 原文拼接（零失真）")
    print("  mode detailed    → 详细润色（保留全部要点）")
    print("=" * 50)

    current_mode = "concise"

    while True:
        user_input = input(f"\n👤 你（当前模式：{current_mode}）：").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if user_input.lower().startswith("mode "):
            new_mode = user_input.split(maxsplit=1)[1].strip().lower()
            if new_mode in ("concise", "faithful", "detailed"):
                current_mode = new_mode
                print(f"✅ 已切换到 {current_mode} 模式")
            else:
                print("❌ 无效模式，请用 concise / faithful / detailed")
            continue

        if not user_input:
            continue

        try:
            run_plan_execute(
                user_input,
                max_replans=2,
                summary_mode=current_mode,
                verbose=True
            )
        except Exception as e:
            print(f"❌ 运行出错：{e}")