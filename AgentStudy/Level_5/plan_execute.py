"""
Level 5: Plan-and-Execute Agent + Reflection (Evidence-Based)
复用 agent.py 中的工具、execute_tool、TOOLS_SCHEMA、client、MODEL
支持三种汇总模式：concise / faithful / detailed
支持基于证据的反思：审查者能看到工具调用记录，避免盲审
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

关于 status 字段：
- 如果工具调用失败（网络错误、参数错误、工具不存在）→ status = "failed"
- 如果工具调用成功，但返回内容不足以回答当前步骤 → status = "success"，
  并在 result 里写"[信息不足：具体缺失什么]"
- 只有"信息不足"时，结果里必须保留这个标注，不得编造

执行完成后，你必须给出这一步的【实际产出】，而不是描述你打算怎么做。
如果任务是"整合信息"，你必须直接输出整合后的完整内容。

你的整合内容必须严格来自前序步骤提供的信息。如果信息不完整，
请在结果中明确标注"[信息不足]"，而不是用通用知识补全。

当工具返回的内容明显在句子/段落中间被截断，且你无法从工具返回中
直接得到该信息时，必须写"[信息不足：具体缺失内容]",
不得凭常识或上下文补全。

例如，如果工具返回"机器学习主要分为三"，后面没有内容，
你应该写"[信息不足：三类学习方式的具体名称未被检索到]",
而不是自行补成"监督学习、无监督学习和强化学习"。
"""


def execute_step(step: str, task: str, completed: list,
                 feedback: str = None,
                 max_steps: int = 5, verbose: bool = True):
    """
    用 ReAct 循环执行单个步骤。
    返回 (status, result, tool_log)。
        - status: "success" 或 "failed"
        - result: 纯文本结果
        - tool_log: 本步中所有工具调用的记录，用于审查者核查
    """
    if completed:
        context = "\n".join(
            f"- {c['step']} → {c['result']}" for c in completed
        )
    else:
        context = "（无）"

    system_content = EXECUTOR_PROMPT.format(
        task=task, context=context, step=step
    )

    if feedback:
        system_content += (
            f"\n\n【重要】上一次执行被审查者拒绝了，反馈如下：\n{feedback}\n"
            f"请针对这些问题重新执行，务必避免相同的错误。"
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"请执行：{step}"},
    ]

    final_content = "（执行步数超限）"
    tool_log = []   # ← 记录本步所有工具调用

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

                # 记录工具调用证据
                tool_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            final_content = msg.content or "（本步无输出）"
            break

    text = final_content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        status = data.get("status", "success").lower()
        result = data.get("result", text)
        return status, result, tool_log
    except Exception:
        return "success", final_content, tool_log


# ============================================================
# 2b. 审查者：基于证据检查执行器输出质量
# ============================================================

REFLECTOR_PROMPT = """你是一个严格的审查者，负责检查任务执行器对某一步的执行结果。

原始任务：{task}
当前步骤：{step}

已完成的步骤及结果：
{context}

【执行器的工具调用证据】
{evidence}

【执行器的最终输出】
{result}

请基于以上"工具调用证据"和"最终输出"进行严格审查。审查维度：

1. **忠实度**：最终输出中的每个事实，是否能在"工具调用证据"或"已完成步骤"中找到依据？
   执行器有没有在证据之外添加工具没返回的内容？

2. **完整性**：这一步的任务是否真正完成？有没有遗漏关键信息？

3. **准确性**：有没有事实错误、逻辑矛盾或计算错误？

4. **诚实性**：如果信息不足，执行器有没有明确标注"[信息不足]",
   而不是用通用知识悄悄补全或编造来源？

审查标准：
- 如果执行器有工具调用，且最终输出的事实与工具返回一致 → 通过。
- 如果最终输出包含"工具未返回"的事实（无论是否合理） → 不通过。
- 如果执行器编造了"来源文档名"、"更新时间"之类的元数据 → 不通过。
- 如果最终输出只是"描述了打算做什么"而没有实际产出 → 不通过。
- 如果这一步不需要工具（例如纯计算或纯逻辑推理），且输出合理 → 通过。
- 如果执行器对工具返回中明显被截断的内容，正确地标注了"[信息不足]",
  应视为通过（这是诚实的行为，不是缺陷）。
- 只有当执行器对截断内容自行"补全"却未说明时，才判定为不通过。

请严格按以下 JSON 输出，不要任何解释、不要 markdown 代码块：
{{"passed": true 或 false, "critique": "如果不通过，写具体问题和证据位置；如果通过，写'无问题'"}}
"""


def reflect(step: str, task: str, completed: list, result: str,
            tool_log: list = None, verbose: bool = True):
    """
    审查执行器的输出（带工具调用证据）。
    返回 (passed: bool, critique: str)。
    """
    if completed:
        context = "\n".join(
            f"- {c['step']} → {c['result'][:200]}" for c in completed
        )
    else:
        context = "（无）"

    # 格式化工具调用证据
    if tool_log:
        evidence_lines = []
        for i, t in enumerate(tool_log, 1):
            evidence_lines.append(
                f"[工具调用 {i}] {t['tool']}({t['args']})\n"
                f"[工具返回 {i}] {t['result'][:800]}"
            )
        evidence = "\n\n".join(evidence_lines)
    else:
        evidence = "（执行器没有调用任何工具）"

    prompt = REFLECTOR_PROMPT.format(
        task=task, step=step, context=context,
        evidence=evidence, result=result
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
        data = json.loads(text)
        passed = bool(data.get("passed", True))
        critique = data.get("critique", "无问题")
        return passed, critique
    except Exception as e:
        if verbose:
            print(f"      [审查解析失败] {e}，兜底放行")
        return True, "无问题"


# ============================================================
# 2c. 带反思的步骤执行：执行 → 审查 → 重做
# ============================================================

def execute_step_with_reflection(step: str, task: str, completed: list,
                                 max_reflections: int = 2,
                                 verbose: bool = True):
    """
    执行单步，并自动进行多轮反思。
    返回 (status, result, reflection_log)。
    """
    feedback = None
    reflection_log = []

    for attempt in range(max_reflections + 1):
        if verbose:
            if attempt == 0:
                print(f"\n   ▶️ 执行：{step}")
            else:
                print(f"\n   🔁 第 {attempt} 次重做：{step}")

        # 接收工具调用证据
        status, result, tool_log = execute_step(
            step, task, completed,
            feedback=feedback, verbose=verbose
        )

        if status == "failed":
            reflection_log.append({
                "attempt": attempt,
                "passed": False,
                "critique": "执行器返回 failed 状态",
            })
            return status, result, reflection_log

        if verbose:
            print(f"   ✅ 执行结果（{len(result)} 字符）：{result[:150]}...")

        # 把 tool_log 传给审查者
        passed, critique = reflect(
            step, task, completed, result,
            tool_log=tool_log, verbose=verbose
        )
        reflection_log.append({
            "attempt": attempt,
            "passed": passed,
            "critique": critique,
        })

        if passed:
            if verbose:
                print(f"   ✅ 审查通过")
            return status, result, reflection_log

        if verbose:
            print(f"   ⚠️ 审查未通过：{critique[:150]}")

        if attempt < max_reflections:
            feedback = critique
        else:
            if verbose:
                print(f"   ⚠️ 达到最大反思次数（{max_reflections}），保留当前结果")
            return status, result, reflection_log

    return "success", result, reflection_log


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

重要：如果反复尝试（2 次以上）都无法从知识库、搜索等工具获得某个信息，
说明该信息在现有资源中不存在。此时不要继续规划"换个关键词再检索"，
而是直接返回：
{"steps": [], "reason": "无法获取该信息，请向用户说明信息缺失。"}

空 steps 列表表示任务终止。
"""


def replan(task: str, completed: list, failed_step: str,
           error: str, verbose: bool = True):
    """重新规划剩余步骤。返回 (steps, reason)。"""
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
        data = json.loads(text)
        steps = data.get("steps", [])
        reason = data.get("reason", "")
        if verbose:
            if steps:
                print(f"\n🔁 重新规划，共 {len(steps)} 步：")
                for i, s in enumerate(steps, 1):
                    print(f"   {i}. {s}")
            else:
                print(f"\n🛑 重规划器决定终止：{reason}")
        return steps, reason
    except Exception as e:
        print(f"[重规划失败] {e}")
        return [], f"重规划 JSON 解析失败：{e}"


# ============================================================
# 4. 汇总器：三种模式（concise / faithful / detailed）
# ============================================================

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


def summarize_faithful(task: str, completed: list) -> str:
    """faithful 模式：直接拼接各步骤结果，完全不做 LLM 加工"""
    parts = []
    for c in completed:
        parts.append(f"【{c['step']}】\n{c['result']}")
    return "\n\n".join(parts)


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
        temperature=0.1,
    )
    return response.choices[0].message.content


# ============================================================
# 5. 主流程：规划 → 执行 → 重规划 → 汇总
# ============================================================

def run_plan_execute(task: str, max_replans: int = 2,
                     max_reflections: int = 2,
                     summary_mode: str = "concise",
                     verbose: bool = True):
    """
    Plan-and-Execute 主流程（含 Evidence-Based Reflection）。
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"🎯 任务：{task}")
        print(f"📝 汇总模式：{summary_mode}")
        print(f"🔍 每步最多反思 {max_reflections} 次")
        print(f"{'='*50}")

    steps = plan(task, verbose=verbose)
    if not steps:
        return "规划失败，无法执行。"

    completed = []
    replan_count = 0
    total_reflections = 0
    total_rejections = 0

    while steps:
        step = steps[0]

        status, result, ref_log = execute_step_with_reflection(
            step, task, completed,
            max_reflections=max_reflections,
            verbose=verbose
        )

        total_reflections += len(ref_log)
        total_rejections += sum(1 for r in ref_log if not r["passed"])

        # ==================== 改动点 ====================
        if status == "failed":
            if verbose:
                print(f"   ⚠️ 步骤失败：{result[:80]}")

            # 超过最大重规划次数 → 直接终止
            if replan_count >= max_replans:
                return f"任务失败（超过最大重规划次数）：{result}"

            # 重规划器现在返回 (steps, reason)
            new_steps, reason = replan(
                task, completed, step, result, verbose=verbose
            )

            # 重规划器主动放弃（空 steps）→ 优雅终止，不进入汇总
            if not new_steps:
                if verbose:
                    print(f"\n🛑 重规划器决定终止任务：{reason}")
                return (
                    f"任务无法继续完成。\n"
                    f"失败步骤：{step}\n"
                    f"原因：{reason or result}"
                )

            # 有新的步骤 → 继续执行
            steps = new_steps
            replan_count += 1
            continue
        # =================================================

        completed.append({"step": step, "result": result})
        steps = steps[1:]

    if verbose:
        print(f"\n{'='*50}")
        print(f"📝 正在汇总最终答案（模式：{summary_mode}）...")
        print(f"🔍 反思统计：共审查 {total_reflections} 次，"
              f"被拒绝 {total_rejections} 次")

    if summary_mode == "faithful":
        final = summarize_faithful(task, completed)
    elif summary_mode == "detailed":
        final = summarize_detailed(task, completed)
    else:
        final = summarize(task, completed)

    if verbose:
        print(f"\n✅ 最终答案：\n{final}")
    return final


# ============================================================
# 6. 交互式运行
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("🧠 Plan-and-Execute Agent + Evidence-Based Reflection 已启动")
    print("指令：")
    print("  quit / exit / q  → 退出")
    print("  mode concise     → LLM 汇总（默认）")
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
                print("❌ 无效模式")
            continue

        if not user_input:
            continue

        try:
            run_plan_execute(
                user_input,
                max_replans=2,
                max_reflections=2,
                summary_mode=current_mode,
                verbose=True
            )
        except Exception as e:
            print(f"❌ 运行出错：{e}")