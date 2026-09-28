"""
记忆管理工具：短期记忆（会话历史）+ 长期记忆（关键事实）
"""

import os
import json

SESSION_FILE = "session.json"
MEMORY_FILE = "long_term_memory.json"


# ============================================================
# 短期记忆：会话历史
# ============================================================

def load_session() -> list:
    """加载会话历史，返回一个 messages 列表（不含 system）"""
    if not os.path.exists(SESSION_FILE):
        return []
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_session(history: list):
    """保存会话历史到本地文件"""
    # 只保留最近的 N 条，防止文件无限增长
    MAX_HISTORY = 20
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def clear_session():
    """清空会话历史（用于重新开始）"""
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)


# ============================================================
# 长期记忆：关键事实
# ============================================================

def load_memory() -> dict:
    """加载长期记忆（键值对）"""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_long_term_memory(key: str, value: str) -> str:
    """将关键事实写入长期记忆"""
    memory = load_memory()
    memory[key] = value
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return f"已记住：{key} = {value}"


def recall_long_term_memory(key: str) -> str:
    """从长期记忆里读取某个键"""
    memory = load_memory()
    if key in memory:
        return memory[key]
    return f"没有找到关于「{key}」的记忆"


def format_memory_for_prompt() -> str:
    """把长期记忆格式化成一段文字，方便注入到 system prompt"""
    memory = load_memory()
    if not memory:
        return ""
    lines = ["\n\n以下是你已知的关于用户的信息："]
    for k, v in memory.items():
        lines.append(f"- {k}：{v}")
    return "\n".join(lines)