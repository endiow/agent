"""
长期记忆工具：使用 JSON 文件持久化用户信息。
长期记忆文件存放在 memory/ 目录下。
使用线程锁解决并发写入冲突。
"""

import os
import json
import threading
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

# 记忆存储目录和文件
MEMORY_DIR = "memory"
MEMORY_FILE = os.path.join(MEMORY_DIR, "long_term_memory.json")

# 全局线程锁，防止并发写入冲突
_memory_lock = threading.Lock()


@dataclass
class UserContext:
    """运行时上下文，包含当前用户 ID"""
    user_id: str


def ensure_memory_dir():
    """确保 memory/ 目录存在（供本模块和 agent.py 共用）"""
    os.makedirs(MEMORY_DIR, exist_ok=True)


def _load_all() -> dict:
    """加载整个记忆文件，返回 {user_id: {key: value}} 结构"""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all(data: dict):
    """写回整个记忆文件"""
    ensure_memory_dir()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@tool
def save_memory(key: str, value: str, runtime: ToolRuntime[UserContext]) -> str:
    """将用户的关键信息保存到长期记忆。当用户告诉你关于他自己的重要信息时调用此工具。
    例如：姓名、职业、喜好、习惯、研究方向、偏好设置等。

    Args:
        key: 要记住的信息的键，例如 '姓名'、'职业'、'偏好'
        value: 要记住的信息的值，例如 '小明'、'程序员'、'喜欢简洁回答'
    """
    user_id = runtime.context.user_id

    # 加锁：确保“读取-修改-写入”的原子性
    with _memory_lock:
        data = _load_all()
        if user_id not in data:
            data[user_id] = {}
        data[user_id][key] = value
        _save_all(data)

    return f"已记住：{key} = {value}"


@tool
def recall_memory(key: str, runtime: ToolRuntime[UserContext]) -> str:
    """查询用户的长期记忆。当用户询问关于他自己的具体信息时调用此工具。
    例如：'我叫什么名字'、'我做什么工作'、'你记得我的研究方向吗'。

    Args:
        key: 要查询的信息关键词，可以是问题原文中的关键部分，
             例如 '名字'、'姓名'、'职业'、'研究方向'、'偏好'。
             不需要精确匹配，系统会自动做模糊查找。
    """
    user_id = runtime.context.user_id
    with _memory_lock:
        data = _load_all()

    profile = data.get(user_id, {})
    if not profile:
        return "目前没有保存关于你的任何信息。"

    # 1. 精确匹配
    if key in profile:
        return f"{key}：{profile[key]}"

    # 2. 模糊匹配
    for k, v in profile.items():
        if key in k or k in key:
            return f"{k}：{v}"

    # 3. 兜底：返回全部记忆，让模型自己找
    items = [f"- {k}：{v}" for k, v in profile.items()]
    return f"没有精确找到「{key}」，但以下是我记得的全部信息：\n" + "\n".join(items)


@tool
def list_memories(runtime: ToolRuntime[UserContext]) -> str:
    """列出长期记忆中保存的所有信息。当用户询问‘你都记得我什么’时调用。"""
    user_id = runtime.context.user_id

    with _memory_lock:
        data = _load_all()

    profile = data.get(user_id, {})

    if not profile:
        return "目前没有保存关于你的任何信息。"

    items = [f"- {k}：{v}" for k, v in profile.items()]
    return "以下是我记得的关于你的信息：\n" + "\n".join(items)