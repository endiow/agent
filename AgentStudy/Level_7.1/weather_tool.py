"""
本地实时天气工具
- 位置：通过 IP 定位获取城市级经纬度
- 天气：调用和风天气 JWT 认证接口
- 封装：作为 LangChain @tool 供 Agent 调用
"""

import os
import time
import jwt
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ── 配置（从 .env 读取）────────────────────────────────
QWEATHER_PROJECT_ID = os.getenv("QWEATHER_PROJECT_ID")
QWEATHER_KEY_ID = os.getenv("QWEATHER_KEY_ID")
QWEATHER_DEVELOPER_ID=os.getenv("QWEATHER_DEVELOPER_ID")
QWEATHER_PRIVATE_KEY_PATH = os.getenv("QWEATHER_PRIVATE_KEY_PATH", "ed25519-private.pem")
QWEATHER_API_HOST = os.getenv("QWEATHER_API_HOST")


# ── JWT 生成与缓存 ─────────────────────────────────────
_jwt_cache = {"token": None, "exp": 0}


def _generate_jwt() -> str:
    """生成或复用有效的 JWT Token"""
    now = int(time.time())

    # 缓存还有 30 秒以上有效期就直接复用
    if _jwt_cache["token"] and _jwt_cache["exp"] - now > 30:
        return _jwt_cache["token"]

    with open(QWEATHER_PRIVATE_KEY_PATH, "rb") as f:
        private_key = f.read()

    payload = {
        "iss": QWEATHER_DEVELOPER_ID,
        "sub": QWEATHER_PROJECT_ID,
        "iat": now,
        "exp": now + 3600,   # 有效期 1 小时
    }
    headers = {"kid": QWEATHER_KEY_ID, "alg": "EdDSA"}

    token = jwt.encode(payload, private_key, algorithm="EdDSA", headers=headers)
    # print(token)

    _jwt_cache["token"] = token
    _jwt_cache["exp"] = payload["exp"]
    return token


# ── IP 定位 ────────────────────────────────────────────
def _get_current_location() -> tuple:
    """通过 IP 获取当前大致经纬度和城市名"""
    resp = requests.get(
        "http://ip-api.com/json/",
        params={"lang": "zh-CN"},
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "success":
        raise Exception(data.get("message", "IP 定位失败"))

    return data["lat"], data["lon"], data.get("city", "未知城市")


# ── 和风天气 API ───────────────────────────────────────
def _fetch_weather(location: str) -> dict:
    """调用和风天气实时天气接口"""
    url = f"{QWEATHER_API_HOST}/v7/weather/now"
    headers = {
        "Authorization": f"Bearer {_generate_jwt()}",
        "Accept-Encoding": "gzip",
        "User-Agent": "MyWeatherApp/1.0 (contact@example.com)",
    }
    params = {"location": location, "lang": "zh"}

    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _format_error(code: str) -> str:
    """把和风天气的业务错误码翻译成中文"""
    error_map = {
        "400": "请求参数错误。",
        "401": "JWT 认证失败，请检查项目 ID、凭据 ID 或公钥是否配置正确。",
        "402": "超过访问次数或余额不足。",
        "403": "无权限访问，可能是 API Host 配置错误。",
        "404": "查询的地区暂不支持。",
        "429": "请求过于频繁，请稍后再试。",
        "500": "和风天气服务端错误。",
    }
    return error_map.get(code, f"未知错误码：{code}")


# ── 工具定义 ───────────────────────────────────────────
@tool
def get_local_weather() -> str:
    """获取用户当前位置的实时天气。
    当用户询问“我这里天气怎么样”“当前天气”“本地天气”等
    没有指定城市、但隐含了基于当前位置的询问时，调用此工具。
    """
    missing = [
        name for name, val in [
            ("QWEATHER_PROJECT_ID", QWEATHER_PROJECT_ID),
            ("QWEATHER_KEY_ID", QWEATHER_KEY_ID),
            ("QWEATHER_API_HOST", QWEATHER_API_HOST),
        ] if not val
    ]
    if missing:
        return f"配置缺失：请检查 .env 中的 {', '.join(missing)}"

    if not os.path.exists(QWEATHER_PRIVATE_KEY_PATH):
        return f"私钥文件不存在：{QWEATHER_PRIVATE_KEY_PATH}"

    try:
        # 1. 获取当前位置
        lat, lon, city_name = _get_current_location()
        # 和风天气要求：经度在前，纬度在后，最多两位小数
        location = f"{lon:.2f},{lat:.2f}"

        # 2. 查询天气
        data = _fetch_weather(location)
        if data.get("code") != "200":
            return f"查询天气失败：{_format_error(data.get('code'))}"

        # 3. 格式化
        now = data.get("now", {})
        if not now:
            return f"未能获取到 {city_name} 的天气数据。"

        return (
            f"{city_name} 当前天气：{now.get('text', '未知')}，"
            f"温度 {now.get('temp', 'N/A')}°C，"
            f"体感 {now.get('feelsLike', 'N/A')}°C，"
            f"湿度 {now.get('humidity', 'N/A')}%，"
            f"风向 {now.get('windDir', 'N/A')}，"
            f"风力 {now.get('windScale', 'N/A')} 级"
        )

    except requests.exceptions.Timeout:
        return "网络请求超时，请稍后再试。"
    except requests.exceptions.RequestException as e:
        return f"网络请求失败：{e}"
    except Exception as e:
        return f"获取天气时发生错误：{e}"


@tool
def get_weather_by_city(city: str) -> str:
    """查询指定城市的实时天气。
    当用户明确说出城市名（如“深圳天气怎么样”“北京今天热不热”）时，调用此工具。

    Args:
        city: 城市名称，如深圳、北京、上海
    """
    if not QWEATHER_API_HOST:
        return "配置缺失：QWEATHER_API_HOST 未设置"

    try:
        # 用和风天气的城市搜索 API 把城市名转成 location_id
        url = f"{QWEATHER_API_HOST}/geo/v2/city/lookup"
        headers = {
            "Authorization": f"Bearer {_generate_jwt()}",
            "Accept-Encoding": "gzip",
            "User-Agent": "MyWeatherApp/1.0 (contact@example.com)",
        }
        resp = requests.get(
            url,
            headers=headers,
            params={"location": city, "lang": "zh"},
            timeout=10,
        )
        resp.raise_for_status()
        geo = resp.json()

        if geo.get("code") != "200" or not geo.get("location"):
            return f"未找到城市「{city}」的信息。"

        location_id = geo["location"][0]["id"]
        city_name = geo["location"][0]["name"]

        # 用 location_id 查询天气
        data = _fetch_weather(location_id)
        if data.get("code") != "200":
            return f"查询天气失败：{_format_error(data.get('code'))}"

        now = data.get("now", {})
        return (
            f"{city_name} 当前天气：{now.get('text', '未知')}，"
            f"温度 {now.get('temp', 'N/A')}°C，"
            f"体感 {now.get('feelsLike', 'N/A')}°C，"
            f"湿度 {now.get('humidity', 'N/A')}%，"
            f"风向 {now.get('windDir', 'N/A')}，"
            f"风力 {now.get('windScale', 'N/A')} 级"
        )

    except Exception as e:
        return f"查询 {city} 天气时发生错误：{e}"


# ── 单独测试 ───────────────────────────────────────────
if __name__ == "__main__":
    print("测试本地天气：")
    print(get_local_weather.invoke({}))
    print("\n测试按城市查询：")
    print(get_weather_by_city.invoke({"city": "深圳"}))