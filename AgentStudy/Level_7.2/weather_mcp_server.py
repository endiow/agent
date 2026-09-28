"""
weather_mcp_server.py
将天气工具封装为 MCP Server
"""

from fastmcp import FastMCP
from weather_tool import (
    get_local_weather,
    get_weather_by_city,
)

# 创建 MCP Server 实例
mcp = FastMCP("WeatherService")


# ── 注册工具 ──────────────────────────────────────────

@mcp.tool()
def local_weather() -> str:
    """获取用户当前位置的实时天气。
    当用户询问“我这里天气怎么样”“当前天气”“本地天气”等
    没有指定城市、但隐含了基于当前位置的询问时，调用此工具。
    """
    return get_local_weather.invoke({})


@mcp.tool()
def weather_by_city(city: str) -> str:
    """查询指定城市的实时天气。
    当用户明确说出城市名（如“深圳天气怎么样”“北京今天热不热”）时，调用此工具。

    Args:
        city: 城市名称，如深圳、北京、上海
    """
    return get_weather_by_city.invoke({"city": city})


if __name__ == "__main__":
    # 以 stdio 模式运行（本地进程通信）
    mcp.run(transport="stdio")