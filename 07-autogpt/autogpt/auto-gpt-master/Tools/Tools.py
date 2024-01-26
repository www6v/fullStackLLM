import warnings

from pydantic import Field

warnings.filterwarnings("ignore")
from langchain import SerpAPIWrapper
from langchain.agents import Tool
from py_expression_eval import Parser
from langchain.tools import GooglePlacesTool
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.tools import StructuredTool
from .WebpageUtil import read_webpage
from langchain.tools import tool
from ctparse import ctparse

search_wrapper = SerpAPIWrapper()

search_tool = Tool.from_function(
    func=search_wrapper.run,
    name="Search",
    description="用于通过搜索引擎从互联网搜索信息",
)

@tool("UserLocation")
def mocked_location_tool(foo: str) -> str:
    """用于获取用户当前的位置（城市、区域）"""
    return "Chaoyang District, Beijing, CN"

@tool("Calendar")
def calendar_tool(
        date_exp: str = Field(description="Date expression to be parsed. It must be in English."),
) -> str:
    """用于查询和计算日期/时间"""
    res = ctparse(date_exp)
    date = res.resolution
    return date.dt.strftime("%c")

def evaluate(expr: str) -> str:
    parser = Parser()
    return str(parser.parse(expr).evaluate({}))


calculator_tool = Tool.from_function(
    func=evaluate,
    name="Calculator",
    description="用于计算一个数学表达式的值",
)

g_places = GooglePlacesTool()

map_tool = Tool.from_function(
    func=g_places.run,
    name="Map",
    description="用于搜索一个地名实体的详细地址",
)

weather = OpenWeatherMapAPIWrapper()

weather_tool = Tool.from_function(
    func=weather.run,
    name="Weather",
    description="用于获取一个城市的天气信息，城市名需要以英文输入",
)

webpage_tool = StructuredTool.from_function(
    func=read_webpage,
    name="ReadWebpage",
    description="用于获取一个网页的内容",
)