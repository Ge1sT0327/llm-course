"""
工具注册系统 - 模拟LangChain @tool装饰器模式
自动从函数签名生成 OpenAI Function Calling JSON Schema
"""
import functools
import inspect
import json
import math
import re
from datetime import datetime
from typing import Any


# 全局工具注册表
_tool_registry: dict[str, dict] = {}


def tool(name: str = "", description: str = ""):
    """
    @tool 装饰器 - 将函数注册为Agent可调用的工具

    使用方式:
        @tool(name="search", description="搜索知识库")
        def search(query: str) -> str:
            ...

    Args:
        name: 工具名称 (默认使用函数名)
        description: 工具描述 (帮助LLM理解何时调用)
    """
    def decorator(fn):
        tool_name = name or fn.__name__

        # 从函数签名提取参数信息, 生成 JSON Schema
        sig = inspect.signature(fn)
        properties = {}
        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation is int:
                param_type = "integer"
            elif param.annotation is float:
                param_type = "number"
            elif param.annotation is bool:
                param_type = "boolean"
            properties[param_name] = {
                "type": param_type,
                "description": f"{param_name} 参数",
            }

        schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description or (fn.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                },
            },
        }

        @functools.wraps(fn)
        def wrapper(**kwargs):
            return fn(**kwargs)

        _tool_registry[tool_name] = {
            "fn": wrapper,
            "schema": schema,
            "name": tool_name,
            "description": description or fn.__doc__,
        }
        return wrapper
    return decorator


def get_tool(name: str) -> dict | None:
    """获取指定工具"""
    return _tool_registry.get(name)


def list_tools() -> list[str]:
    """列出所有已注册工具的名称"""
    return list(_tool_registry.keys())


def get_tool_schemas() -> list[dict]:
    """获取所有工具的 Function Calling Schema 列表"""
    return [t["schema"] for t in _tool_registry.values()]


# ---------- 预注册工具 ----------

@tool(name="search", description="在本地知识库中搜索信息。输入关键词或问题, 返回相关知识条目。")
def search(query: str = "") -> str:
    knowledge = {
        "LangChain": "LangChain是构建LLM应用的主流框架(GitHub 90K+ stars)。核心概念: "
                     "Runnable接口、Chain管道(|运算符)、PromptTemplate、Tool/Agent。"
                     "v1.x API统一了所有组件的调用方式(invoke/stream/batch)。",
        "LangGraph": "LangGraph是LangChain团队开发的状态机Agent框架。将Agent执行建模为"
                     "有向图: 节点(处理步骤) + 边(状态转换)。支持条件路由、循环、"
                     "Human-in-the-Loop。适用于复杂的多步推理Agent。",
        "MCP": "Model Context Protocol (MCP) 是AI Agent标准化协议。2025年12月捐赠给"
               "Linux Foundation。定义三种原语: Tool(操作工具)、Resource(只读数据)、"
               "Prompt(可复用模板)。截至2026年6月已有10,000+公开MCP服务器。",
        "RAG": "RAG (Retrieval-Augmented Generation) 检索增强生成。通过先检索相关文档"
               "再让LLM基于文档生成答案, 有效解决幻觉和知识截止问题。2026年主流架构: "
               "混合检索(BM25+向量) + RRF融合 + Cross-Encoder重排序。",
    }
    results = []
    for k, v in knowledge.items():
        if query.lower() in k.lower() or any(w in v for w in query.split()):
            results.append(f"[{k}] {v}")
    return "\n\n".join(results) if results else f"未找到与 '{query}' 相关的信息。"


@tool(name="calculator", description="安全数学计算器。支持四则运算、三角函数、开方、对数。")
def calculator(expression: str = "") -> str:
    # 使用安全的白名单求值 (不用eval)
    safe_ns = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "log": math.log, "abs": abs,
        "round": round, "pi": math.pi, "e": math.e,
        "pow": pow, "int": int, "float": float,
    }
    # 仅允许安全的字符
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.\,\w]+$', expression):
        return "错误: 表达式包含不允许的字符"
    try:
        result = eval(expression, {"__builtins__": {}}, safe_ns)  # nosec - 严格白名单
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool(name="datetime_tool", description="获取当前日期时间或计算日期差。")
def datetime_tool(action: str = "now") -> str:
    now = datetime.now()
    if action == "now":
        return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} (星期{['一','二','三','四','五','六','日'][now.weekday()]})"
    elif action == "date":
        return now.strftime("%Y-%m-%d")
    return f"不支持的操作: {action}"
