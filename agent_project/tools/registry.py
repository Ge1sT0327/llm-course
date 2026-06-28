"""
工具插件系统 - Agent的"手和脚"
支持动态注册、安全执行、结果格式化
"""
import json
import math
import ast
from typing import Any, Callable
from datetime import datetime


class SafeCalculator:
    """安全计算器 - AST白名单求值, 杜绝eval()"""

    SAFE_OPS = {
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub,
    }
    SAFE_FUNCS = {
        "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt,
        "log": math.log, "abs": abs, "round": round,
        "max": max, "min": min, "sum": sum,
        "pi": math.pi, "e": math.e,
    }

    @classmethod
    def calculate(cls, expression: str) -> str:
        """安全计算数学表达式"""
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = cls._eval(tree.body)
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    @classmethod
    def _eval(cls, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op_map = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                      ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
                      ast.Pow: lambda a, b: a ** b}
            if type(node.op) in op_map:
                return op_map[type(node.op)](cls._eval(node.left), cls._eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -cls._eval(node.operand)
        if isinstance(node, ast.Name) and node.id in cls.SAFE_FUNCS:
            return cls.SAFE_FUNCS[node.id]
        if isinstance(node, ast.Call) and node.func.id in cls.SAFE_FUNCS:
            args = [cls._eval(a) for a in node.args]
            return cls.SAFE_FUNCS[node.func.id](*args)
        raise ValueError(f"不支持: {type(node)}")


class ToolRegistry:
    """工具注册表 - 管理Agent可用的所有工具"""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, func: Callable, description: str,
                 parameters: dict | None = None) -> None:
        """注册工具"""
        self._tools[name] = {
            "name": name,
            "func": func,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}, "required": []},
        }

    def execute(self, name: str, arguments: dict) -> str:
        """执行工具"""
        if name not in self._tools:
            return f"错误: 未知工具 '{name}'. 可用: {list(self._tools.keys())}"
        try:
            result = self._tools[name]["func"](**arguments)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {e}"

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_description(self) -> str:
        """生成工具描述文本 (用于LLM prompt)"""
        lines = []
        for name, info in self._tools.items():
            params = json.dumps(info.get("parameters", {}).get("properties", {}), ensure_ascii=False)
            lines.append(f"- {name}: {info['description']} 参数: {params}")
        return "\n".join(lines) if lines else "无可用工具"

    def get_openai_schemas(self) -> list[dict]:
        """生成OpenAI Function Calling格式"""
        schemas = []
        for name, info in self._tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"],
                },
            })
        return schemas


# ===== 内置工具 =====

def builtin_search(query: str) -> str:
    """搜索本地知识库"""
    KB = {
        "DeepSeek V4": "2026年4月发布, 1.6T MoE, 全栈昇腾910C训练, SWE-bench 80.6%",
        "Qwen3.7": "2026年5月发布, 235B MoE, 1M上下文, 多模态, AA Index #1",
        "GLM-5.2": "2026年6月发布, Agent T0级, 自研GLM架构",
        "MCP": "AI Agent标准协议, Linux Foundation托管, 10000+服务器",
        "RAG": "检索增强生成, 减少幻觉, 2026年标准: 混合检索+RRF+重排序",
    }
    for k, v in KB.items():
        if query.lower() in k.lower():
            return f"[{k}] {v}"
    return f"未找到 '{query}' 相关信息"


def builtin_datetime(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前时间"""
    return datetime.now().strftime(fmt)


def create_default_registry() -> ToolRegistry:
    """创建预配置的工具注册表"""
    r = ToolRegistry()
    r.register("search", builtin_search, "搜索知识库",
               {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
    r.register("calculator", SafeCalculator.calculate, "安全计算器",
               {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]})
    r.register("datetime", builtin_datetime, "获取时间",
               {"type": "object", "properties": {"fmt": {"type": "string"}}})
    return r
