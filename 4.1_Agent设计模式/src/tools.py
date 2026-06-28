"""
安全工具集 - 实现可被Agent调用的工具函数
所有工具使用 ast.literal_eval 或专用解析器, 杜绝 eval() 安全漏洞
"""
import ast
import math
import operator
from datetime import datetime
from typing import Any


# ---------- 安全的表达式求值器 ----------

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "abs": abs, "round": round,
    "pi": math.pi, "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> Any:
    """递归求值AST节点, 仅允许安全的数学运算"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"不支持的运算符: {op_type}")
        return _SAFE_OPS[op_type](
            _safe_eval_node(node.left),
            _safe_eval_node(node.right),
        )
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_safe_eval_node(node.operand)
        raise ValueError("不支持的一元运算符")
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in _SAFE_FUNCS:
            raise ValueError(f"不支持的函数: {func_name}")
        args = [_safe_eval_node(a) for a in node.args]
        return _SAFE_FUNCS[func_name](*args)
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCS:
            return _SAFE_FUNCS[node.id]
        raise ValueError(f"未定义的变量: {node.id}")
    raise ValueError(f"不支持的表达式类型: {type(node)}")


def safe_calculate(expression: str) -> float:
    """
    安全计算数学表达式 (不使用eval)

    Args:
        expression: 数学表达式字符串, 如 "sqrt(144) + 3 * 5"

    Returns:
        计算结果

    Raises:
        ValueError: 表达式包含不允许的操作
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return _safe_eval_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        raise ValueError(f"表达式错误: {e}")


# ---------- 知识库搜索 ----------

_KNOWLEDGE_BASE = {
    "Agent设计模式": "Agent常用设计模式包括: ReAct(推理-行动循环)、Plan-and-Solve(先规划后执行)、"
                     "Multi-Agent(多Agent协作)。ReAct由Yao等人(2023)提出, 核心是在Thought-Action-"
                     "Observation循环中交替推理与行动。",
    "ReAct模式": "ReAct = Reasoning + Acting。流程: Thought(思考下一步) -> Action(执行工具) -> "
                "Observation(观察结果) -> 循环直到得到最终答案。优势: 可解释性强, 支持多步推理。",
    "Function Calling": "Function Calling是LLM调用外部工具的标准化机制。通过JSON Schema定义工具接口, "
                       "模型自动判断是否需要调用工具并生成结构化参数。2026年MCP协议进一步标准化了工具接入。",
    "MCP协议": "Model Context Protocol (MCP) 是Anthropic于2024年底提出的AI Agent标准化协议, "
              "2025年12月捐赠给Linux Foundation。截至2026年6月已有10,000+公开MCP服务器。"
              "MCP定义三种原语: Tool(工具)、Resource(资源)、Prompt(提示模板)。",
    "DeepSeek V4": "DeepSeek V4于2026年4月发布, 采用1.6T参数MoE架构(37B活跃参数)。"
                   "支持1M上下文, API成本约0.87元/百万输出tokens。在代码生成和数学推理上达到SOTA。",
    "Qwen3.7": "Qwen3.7-Max于2026年5月发布, 235B-A22B MoE架构, 支持1M上下文。"
               "原生多模态(视觉+音频+工具调用), 中文能力在国产模型中排名第一。",
}


def search_knowledge(query: str) -> str:
    """
    在本地知识库中搜索相关信息

    Args:
        query: 搜索关键词

    Returns:
        匹配的知识条目, 或提示未找到
    """
    results = []
    query_lower = query.lower()
    for key, value in _KNOWLEDGE_BASE.items():
        if query_lower in key.lower() or any(
            word in value for word in query_lower.split()
        ):
            results.append(f"[{key}] {value}")

    if results:
        return "\n\n".join(results[:3])  # 最多返回3条
    return f"未找到与 '{query}' 相关的知识条目。请尝试其他关键词。"


def get_datetime(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    获取当前日期时间

    Args:
        format_str: 日期时间格式

    Returns:
        格式化后的当前时间字符串
    """
    return datetime.now().strftime(format_str)


# ---------- 工具注册表 ----------

class ToolRegistry:
    """工具注册表 - 管理Agent可调用的工具集合"""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, func: callable, description: str, parameters: dict):
        """
        注册一个工具

        Args:
            name: 工具名称
            func: 工具函数
            description: 工具功能描述
            parameters: JSON Schema格式的参数定义
        """
        self._tools[name] = {
            "func": func,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
        }

    def get_schemas(self) -> list[dict]:
        """获取所有工具的OpenAI Function Calling格式Schema列表"""
        return [t["schema"] for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """
        执行指定工具

        Args:
            name: 工具名称
            arguments: 工具参数字典

        Returns:
            工具执行结果字符串

        Raises:
            ValueError: 工具不存在
        """
        if name not in self._tools:
            raise ValueError(f"未知工具: {name}. 可用工具: {list(self._tools.keys())}")

        func = self._tools[name]["func"]
        try:
            result = func(**arguments)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {e}"


# ---------- 创建默认工具注册表 ----------

def create_default_registry() -> ToolRegistry:
    """创建包含默认工具的注册表"""
    registry = ToolRegistry()

    registry.register(
        name="search_knowledge",
        func=search_knowledge,
        description="在本地知识库中搜索AI/LLM/Agent相关知识",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题",
                },
            },
            "required": ["query"],
        },
    )

    registry.register(
        name="calculator",
        func=safe_calculate,
        description="安全计算数学表达式, 支持四则运算/三角函数/开方/对数",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式, 如 'sqrt(144) + 3 * 5'",
                },
            },
            "required": ["expression"],
        },
    )

    registry.register(
        name="get_datetime",
        func=get_datetime,
        description="获取当前日期和时间",
        parameters={
            "type": "object",
            "properties": {
                "format_str": {
                    "type": "string",
                    "description": "日期格式, 默认为 '%Y-%m-%d %H:%M:%S'",
                },
            },
        },
    )

    return registry
