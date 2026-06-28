"""
LangGraph风格状态机 - 模拟LangGraph的StateGraph概念
用于构建有状态的Agent工作流
"""
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentState:
    """Agent状态 - 在图中各节点间传递"""
    messages: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    current_step: int = 0
    max_steps: int = 10
    next_action: str = "continue"  # continue | finish | error
    output: str = ""


class GraphNode:
    """图中的节点 - 处理状态并返回更新后的状态"""

    def __init__(self, name: str, handler: Callable[[AgentState], AgentState]):
        self.name = name
        self.handler = handler

    def execute(self, state: AgentState) -> AgentState:
        """执行节点逻辑"""
        state.current_step += 1
        return self.handler(state)


class GraphEdge:
    """图中的边 - 根据状态决定下一个节点"""

    def __init__(self, from_node: str, to_node: str,
                 condition: Callable[[AgentState], bool] | None = None):
        self.from_node = from_node
        self.to_node = to_node
        self.condition = condition or (lambda s: True)


class StateGraph:
    """
    LangGraph风格的StateGraph实现

    使用方式:
        graph = StateGraph()
        graph.add_node("think", think_handler)
        graph.add_node("act", act_handler)
        graph.add_edge("think", "act", condition=lambda s: s.next_action == "continue")
        graph.add_edge("think", "__end__", condition=lambda s: s.next_action == "finish")
        graph.set_entry("think")
        result = graph.run(initial_state)

    核心概念:
    - Node: 处理步骤, 接收状态并返回修改后的状态
    - Edge: 状态转换, 可带条件判断
    - State: 贯穿所有节点的共享状态对象
    """

    def __init__(self, name: str = "StateGraph"):
        self.name = name
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self.entry_node: str | None = None

    def add_node(self, name: str, handler: Callable[[AgentState], AgentState]):
        """添加节点"""
        self.nodes[name] = GraphNode(name, handler)

    def add_edge(self, from_node: str, to_node: str,
                 condition: Callable[[AgentState], bool] | None = None):
        """添加边(可带条件)"""
        self.edges.append(GraphEdge(from_node, to_node, condition))

    def set_entry(self, node_name: str):
        """设置入口节点"""
        self.entry_node = node_name

    def _get_next(self, current: str, state: AgentState) -> str:
        """根据当前节点和状态决定下一个节点"""
        for edge in self.edges:
            if edge.from_node == current and edge.condition(state):
                return edge.to_node
        return "__end__"

    def run(self, initial_state: AgentState, verbose: bool = True) -> AgentState:
        """
        运行状态机

        Args:
            initial_state: 初始状态
            verbose: 是否打印执行过程

        Returns:
            最终状态
        """
        if not self.entry_node:
            raise ValueError("未设置入口节点, 请调用 set_entry()")

        state = initial_state
        current_node = self.entry_node

        while current_node != "__end__":
            if state.current_step >= state.max_steps:
                if verbose:
                    print(f"[Graph] 达到最大步数限制 ({state.max_steps})")
                break

            if current_node not in self.nodes:
                raise ValueError(f"节点不存在: {current_node}")

            node = self.nodes[current_node]
            if verbose:
                print(f"[Graph] 步骤{state.current_step + 1}: 执行节点 '{current_node}'")

            state = node.execute(state)
            next_node = self._get_next(current_node, state)
            current_node = next_node

        return state
