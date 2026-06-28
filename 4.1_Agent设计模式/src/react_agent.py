"""
ReAct Agent实现 - Reasoning + Acting 循环
基于 Yao et al. (2023) 的 ReAct 论文范式
"""
import json
from .llm_client import LLMClient
from .tools import ToolRegistry


# ReAct Agent的System Prompt模板
REACT_SYSTEM_PROMPT = """你是一个使用ReAct(推理-行动)模式的智能Agent。

## 工作流程
对于用户的每个问题, 你需要在 Thought(思考) -> Action(行动) -> Observation(观察) 循环中反复迭代。

## 输出格式
每次响应必须严格遵循以下JSON格式:
```json
{
  "thought": "你当前的推理过程 - 分析已有信息, 判断是否需要更多信息, 规划下一步",
  "action": "要调用的工具名称 (如果不需要工具则设为 null)",
  "action_input": "传递给工具的参数 (如果不需要工具则设为 null)",
  "final_answer": "如果已有足够信息回答问题, 在此给出最终答案 (否则设为 null)"
}
```

## 规则
1. 每次只输出一个JSON对象, 不要输出其他内容
2. 如果发现信息不足, 使用 action 调用工具获取更多信息
3. 当信息足够时, 设置 final_answer 给出完整答案
4. 最多进行5轮推理-行动循环"""


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent

    工作流程:
    1. 接收用户问题
    2. Thought: 分析当前状态, 推理下一步
    3. Action: 选择合适的工具并调用
    4. Observation: 观察工具返回结果
    5. 重复步骤2-4直到获得足够信息
    6. 输出最终答案

    核心优势:
    - 可解释性强: 每一步推理过程可见
    - 支持多步推理: 复杂问题可分解为多个子问题
    - 工具灵活调用: 根据推理结果动态选择工具
    """

    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry,
                 max_iterations: int = 5):
        """
        初始化ReAct Agent

        Args:
            llm_client: LLM客户端实例
            tool_registry: 工具注册表
            max_iterations: 最大推理-行动循环次数 (防止无限循环)
        """
        self.llm = llm_client
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.history: list[dict] = []  # 记录完整的推理-行动历史

    def run(self, query: str, verbose: bool = True) -> str:
        """
        执行ReAct循环, 解决用户问题

        Args:
            query: 用户问题
            verbose: 是否打印详细的推理过程

        Returns:
            最终答案字符串

        Raises:
            RuntimeError: 超过最大循环次数仍未得出答案
        """
        self.history = []
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n{'='*50}")
                print(f"  ReAct 第 {iteration} 轮")
                print(f"{'='*50}")

            # 调用LLM获取下一步推理
            response = self.llm.chat(
                messages=messages,
                temperature=0.3,  # 低温度保证推理的确定性
                tools=self.tools.get_schemas(),
            )

            # 解析Agent的JSON响应
            try:
                content = response["content"].strip()
                # 清理可能的markdown代码块标记
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                step = json.loads(content.strip())
            except json.JSONDecodeError:
                # JSON解析失败, 尝试从文本中提取
                step = {"thought": response["content"], "action": None,
                        "action_input": None, "final_answer": None}

            thought = step.get("thought", "")
            action = step.get("action")
            action_input = step.get("action_input", {})
            final_answer = step.get("final_answer")

            if verbose:
                print(f"  [思考] {thought}")

            # 记录历史
            self.history.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "action_input": action_input,
            })

            # 如果Agent给出了最终答案, 结束循环
            if final_answer:
                if verbose:
                    print(f"  [答案] {final_answer}")
                return final_answer

            # 如果Agent选择调用工具
            if action:
                if verbose:
                    print(f"  [行动] 调用工具: {action}({action_input})")

                try:
                    observation = self.tools.execute(action, action_input)
                except ValueError as e:
                    observation = f"工具调用失败: {e}"

                if verbose:
                    print(f"  [观察] {observation[:200]}")

                self.history[-1]["observation"] = observation

                # 将工具调用和结果添加到消息历史
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(step, ensure_ascii=False),
                })
                messages.append({
                    "role": "user",
                    "content": f"工具 {action} 的执行结果:\n{observation}\n\n请继续推理或给出最终答案。",
                })
            else:
                # 没有工具调用也没有最终答案, 提示继续
                messages.append({
                    "role": "assistant",
                    "content": response["content"],
                })
                messages.append({
                    "role": "user",
                    "content": "请继续推理。如果需要调用工具, 请指定action。如果已有答案, 请给出final_answer。",
                })

        # 超过最大迭代次数, 强制要求LLM给出答案
        messages.append({
            "role": "user",
            "content": "已达到最大推理步数。请基于当前所有信息, 给出你能提供的最佳答案。",
        })
        final = self.llm.chat(messages=messages, temperature=0.3)
        return final["content"] or "无法在限定步数内完成推理。"

    def get_summary(self) -> str:
        """获取推理过程摘要"""
        lines = ["ReAct 推理过程摘要:"]
        for h in self.history:
            lines.append(f"  第{h['iteration']}轮: {h['thought'][:80]}...")
            if h.get('action'):
                lines.append(f"    工具: {h['action']}")
        return "\n".join(lines)
