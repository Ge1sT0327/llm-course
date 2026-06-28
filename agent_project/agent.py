"""
智能Agent核心 - ReAct推理循环 + Plan-and-Solve规划
智能知识库Agent的大脑
"""
import json
import re
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class AgentStep:
    """Agent单步执行记录"""
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""
    step_num: int = 0


class ReActAgent:
    """ReAct (Reasoning + Acting) Agent - 在Thought-Action-Observation循环中推理"""

    REACT_PROMPT = """你是一个智能Agent, 在Thought-Action-Observation循环中解决问题。

可用工具:
{tools_description}

回答格式:
Thought: 思考下一步应该做什么
Action: 工具名称
Action Input: 工具参数(JSON格式)
Observation: 工具返回的结果
... (可重复)

当你获得足够信息后:
Thought: 我已获得足够信息
Final Answer: 最终回答

用户问题: {question}

{history}
"""

    def __init__(self, llm_client, tool_registry, max_steps: int = 10, verbose: bool = True):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_steps = max_steps
        self.verbose = verbose
        self.trace: list[AgentStep] = []

    def run(self, question: str) -> str:
        """执行ReAct循环, 返回最终答案"""
        self.trace = []
        history = ""

        for step_num in range(1, self.max_steps + 1):
            tools_desc = self.tools.get_description()
            prompt = self.REACT_PROMPT.format(
                tools_description=tools_desc,
                question=question,
                history=history,
            )

            response = self.llm.chat(prompt, temperature=0.3, max_tokens=500)
            text = response["content"]
            step = self._parse_step(text, step_num)
            self.trace.append(step)

            if self.verbose:
                print(f"  [Step {step_num}] Thought: {step.thought[:80]}...")

            # 检查是否到达最终答案
            if "Final Answer:" in text:
                final = text.split("Final Answer:")[-1].strip()
                return final

            # 执行工具调用
            if step.action and step.action in self.tools.list_tools():
                try:
                    args = json.loads(step.action_input) if step.action_input else {}
                    result = self.tools.execute(step.action, args)
                    step.observation = str(result)[:500]
                    history += f"\nStep {step_num}: Action={step.action}, Result={step.observation[:100]}\n"
                except Exception as e:
                    step.observation = f"Error: {e}"

        return "Agent达到最大步数限制, 以下是收集到的信息:\n" + "\n".join(
            f"Step {s.step_num}: {s.observation[:100]}" for s in self.trace if s.observation
        )

    def _parse_step(self, text: str, step_num: int) -> AgentStep:
        """从LLM输出中解析Thought/Action/Observation"""
        step = AgentStep(step_num=step_num)
        patterns = {
            "thought": r"Thought:\s*(.+?)(?=\n(?:Action|Final|Observation)|\Z)",
            "action": r"Action:\s*(.+?)(?=\n|$)",
            "action_input": r"Action Input:\s*(.+?)(?=\n(?:Observation|Thought)|\Z)",
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                setattr(step, field, match.group(1).strip())
        return step


class PlanSolveAgent:
    """Plan-and-Solve Agent - 先规划再执行"""

    PLAN_PROMPT = """你是一个任务规划Agent。将用户的复杂任务分解为可执行步骤列表。

用户任务: {question}

请输出:
Plan:
1. [步骤1描述]
2. [步骤2描述]
...
N. [步骤N描述]

每个步骤应具体可执行, 步骤间逻辑连贯。"""

    SOLVE_PROMPT = """执行以下任务步骤。使用可用工具获取信息。

{plan}

当前步骤: Step {current_step}
可用工具: {tools}

请执行当前步骤, 收集必要信息。输出格式:
Step Result: 该步骤的执行结果"""

    def __init__(self, llm_client, tool_registry, verbose: bool = True):
        self.llm = llm_client
        self.tools = tool_registry
        self.verbose = verbose

    def run(self, question: str) -> dict:
        """执行Plan-and-Solve, 返回计划和各步骤结果"""
        # Phase 1: 规划
        plan_prompt = self.PLAN_PROMPT.format(question=question)
        response = self.llm.chat(plan_prompt, temperature=0.3, max_tokens=500)
        plan_text = response["content"]
        steps = [s.strip() for s in plan_text.split("\n") if s.strip() and s[0].isdigit()]

        if self.verbose:
            print(f"  规划了 {len(steps)} 个步骤")
            for s in steps[:5]:
                print(f"    {s[:80]}")

        # Phase 2: 逐步执行
        results = []
        for i, step in enumerate(steps, 1):
            solve_prompt = self.SOLVE_PROMPT.format(
                plan=plan_text,
                current_step=i,
                tools=self.tools.get_description(),
            )
            response = self.llm.chat(solve_prompt, temperature=0.3, max_tokens=400)
            results.append({"step": i, "plan": step[:100], "result": response["content"][:300]})
            if self.verbose:
                print(f"  [Step {i}] done ({len(response['content'])} chars)")

        return {"plan": plan_text, "steps": steps, "results": results}
