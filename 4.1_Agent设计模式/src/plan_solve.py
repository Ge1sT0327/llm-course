"""
Plan-and-Solve Agent实现
先规划(Plan)再执行(Solve), 适用于复杂多步骤问题
"""
import json
from .llm_client import LLMClient
from .tools import ToolRegistry


PLAN_SYSTEM_PROMPT = """你是一个使用Plan-and-Solve(先规划后执行)模式的智能Agent。

## 工作流程
1. PLAN阶段: 分析用户问题, 将其分解为可执行的步骤序列
2. SOLVE阶段: 按顺序执行每个步骤, 调用工具获取所需信息

## PLAN阶段输出格式
```json
{
  "plan": [
    {"step": 1, "description": "步骤描述", "tool": "工具名或null", "tool_input": "参数或null"},
    {"step": 2, "description": "步骤描述", "tool": "工具名或null", "tool_input": "参数或null"}
  ],
  "final_answer": null
}
```

## SOLVE阶段输出格式
每完成一个步骤后:
```json
{
  "current_step": 2,
  "step_result": "该步骤的执行结果总结",
  "final_answer": null
}
```

全部步骤完成后:
```json
{
  "current_step": null,
  "step_result": null,
  "final_answer": "基于所有步骤的完整答案"
}
```"""


class PlanSolveAgent:
    """
    Plan-and-Solve (先规划后执行) Agent

    适用场景:
    - 需要多步骤才能完成的复杂任务
    - 问题有清晰的结构, 可以分解为独立子任务
    - 需要先收集信息再综合分析的场景

    优势:
    - 结构化: 问题被分解为明确的步骤序列
    - 可控: 可以在执行前审查和调整计划
    - 可追溯: 每步的执行结果都被记录
    """

    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry):
        """
        初始化Plan-and-Solve Agent

        Args:
            llm_client: LLM客户端实例
            tool_registry: 工具注册表
        """
        self.llm = llm_client
        self.tools = tool_registry
        self.plan: list[dict] = []
        self.results: list[dict] = []

    def run(self, query: str, verbose: bool = True) -> str:
        """
        执行Plan-and-Solve流程

        Args:
            query: 用户问题
            verbose: 是否打印详细过程

        Returns:
            最终答案
        """
        self.plan = []
        self.results = []

        # ---- Phase 1: PLAN ----
        if verbose:
            print("\n" + "=" * 50)
            print("  [阶段1] PLAN - 制定执行计划")
            print("=" * 50)

        plan_messages = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": f"请为以下问题制定执行计划:\n{query}"},
        ]

        plan_response = self.llm.chat(plan_messages, temperature=0.3)

        try:
            content = plan_response["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]
            plan_data = json.loads(content)
            self.plan = plan_data.get("plan", [])
        except json.JSONDecodeError:
            # 无法解析JSON, 使用默认的单步计划
            self.plan = [{
                "step": 1, "description": "直接回答用户问题",
                "tool": None, "tool_input": None,
            }]

        if verbose:
            for p in self.plan:
                tool_info = f" -> {p['tool']}({p['tool_input']})" if p.get('tool') else ""
                print(f"  步骤{p['step']}: {p['description']}{tool_info}")

        # ---- Phase 2: SOLVE ----
        if verbose:
            print("\n" + "=" * 50)
            print("  [阶段2] SOLVE - 逐步执行计划")
            print("=" * 50)

        context = []  # 积累的上下文信息

        for step in self.plan:
            step_num = step["step"]
            description = step["description"]
            tool_name = step.get("tool")
            tool_input = step.get("tool_input", "")

            if verbose:
                print(f"\n  >>> 执行步骤{step_num}: {description}")

            step_result = ""

            # 如果需要调用工具
            if tool_name and tool_name != "null":
                try:
                    # 解析工具参数
                    if isinstance(tool_input, str):
                        params = {"expression": tool_input} if tool_name == "calculator" else {"query": tool_input}
                    else:
                        params = tool_input or {}

                    tool_output = self.tools.execute(tool_name, params)
                    step_result = f"工具 {tool_name} 返回: {tool_output}"
                    if verbose:
                        print(f"    工具结果: {tool_output[:200]}")
                except Exception as e:
                    step_result = f"工具调用失败: {e}"
                    if verbose:
                        print(f"    错误: {e}")
            else:
                # 不需要工具, 直接让LLM处理该步骤
                solve_messages = [
                    {"role": "system", "content": f"执行计划步骤{step_num}: {description}"},
                    {"role": "user", "content": f"已有上下文:\n{chr(10).join(context)}\n\n请完成此步骤并给出结果。"},
                ]
                solve_response = self.llm.chat(solve_messages, temperature=0.3)
                step_result = solve_response["content"]

            context.append(f"[步骤{step_num}] {description}\n结果: {step_result}")
            self.results.append({
                "step": step_num,
                "description": description,
                "result": step_result,
            })

        # ---- Phase 3: SYNTHESIZE ----
        if verbose:
            print("\n" + "=" * 50)
            print("  [阶段3] SYNTHESIZE - 综合所有结果")
            print("=" * 50)

        synthesize_prompt = f"""基于以下步骤执行结果, 给出完整答案。

原始问题: {query}

执行结果:
{chr(10).join(context)}

请综合所有信息, 用中文给出完整、准确的答案。"""

        final_response = self.llm.chat([
            {"role": "user", "content": synthesize_prompt},
        ], temperature=0.3)

        final_answer = final_response["content"]
        if verbose:
            print(f"  [最终答案] {final_answer[:300]}...")

        return final_answer
