"""
Agent实现 - 基于LLM Function Calling的工具选择与执行
"""
import json
import os
import re
from openai import OpenAI
from .tools import get_tool_schemas, get_tool, list_tools


class SimpleAgent:
    """
    基于Function Calling的Agent

    工作流程:
    1. 接收用户查询
    2. LLM判断是否需要调用工具 → 选择合适的工具+参数
    3. 如果需要工具: 执行工具 → 将结果反馈给LLM → 生成最终回复
    4. 如果不需要: 直接回复
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
    ):
        """
        Args:
            model: 模型ID
            base_url: API端点
            api_key: API密钥 (默认从环境变量获取)
        """
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.has_api = api_key is not None
        if self.has_api:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_url = base_url

    def run(self, query: str, verbose: bool = True) -> str:
        """
        执行Agent查询

        Args:
            query: 用户查询
            verbose: 是否打印详细过程

        Returns:
            Agent的最终回复
        """
        # 步骤1: 让LLM选择工具
        tool_schemas = get_tool_schemas()
        tool_names = list_tools()

        if verbose:
            print(f"[Agent] 收到查询: {query}")
            print(f"[Agent] 可用工具 ({len(tool_names)}): {', '.join(tool_names)}")

        # 构建工具选择提示
        select_prompt = f"""根据用户问题选择合适的工具。可用工具: {json.dumps(tool_names, ensure_ascii=False)}

用户问题: {query}

返回JSON格式: {{"tool": "工具名", "params": {{"参数名": "参数值"}}}} 或 {{"tool": "none", "answer": "直接回答"}}
只返回JSON, 不要其他文字。"""

        if self.has_api:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是工具选择专家。只返回JSON。"},
                        {"role": "user", "content": select_prompt},
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
            except Exception as e:
                return f"API调用失败: {e}"
        else:
            # 无API时的降级处理
            content = self._local_tool_select(query, tool_names)

        # 解析工具选择
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                selection = json.loads(json_match.group())
            else:
                selection = {"tool": "none", "answer": content}

            tool_name = selection.get("tool", "none")
            params = selection.get("params", {})

            if verbose:
                print(f"[Agent] 决策: tool={tool_name}, params={params}")

            # 如果不需要工具, 直接返回
            if tool_name == "none":
                direct_answer = selection.get("answer", content)
                if verbose:
                    print(f"[Agent] 直接回答: {direct_answer[:200]}...")
                return direct_answer

            # 执行工具
            tool_entry = get_tool(tool_name)
            if not tool_entry:
                return f"未知工具: {tool_name}"

            tool_result = tool_entry["fn"](**params)
            if verbose:
                print(f"[Agent] 工具 {tool_name} 返回: {str(tool_result)[:200]}")

            # 综合工具结果生成最终回答
            if self.has_api:
                try:
                    final_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "根据工具返回的结果, 用中文给出清晰友好的回答。"},
                            {"role": "user", "content": f"用户问题: {query}\n工具'{tool_name}'返回: {tool_result}\n请给出回答。"},
                        ],
                        temperature=0.3,
                    )
                    return final_response.choices[0].message.content
                except Exception:
                    pass

            return f"根据{query}的查询结果:\n{tool_result}"

        except json.JSONDecodeError:
            return f"无法解析工具选择: {content[:200]}"

    def _local_tool_select(self, query: str, tool_names: list[str]) -> str:
        """无API时的本地工具选择 (基于关键词匹配的降级方案)"""
        query_lower = query.lower()
        if any(w in query_lower for w in ["搜索", "查找", "search", "什么是", "解释"]):
            return f'{{"tool": "search", "params": {{"query": "{query}"}}}}'
        if any(w in query_lower for w in ["计算", "算", "等于", "+", "-", "*", "/"]):
            return f'{{"tool": "calculator", "params": {{"expression": "{query}"}}}}'
        if any(w in query_lower for w in ["时间", "日期", "今天", "现在"]):
            return '{"tool": "datetime_tool", "params": {"action": "now"}}'
        return f'{{"tool": "none", "answer": "无API模式: 收到查询\\"{query}\\"。请配置API Key以获得完整Agent体验。"}}'
