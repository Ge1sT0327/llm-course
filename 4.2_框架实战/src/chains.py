"""
Chain管道模式实现 - 模拟LangChain Runnable接口的 | 运算符
"""
import time
import json
from typing import Any, Callable, Optional
from .monitor import ChainMonitor


class Chain:
    """Chain基类 - 所有可链接组件的抽象"""

    def __init__(self, name: str = "Chain"):
        self.name = name

    def invoke(self, input_data: Any) -> Any:
        """执行Chain并返回结果 (子类必须实现)"""
        raise NotImplementedError

    def __or__(self, other: "Chain") -> "Pipeline":
        """重载 | 运算符实现链式组合: step1 | step2 | step3"""
        if isinstance(self, Pipeline):
            self.steps.append(other)
            return self
        pipeline = Pipeline(f"{self.name}|{other.name}")
        pipeline.steps = [self, other]
        return pipeline


class Pipeline(Chain):
    """管道 - 按顺序执行多个Chain步骤"""

    def __init__(self, name: str = "Pipeline"):
        super().__init__(name)
        self.steps: list[Chain] = []

    def invoke(self, input_data: Any) -> Any:
        """按顺序执行每个步骤, 前一步的输出作为下一步的输入"""
        data = input_data
        for step in self.steps:
            data = step.invoke(data)
        return data


class PromptTemplate(Chain):
    """提示词模板 - 用输入变量填充模板"""

    def __init__(self, template: str, name: str = "PromptTemplate"):
        """
        Args:
            template: 模板字符串, 使用 {variable_name} 语法
            name: 组件名称
        """
        super().__init__(name)
        self.template = template

    def invoke(self, input_data: Any) -> str:
        """
        用输入数据填充模板

        Args:
            input_data: 字符串或包含模板变量的字典

        Returns:
            填充后的提示词字符串
        """
        if isinstance(input_data, dict):
            return self.template.format(**input_data)
        return self.template.format(input=input_data)


class LLMStep(Chain):
    """
    LLM调用步骤 - 包装API调用为Chain组件
    使用OpenAI兼容接口, 支持DeepSeek/Qwen/GLM
    """

    def __init__(
        self,
        system_prompt: str = "你是一个有帮助的AI助手。",
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        name: str = "LLMStep",
    ):
        super().__init__(name)
        self.system_prompt = system_prompt
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, user_prompt: str) -> dict:
        """
        调用LLM并返回结果

        Args:
            user_prompt: 用户提示词

        Returns:
            {"content": str, "usage": {"input": int, "output": int}}
        """
        import os
        from openai import OpenAI

        api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return {"content": f"[模拟LLM响应] 收到提示词 ({len(user_prompt)} 字符)", "usage": {"input": 0, "output": 0}}

        client = OpenAI(api_key=api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            },
        }


class Parser(Chain):
    """解析器 - 对LLM输出进行后处理"""

    def __init__(self, parser_fn: Callable[[Any], Any], name: str = "Parser"):
        """
        Args:
            parser_fn: 解析函数, 接收LLM输出, 返回解析后的数据
            name: 组件名称
        """
        super().__init__(name)
        self.parser_fn = parser_fn

    def invoke(self, llm_output: dict) -> Any:
        """解析LLM输出"""
        return self.parser_fn(llm_output)


# ---------- 预置解析器 ----------

def json_parser(llm_output: dict) -> dict:
    """从LLM输出中提取JSON"""
    text = llm_output.get("content", "")
    # 尝试提取JSON块
    import re
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"raw_text": text}
