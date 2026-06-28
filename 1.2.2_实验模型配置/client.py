"""
统一LLM客户端工厂 — 所有实验共享

使用示例:
    from config import get_client

    # 方式1: 按提供商ID获取
    client = get_client("deepseek")
    result = client.chat("你好")

    # 方式2: 自动选择第一个可用的提供商
    client = get_client()  # 自动检测
"""
import os
import json
from typing import Optional
from openai import OpenAI
from .providers import PROVIDERS, get_provider_info


class LLMClient:
    """
    统一LLM客户端 — 封装OpenAI兼容接口

    支持所有已在 providers.py 中注册的模型提供商
    """

    def __init__(self, provider: str = "deepseek", api_key: Optional[str] = None,
                 model: Optional[str] = None):
        """
        Args:
            provider: 提供商ID ('deepseek', 'qwen', 'glm', 'kimi', 'doubao')
            api_key: API密钥 (默认从环境变量读取)
            model: 覆盖默认模型ID (可选)
        """
        info = get_provider_info(provider)
        api_key = api_key or os.getenv(info["env_key"])
        if not api_key:
            raise ValueError(
                f"未找到 {info['name']} 的API Key.\n"
                f"请在 .env 文件中设置: {info['env_key']}=sk-xxxxx\n"
                f"或设置环境变量: export {info['env_key']}=sk-xxxxx"
            )

        self.client = OpenAI(api_key=api_key, base_url=info["base_url"])
        self.model = model or info["model"]
        self.provider = provider
        self._info = info

    @property
    def name(self) -> str:
        return self._info["name"]

    def chat(
        self,
        messages: list[dict] | str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[list[dict]] = None,
        stream: bool = False,
    ) -> dict | object:
        """
        调用LLM对话

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}] 或纯文本字符串
            system: 系统提示词 (仅当 messages 为字符串时有效)
            temperature: 温度 (0=确定, 1=创造)
            max_tokens: 最大输出token数
            tools: Function Calling工具定义列表 (可选)
            stream: 是否流式输出

        Returns:
            dict: {
                "content": str,
                "tool_calls": list,
                "usage": {"input": int, "output": int},
                "model": str,
                "provider": str,
            }
            或 StreamingResponse 对象 (stream=True时)
        """
        # 允许传入纯文本字符串
        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": system} if system else None,
                {"role": "user", "content": messages},
            ]
            messages = [m for m in messages if m is not None]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if stream:
            kwargs["stream"] = True
            return self.client.chat.completions.create(**kwargs)

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        result = {
            "content": choice.message.content or "",
            "tool_calls": [],
            "usage": {
                "input": response.usage.prompt_tokens if response.usage else 0,
                "output": response.usage.completion_tokens if response.usage else 0,
            },
            "model": self.model,
            "provider": self.provider,
        }

        # 解析 Function Calling 工具调用
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                result["tool_calls"].append({
                    "id": getattr(tc, 'id', ''),
                    "name": tc.function.name,
                    "arguments": args,
                })

        return result

    def chat_stream(self, messages, system="", temperature=0.7):
        """
        流式对话 — 逐token返回 (生成器)
        """
        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": system} if system else None,
                {"role": "user", "content": messages},
            ]
            messages = [m for m in messages if m is not None]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def get_client(provider: Optional[str] = None, **kwargs) -> LLMClient:
    """
    工厂函数 — 获取LLM客户端实例

    如果未指定 provider, 自动检测第一个已配置API Key的提供商

    Args:
        provider: 提供商ID (可选, 默认自动检测)
        **kwargs: 传递给 LLMClient 的额外参数

    Returns:
        LLMClient 实例

    Raises:
        RuntimeError: 没有任何提供商配置了API Key
    """
    if provider:
        return LLMClient(provider, **kwargs)

    # 自动检测: 按优先级选择第一个可用的
    priority = ["deepseek", "qwen-plus", "qwen", "glm", "kimi", "doubao"]
    import os as _os
    for p in priority:
        info = get_provider_info(p)
        if _os.getenv(info["env_key"]):
            return LLMClient(p, **kwargs)

    raise RuntimeError(
        "未检测到任何已配置的API Key.\n"
        "请在项目根目录创建 .env 文件:\n"
        "  DEEPSEEK_API_KEY=sk-xxxxx\n"
        "  DASHSCOPE_API_KEY=sk-xxxxx\n"
        "或设置相应的环境变量."
    )
