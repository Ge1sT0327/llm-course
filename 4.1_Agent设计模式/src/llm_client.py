"""
LLM客户端模块 - 统一封装国产大模型API调用
支持 DeepSeek V4, Qwen3.7, GLM-5.2 等OpenAI兼容接口
"""
import os
import json
from typing import Optional
from openai import OpenAI


class LLMClient:
    """统一的LLM API客户端, 支持多个国产模型提供商"""

    # 2026年6月最新模型配置
    PROVIDERS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",  # 后端实际为 DeepSeek V4
            "description": "DeepSeek V4 - 685B MoE, 极致性价比",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.7-max",  # Qwen3.7-Max - 2026年5月发布
            "description": "Qwen3.7-Max - 1M上下文, 中文最强, 多模态",
        },
        "glm": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4/",
            "model": "glm-5.2",  # GLM-5.2 - 2026年6月发布
            "description": "GLM-5.2 - Agent能力T0级",
        },
    }

    def __init__(self, provider: str = "deepseek", api_key: Optional[str] = None):
        """
        初始化LLM客户端

        Args:
            provider: 模型提供商 ('deepseek', 'qwen', 'glm')
            api_key: API密钥, 不提供则从环境变量读取
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"不支持的提供商: {provider}. 可选: {list(self.PROVIDERS.keys())}")

        config = self.PROVIDERS[provider]

        # 自动从环境变量获取API Key
        env_key_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "glm": "ZHIPU_API_KEY",
        }
        api_key = api_key or os.getenv(env_key_map[provider])
        if not api_key:
            raise ValueError(
                f"未找到{provider}的API Key. "
                f"请设置环境变量 {env_key_map[provider]} 或传入 api_key 参数"
            )

        self.client = OpenAI(api_key=api_key, base_url=config["base_url"])
        self.model = config["model"]
        self.provider = provider

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """
        调用LLM进行对话

        Args:
            messages: OpenAI格式的消息列表
            temperature: 温度参数 (0=确定性, 1=创造性)
            max_tokens: 最大输出token数
            tools: 可选的工具定义列表 (Function Calling)

        Returns:
            dict: {
                "content": 回复文本,
                "tool_calls": 工具调用列表 (如有),
                "usage": {"input": int, "output": int}
            }
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        result = {
            "content": choice.message.content or "",
            "tool_calls": [],
            "usage": {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            },
        }

        # 解析工具调用
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        return result
