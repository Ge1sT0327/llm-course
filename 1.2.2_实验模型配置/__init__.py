"""
课程全局配置模块 — 一次配置, 所有实验公用

使用方法:
    from config import get_client, verify_config, list_providers

    # 验证配置
    verify_config()

    # 获取客户端
    client = get_client("deepseek")
    response = client.chat("你好")

    # 列出可用提供商
    print(list_providers())
"""
from .client import get_client, LLMClient
from .verify import verify_config, diagnose
from .providers import PROVIDERS, list_providers, get_provider_info
