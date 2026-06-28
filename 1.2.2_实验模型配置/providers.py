"""
模型提供商注册表 — 2026年6月最新配置

新增提供商: 在此文件添加一个条目即可全局使用
"""
import os
from pathlib import Path

# 自动定位项目根目录 (config模块所在目录的父目录)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2026年6月最新模型提供商配置
PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek V4",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "pricing": "输入 ¥1/1M tokens, 输出 ¥2/1M tokens",
        "strengths": "代码生成, 数学推理, 极致性价比",
        "docs": "https://platform.deepseek.com/api-docs",
    },
    "deepseek-r1": {
        "name": "DeepSeek-R1-0528",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-reasoner",
        "env_key": "DEEPSEEK_API_KEY",
        "pricing": "输入 ¥4/1M tokens, 输出 ¥16/1M tokens",
        "strengths": "深度推理, CoT思维链, 复杂问题求解",
        "docs": "https://platform.deepseek.com/api-docs",
    },
    "qwen": {
        "name": "Qwen3.7-Max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-max",
        "env_key": "DASHSCOPE_API_KEY",
        "pricing": "输入 ¥20/1M tokens, 输出 ¥60/1M tokens",
        "strengths": "中文最强, 多模态(视觉+音频), 1M上下文",
        "docs": "https://dashscope.aliyun.com",
    },
    "qwen-plus": {
        "name": "Qwen3.7-Plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-plus",
        "env_key": "DASHSCOPE_API_KEY",
        "pricing": "输入 ¥2/1M tokens, 输出 ¥6/1M tokens",
        "strengths": "快速响应, 高性价比, 通用任务",
        "docs": "https://dashscope.aliyun.com",
    },
    "glm": {
        "name": "GLM-5.2",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-5.2",
        "env_key": "ZHIPU_API_KEY",
        "pricing": "输入 ¥50/1M tokens, 输出 ¥50/1M tokens",
        "strengths": "Agent能力T0级, 工具调用, 自研GLM架构",
        "docs": "https://open.bigmodel.cn",
    },
    "kimi": {
        "name": "Kimi K2.7",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2.7",
        "env_key": "MOONSHOT_API_KEY",
        "pricing": "输入 ¥12/1M tokens, 输出 ¥12/1M tokens",
        "strengths": "256K无损上下文, 92%召回率, 长文档分析",
        "docs": "https://platform.moonshot.cn",
    },
    "doubao": {
        "name": "豆包 Seed 2.0 Pro",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-2.0-pro",
        "env_key": "DOUBAO_API_KEY",
        "pricing": "输入 ¥0.8/1M tokens, 输出 ¥2/1M tokens",
        "strengths": "多模态最强(90.66分), C端体验最佳",
        "docs": "https://console.volcengine.com/ark",
    },
}


def list_providers() -> list[str]:
    """列出所有已注册的提供商ID"""
    return list(PROVIDERS.keys())


def get_provider_info(provider: str) -> dict:
    """
    获取提供商详细信息

    Args:
        provider: 提供商ID (如 'deepseek', 'qwen')

    Returns:
        包含 name, base_url, model, pricing, strengths 等字段的字典
    """
    if provider not in PROVIDERS:
        raise KeyError(f"未知提供商: {provider}. 可用: {list_providers()}")
    info = PROVIDERS[provider].copy()
    # 检查API Key是否已配置
    info["api_configured"] = bool(os.getenv(info["env_key"]))
    return info


def get_env_path() -> Path:
    """获取 .env 文件路径"""
    return PROJECT_ROOT / ".env"


def load_dotenv_if_exists() -> bool:
    """加载 .env 文件 (如果存在)"""
    env_path = get_env_path()
    if env_path.exists():
        # 手动解析 .env 文件 (避免依赖 python-dotenv)
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value and key not in os.environ:
                        os.environ[key] = value
        return True
    return False


# 模块加载时自动加载 .env
load_dotenv_if_exists()
