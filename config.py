"""
配置模块快捷入口 — 指向 1.2.2_实验模型配置

所有实验统一使用:
    from config import get_client, verify_config
"""
import sys
import importlib.util
from pathlib import Path

# 定位真正的 config 模块
_real_config = Path(__file__).resolve().parent / "1.2.2_实验模型配置" / "__init__.py"
spec = importlib.util.spec_from_file_location("_config_core", _real_config)
_core = importlib.util.module_from_spec(spec)
sys.modules["_config_core"] = _core
spec.loader.exec_module(_core)

# 重新导出所有公共API
get_client = _core.get_client
LLMClient = _core.LLMClient
verify_config = _core.verify_config
diagnose = _core.diagnose
list_providers = _core.list_providers
get_provider_info = _core.get_provider_info
PROVIDERS = _core.PROVIDERS

__all__ = [
    "get_client", "LLMClient",
    "verify_config", "diagnose",
    "list_providers", "get_provider_info",
    "PROVIDERS",
]
