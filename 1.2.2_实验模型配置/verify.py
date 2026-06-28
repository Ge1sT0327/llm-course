"""
配置验证与诊断工具 — 一键检查所有API配置状态
"""
import os
import sys
from .providers import PROVIDERS, PROJECT_ROOT, load_dotenv_if_exists


def verify_config() -> dict:
    """
    验证全局配置, 返回完整的诊断报告

    Returns:
        dict: {
            "status": "ok" | "partial" | "none",
            "providers": {...},
            "issues": [...],
        }
    """
    report = {
        "status": "none",
        "project_root": str(PROJECT_ROOT),
        "python_version": sys.version,
        "providers": {},
        "issues": [],
        "available_count": 0,
        "total_count": len(PROVIDERS),
    }

    # 检查 .env 文件
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv_if_exists()
        report["env_file"] = str(env_path)
    else:
        report["env_file"] = None
        report["issues"].append(
            "未找到 .env 文件. 请在项目根目录创建:\n"
            f"  {PROJECT_ROOT / '.env'}\n"
            "  内容示例见 .env.example"
        )

    # 检查每个提供商
    for provider_id, info in PROVIDERS.items():
        key = info["env_key"]
        value = os.getenv(key, "")
        configured = bool(value)
        masked = value[:4] + "****" + value[-4:] if len(value) > 8 else "****" if value else ""

        report["providers"][provider_id] = {
            "name": info["name"],
            "model": info["model"],
            "configured": configured,
            "key_preview": masked if configured else "未配置",
            "pricing": info["pricing"],
        }
        if configured:
            report["available_count"] += 1

    # 判断整体状态
    if report["available_count"] == 0:
        report["status"] = "none"
        report["issues"].append(
            "所有提供商均未配置 API Key.\n"
            "请注册并获取至少一个平台的API Key:\n"
            "  DeepSeek: https://platform.deepseek.com (推荐, 最便宜)\n"
            "  Qwen:     https://dashscope.aliyun.com\n"
            "  GLM:      https://open.bigmodel.cn"
        )
    elif report["available_count"] < len(PROVIDERS):
        report["status"] = "partial"
    else:
        report["status"] = "ok"

    # 检查 openai 库
    try:
        import openai
        report["openai_version"] = openai.__version__
    except ImportError:
        report["issues"].append("未安装 openai 库. 请运行: pip install openai")

    return report


def diagnose():
    """打印格式化的配置诊断报告"""
    report = verify_config()

    print("=" * 60)
    print("  模型配置诊断报告")
    print("=" * 60)
    print(f"  项目路径: {report['project_root']}")
    print(f"  Python:   {report['python_version'].split()[0]}")
    print(f"  .env文件: {'已找到' if report['env_file'] else '未找到'}")
    print(f"  OpenAI库: {report.get('openai_version', '未安装')}")
    print()

    # 提供商状态表
    print(f"  已配置: {report['available_count']}/{report['total_count']} 个提供商")
    print(f"  {'提供商':<16} {'模型':<22} {'状态':<10} {'Key预览':<20}")
    print(f"  {'-'*16} {'-'*22} {'-'*10} {'-'*20}")
    for pid, p in report["providers"].items():
        status_icon = "OK" if p["configured"] else "--"
        print(f"  {pid:<16} {p['model']:<22} {status_icon:<10} {p['key_preview']:<20}")

    # 问题
    if report["issues"]:
        print(f"\n  注意事项:")
        for i, issue in enumerate(report["issues"], 1):
            print(f"    {i}. {issue}")

    # 建议
    if report["available_count"] == 0:
        print(f"\n  >>> 快速开始:")
        print(f"      1. 访问 https://platform.deepseek.com 注册 (免费)")
        print(f"      2. 获取 API Key")
        print(f"      3. 在项目根目录创建 .env 文件:")
        print(f"         DEEPSEEK_API_KEY=sk-你的key")
        print(f"      4. 运行: from config import get_client")
    else:
        print(f"\n  >>> 运行: from config import get_client")
        print(f"      client = get_client()  # 自动选择可用提供商")

    print("=" * 60)
    return report
