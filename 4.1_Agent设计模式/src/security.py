"""
安全模块 - Agent安全验证与沙箱执行
"""
import os
import re
import tempfile
from pathlib import Path
from typing import Optional


class SecurityValidator:
    """安全验证器 - 检测和阻止恶意输入/操作"""

    # 危险模式列表
    DANGEROUS_PATTERNS = [
        r'os\.system\s*\(',
        r'subprocess\.(call|run|Popen)\s*\(',
        r'__import__\s*\(',
        r'eval\s*\(',
        r'exec\s*\(',
        r'open\s*\([^)]*[\'"]w',  # 写文件操作
        r'shutil\.(rmtree|move|copy)',
        r'import\s+(os|subprocess|shutil|sys)',
    ]

    # 允许的安全路径前缀
    SAFE_PATHS = [
        tempfile.gettempdir(),  # 临时目录
    ]

    @classmethod
    def validate_code(cls, code: str) -> tuple[bool, Optional[str]]:
        """
        检查代码是否包含危险操作

        Args:
            code: 待检查的代码字符串

        Returns:
            (is_safe, error_message)
        """
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"检测到危险操作: 匹配模式 '{pattern}'"
        return True, None

    @classmethod
    def validate_path(cls, path: str) -> tuple[bool, Optional[str]]:
        """
        检查文件路径是否安全 (防止路径遍历攻击)

        Args:
            path: 文件路径

        Returns:
            (is_safe, error_message)
        """
        try:
            resolved = str(Path(path).resolve())
        except (OSError, ValueError):
            return False, f"无效路径: {path}"

        # 检查是否在安全目录范围内
        for safe_path in cls.SAFE_PATHS:
            if resolved.startswith(str(Path(safe_path).resolve())):
                return True, None

        # 检查是否在项目目录内
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if resolved.startswith(project_root):
            return True, None

        return False, f"路径 '{path}' 不在允许的目录范围内"

    @classmethod
    def sanitize_input(cls, text: str) -> str:
        """
        清洗用户输入, 移除潜在的注入内容

        Args:
            text: 原始输入文本

        Returns:
            清洗后的文本
        """
        # 移除null字节
        text = text.replace('\x00', '')
        # 限制长度
        if len(text) > 10000:
            text = text[:10000] + "...(已截断)"
        return text


def safe_execute(
    code: str,
    working_dir: Optional[str] = None,
    timeout: int = 30,
) -> tuple[bool, str]:
    """
    在沙箱环境中安全执行Python代码

    Args:
        code: 要执行的代码
        working_dir: 工作目录 (默认使用临时目录)
        timeout: 超时秒数

    Returns:
        (success, output)
    """
    # 安全验证
    is_safe, error = SecurityValidator.validate_code(code)
    if not is_safe:
        return False, f"安全验证失败: {error}"

    working_dir = working_dir or tempfile.gettempdir()

    # 写入临时文件
    tmp_path = os.path.join(working_dir, "_agent_exec.py")
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(code)
    except OSError as e:
        return False, f"无法写入临时文件: {e}"

    # 在子进程中安全执行
    import subprocess
    try:
        result = subprocess.run(
            ['python', tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
            env={**os.environ, 'PYTHONPATH': working_dir},
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"代码执行超时 ({timeout}秒)"
    finally:
        # 清理临时文件
        try:
            os.remove(tmp_path)
        except OSError:
            pass
