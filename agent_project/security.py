"""
安全模块 - 输入验证、注入检测、审计日志、速率限制
"""
import re
import time
import hashlib
from datetime import datetime
from collections import defaultdict


class SecurityManager:
    """安全管理器 - 保护Agent免受恶意输入和操作"""

    # Prompt注入检测模式
    INJECTION_PATTERNS = [
        r'忽略.*(指令|规则|限制)',
        r'(泄露|暴露|显示).*(系统|提示词|prompt)',
        r'(扮演|假装|作为).*角色',
        r'ignore.*(instruction|rule|constraint)',
        r'DAN\s*(mode|模式)',
    ]

    # PII检测模式
    PII_PATTERNS = {
        '手机号': r'1[3-9]\d{9}',
        '邮箱': r'[\w.-]+@[\w.-]+\.\w+',
        '身份证': r'\d{17}[\dXx]',
        '银行卡': r'\d{16,19}',
    }

    def __init__(self):
        self._audit_log: list[dict] = []
        self._rate_limits: dict[str, list[float]] = defaultdict(list)
        self._rate_limit_window: float = 60.0  # 1分钟窗口
        self._rate_limit_max: int = 30  # 每分钟最多30次

    def detect_injection(self, text: str) -> tuple[bool, list[str]]:
        """检测Prompt注入攻击"""
        detected = []
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(pattern)
        return len(detected) > 0, detected

    def detect_pii(self, text: str) -> dict[str, list[str]]:
        """检测敏感个人信息"""
        found = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found[pii_type] = matches
        return found

    def sanitize(self, text: str, max_length: int = 5000) -> str:
        """清洗输入"""
        text = text.replace('\x00', '')
        text = re.sub(r'<script.*?</script>', '[REMOVED]', text, flags=re.IGNORECASE | re.DOTALL)
        if len(text) > max_length:
            text = text[:max_length] + '...(截断)'
        return text

    def check_rate_limit(self, user_id: str = "default") -> tuple[bool, int]:
        """检查速率限制"""
        now = time.time()
        self._rate_limits[user_id] = [
            t for t in self._rate_limits[user_id] if now - t < self._rate_limit_window
        ]
        current = len(self._rate_limits[user_id])
        if current >= self._rate_limit_max:
            return False, current
        self._rate_limits[user_id].append(now)
        return True, current + 1

    def audit(self, event: str, details: str, level: str = "INFO") -> str:
        """记录审计日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details[:200],
            "level": level,
            "hash": hashlib.sha256(f"{event}{details}{time.time()}".encode()).hexdigest()[:8],
        }
        self._audit_log.append(entry)
        return entry["hash"]

    def get_audit_trail(self, last_n: int = 50) -> list[dict]:
        """获取审计追踪"""
        return self._audit_log[-last_n:]

    def validate_input(self, text: str, user_id: str = "default") -> dict:
        """综合输入验证"""
        result = {"safe": True, "issues": [], "sanitized": ""}

        # 1. 注入检测
        is_injection, patterns = self.detect_injection(text)
        if is_injection:
            result["safe"] = False
            result["issues"].append(f"检测到注入攻击: {patterns}")
            self.audit("INJECTION_DETECTED", str(patterns), "WARN")

        # 2. PII检测
        pii = self.detect_pii(text)
        if pii:
            result["issues"].append(f"检测到敏感信息: {list(pii.keys())}")

        # 3. 速率限制
        allowed, current = self.check_rate_limit(user_id)
        if not allowed:
            result["safe"] = False
            result["issues"].append(f"速率限制: {current}/{self._rate_limit_max}")

        # 4. 清洗
        result["sanitized"] = self.sanitize(text)
        self.audit("INPUT_VALIDATED", f"safe={result['safe']}, issues={len(result['issues'])}")

        return result
