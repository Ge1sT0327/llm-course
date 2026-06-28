"""
会话管理器 - 智能知识库Agent的对话引擎
支持多轮对话上下文管理、记忆系统、对话摘要和流式输出
"""
from typing import Optional
from collections import deque


class ConversationManager:
    """多轮对话管理器 - 处理上下文窗口、记忆压缩和对话状态"""

    def __init__(self, system_prompt: str = "", max_history: int = 20,
                 max_context_tokens: int = 4000, summary_trigger: int = 3000):
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens
        self.summary_trigger = summary_trigger
        self.messages: list[dict] = []
        self._summary: str = ""
        self._token_count: int = 0
        self._session_id: str = ""
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user_message(self, content: str) -> None:
        """添加用户消息, 自动检查是否需要压缩历史"""
        self.messages.append({"role": "user", "content": content})
        self._token_count += len(content) // 2  # 粗略估算: 中文约2字符/token
        if self._token_count > self.summary_trigger:
            self._compress_history()

    def add_assistant_message(self, content: str, metadata: Optional[dict] = None) -> None:
        """添加助手回复"""
        msg = {"role": "assistant", "content": content}
        if metadata:
            msg["metadata"] = metadata
        self.messages.append(msg)
        self._token_count += len(content) // 2

    def get_context(self, last_n: Optional[int] = None) -> list[dict]:
        """获取当前对话上下文, 用于传给LLM"""
        n = last_n or self.max_history
        return self.messages[-n:] if n > 0 else self.messages

    def _compress_history(self) -> None:
        """压缩历史: 保留system prompt + 最近4轮 + 摘要"""
        if len(self.messages) <= 1:
            return
        old = self.messages[1:-4]  # 保留system + 最后4条
        summary_text = " | ".join(
            m.get("content", "")[:100] for m in old
            if m.get("role") in ("user", "assistant")
        )
        self._summary = summary_text[:500]
        preserved = self.messages[:1]  # system prompt
        preserved.append({"role": "system", "content": f"[对话历史摘要] {self._summary}"})
        preserved.extend(self.messages[-4:])
        self.messages = preserved
        self._token_count = sum(len(m.get("content", "")) // 2 for m in preserved)

    def clear(self) -> None:
        """重置对话"""
        self.messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        self._summary = ""
        self._token_count = 0

    @property
    def turn_count(self) -> int:
        """对话轮次"""
        return sum(1 for m in self.messages if m["role"] == "user")

    @property
    def summary(self) -> str:
        return self._summary

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "summary": self._summary,
            "turn_count": self.turn_count,
            "token_count": self._token_count,
        }
