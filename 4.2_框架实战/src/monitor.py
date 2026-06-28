"""
监控系统 - Chain执行监控、耗时统计、Token计数
"""
import time
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ChainStats:
    """单次Chain执行的统计数据"""
    step_name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: str = ""

    @property
    def duration_ms(self) -> float:
        """执行耗时 (毫秒)"""
        return (self.end_time - self.start_time) * 1000

    @property
    def total_tokens(self) -> int:
        """总Token消耗"""
        return self.input_tokens + self.output_tokens


class ChainMonitor:
    """
    Chain执行监控器

    收集每个步骤的:
    - 执行耗时
    - Token消耗 (输入 + 输出)
    - 成功/失败状态
    """

    def __init__(self):
        self.stats: list[ChainStats] = []
        self._current_start: float = 0.0
        self._current_name: str = ""

    def start_step(self, name: str):
        """开始监控一个步骤"""
        self._current_name = name
        self._current_start = time.time()

    def end_step(self, input_tokens: int = 0, output_tokens: int = 0,
                 success: bool = True, error: str = ""):
        """结束当前步骤的监控"""
        stat = ChainStats(
            step_name=self._current_name,
            start_time=self._current_start,
            end_time=time.time(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            error=error,
        )
        self.stats.append(stat)

    def report(self) -> str:
        """生成监控报告"""
        if not self.stats:
            return "无监控数据"

        lines = [
            "=" * 60,
            "  Chain 执行监控报告",
            "=" * 60,
            f"{'步骤':<20} {'耗时(ms)':<12} {'输入Token':<12} {'输出Token':<12} {'状态':<8}",
            "-" * 60,
        ]

        total_time = 0
        total_input = 0
        total_output = 0
        success_count = 0

        for s in self.stats:
            status = "OK" if s.success else f"ERROR: {s.error}"
            lines.append(
                f"{s.step_name:<20} {s.duration_ms:<12.1f} "
                f"{s.input_tokens:<12} {s.output_tokens:<12} {status:<8}"
            )
            total_time += s.duration_ms
            total_input += s.input_tokens
            total_output += s.output_tokens
            if s.success:
                success_count += 1

        lines.append("-" * 60)
        lines.append(f"总计: {len(self.stats)} 步骤, {success_count}/{len(self.stats)} 成功")
        lines.append(f"总耗时: {total_time:.1f}ms ({total_time/1000:.2f}s)")
        lines.append(f"总Token: {total_input + total_output} (输入: {total_input}, 输出: {total_output})")

        return "\n".join(lines)
