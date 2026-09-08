"""
predictive/action.py —— System1 深化：执行延迟预判（纯数学，零神经网络）

## 为什么需要这一层

System1（预判帧）原本只预判**外界物理**：目标还有多久到达。
但从"预判出该动手了"到"动作真正生效"，中间还有**自己这一侧的开销**：

    截图 → 检测 → System1 运算 → 决策 → 发送按键 → 应用收到并响应

这笔开销在 sim 里几乎为零（合成截图，1000+ fps），但在真机上高达
几十到几百毫秒（全屏截图是主要瓶颈）。**不算进去，预判就系统性偏晚**——
表现为"明明预判对了却还是撞了"（真机小恐龙撞第一个仙人掌的根因）。

## 做法

统计每类动作的实际耗时滑动窗口，预测取 **p95（保守）**：
宁可提前一点触发，也不要错过窗口。触发条件从

    eta < reaction_time
变为
    eta < reaction_time + predicted_latency

即"把执行自己的开销也算进提前量"。全部是分位数统计，零神经网络，
保持 System1 的微秒级特性（预测本身 <10µs，只有动作执行是毫秒级）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class ActionLatencyPredictor:
    """动作延迟预测器：滑动窗口 + p95 保守估计。"""

    def __init__(self, window: int = 30, default_ms: float = 0.0,
                 percentile: float = 95.0):
        self.window = int(window)
        self.default_ms = float(default_ms)
        self.percentile = float(percentile)
        self._samples: Dict[str, List[float]] = {}

    # -- 记录 --
    def record(self, action: str, ms: float) -> None:
        buf = self._samples.setdefault(action, [])
        buf.append(float(ms))
        if len(buf) > self.window:
            del buf[0]

    def reset(self, action: Optional[str] = None) -> None:
        if action is None:
            self._samples.clear()
        else:
            self._samples.pop(action, None)

    # -- 预测 --
    def predict(self, action: str = "default", fallback: str = "default") -> float:
        """预测耗时(ms)：该动作样本不足时回退到 fallback，都没有则用 default_ms。"""
        buf = self._samples.get(action) or self._samples.get(fallback)
        if not buf:
            return self.default_ms
        return float(np.percentile(np.array(buf), self.percentile))

    def mean(self, action: str = "default") -> float:
        buf = self._samples.get(action)
        return float(np.mean(buf)) if buf else self.default_ms

    def count(self, action: str = "default") -> int:
        return len(self._samples.get(action, []))

    # -- 报告 --
    def stats(self) -> str:
        if not self._samples:
            return "无样本"
        parts = []
        for k, buf in self._samples.items():
            if not buf:
                continue
            parts.append(f"{k}: n={len(buf)} p95={np.percentile(buf, self.percentile):.1f}ms "
                         f"mean={np.mean(buf):.1f}ms")
        return " | ".join(parts)
