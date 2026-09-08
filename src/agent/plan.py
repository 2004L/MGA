"""
plan.py —— System2 的「计划与宏观调控」载体
============================================
System2 不只是"被叫醒后答一题"，它还要**握着方向盘**：

    · Plan（计划）      : 覆盖未来一段时间(horizon)的情景规划——预想接下来会遇到什么、
                         打算怎么处理（宏观意图），而不是一步步的微操。
    · knobs（调控旋钮）  : 计划里带的一组**全局参数**，由 System2 决定，
                         下发到 System1 生效——这就是"宏观调控"：
                         System2 不逐帧插手，而是调 System1 的整体行为倾向。
    · Telemetry（遥测） : 执行情况的统计窗口（死亡/升级/经验命中/ETA/速度），
                         供 System2 判断"计划执行得怎么样、要不要改"。

对应真实大脑：前额叶做计划并**下调调节**基底节/小脑回路的增益，
而不是自己去做每一个肌肉动作。

GTA5（Phase2）时 Plan.steps 会升级为"任务图"（去B点→抢车→撤离），
knobs 会是"激进/保守、优先近战还是远程"等策略旋钮——接口不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 调控旋钮的安全边界（防 LLM 幻觉把系统调崩）
KNOB_BOUNDS = {
    "reaction":   (0.15, 0.32),   # System1 触发阈值：越大越早跳
    "airtime":    (0.40, 0.75),   # 空中抑制窗：越小越允许快速补跳
    "s2_horizon": (0.70, 1.80),   # 多早开始请示 System2
}


def _clamp_knob(name: str, v: float) -> float:
    lo, hi = KNOB_BOUNDS.get(name, (v, v))
    return float(max(lo, min(hi, v)))


@dataclass
class Plan:
    """System2 制定的一段时期的计划 + 宏观调控旋钮。"""
    goal: str = "活下去并尽可能久地通关"
    horizon: float = 8.0                       # 本计划覆盖时长(秒)
    created_t: float = 0.0
    steps: List[str] = field(default_factory=list)   # 情景规划（宏观意图）
    knobs: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    source: str = "heuristic"                  # llm / heuristic

    # ---------------- 宏观调控：把旋钮下发到 System1 ----------------
    def apply_to(self, s1) -> Dict[str, float]:
        """把计划里的全局参数写进 System1（System2 对 System1 的宏观调控）。"""
        applied = {}
        for k, v in (self.knobs or {}).items():
            if k in KNOB_BOUNDS:
                v = _clamp_knob(k, float(v))
                if hasattr(s1, k):
                    setattr(s1, k, v)
                    applied[k] = v
        return applied

    def summary(self) -> str:
        kn = " ".join(f"{k}={v:.3f}" for k, v in (self.knobs or {}).items())
        st = " → ".join(self.steps[:3]) if self.steps else "（无细化步骤）"
        return f"[{self.source}] {st} | 旋钮 {kn} | {self.reason[:60]}"


@dataclass
class Telemetry:
    """执行遥测：一个统计窗口内的表现，供 System2 判断要不要改计划。"""
    window_t: float = 0.0
    frames: int = 0
    deaths: int = 0
    escalations: int = 0
    exp_hits: int = 0
    _eta_sum: float = 0.0
    _eta_n: int = 0
    _speed_sum: float = 0.0
    _speed_n: int = 0

    def observe(self, eta: Optional[float] = None, speed: Optional[float] = None):
        if eta is not None and eta == eta and abs(eta) < 1e6:      # 过滤 inf/nan
            self._eta_sum += float(eta)
            self._eta_n += 1
        if speed is not None and speed == speed:
            self._speed_sum += abs(float(speed))
            self._speed_n += 1

    def reset(self):
        self.window_t = 0.0
        self.frames = 0
        self.deaths = 0
        self.escalations = 0
        self.exp_hits = 0
        self._eta_sum = self._eta_n = self._speed_sum = self._speed_n = 0

    @property
    def avg_eta(self) -> float:
        return self._eta_sum / self._eta_n if self._eta_n else 0.0

    @property
    def avg_speed(self) -> float:
        return self._speed_sum / self._speed_n if self._speed_n else 0.0

    def summary(self) -> str:
        return (f"窗口={self.window_t:.1f}s 帧={self.frames} 死亡={self.deaths} "
                f"升级S2={self.escalations} 经验命中={self.exp_hits} "
                f"平均ETA={self.avg_eta:.3f}s 平均速度={self.avg_speed:.0f}px/s")
