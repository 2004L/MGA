"""
types.py —— 分层决策智能体的公共数据结构
========================================
Scene     : 一帧感知结果（像素 + 检测到的游戏元素）
Decision  : 一次决策（动作 + 依据 + 来源 + 置信度 + 是否要求升级）
AgentStats: 闭环运行统计（S1/S2 分工占比、经验写入、死亡次数…）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Scene:
    """感知层输出：一帧的结构化理解（三级降级后的结果）。"""
    frame: Any = None                       # (H,W,3) RGB，原始像素（必要时保留）
    dino_x: Optional[float] = None          # 恐龙相对 x（游戏特化字段，通用化见 game_adapter）
    obstacles: List = field(default_factory=list)   # List[Obstacle]
    elements: List = field(default_factory=list)    # 桌面检测到的 UI 元素（与 obstacles 并存，桌面用）
    t: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.frame is not None


@dataclass
class Decision:
    """一次决策。source 标明来自 S1 还是 S2（架构真相：谁做的决定写清楚）。"""
    action: str = "wait"            # jump / squat / wait / none
    eta: float = float("inf")       # 障碍到达时间(秒)
    vx: float = 0.0                 # 障碍水平速度(px/s，负=向左)
    react: float = 0.22             # 本次生效的触发阈值(秒) = 基线 + 习得残差 + 经验偏置
    source: str = "S1"              # S1 / S2
    confidence: float = 1.0
    escalate: bool = False          # S1 请求升级给 S2
    why: str = ""
    tid: Optional[int] = None       # 目标障碍 id（用于"已触发"去重）
    key: str = ""                   # 情形键（经验检索用）
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def fires(self) -> bool:
        """是否真的要执行动作（wait/none 不执行）。"""
        return self.action in ("jump", "squat")


@dataclass
class AgentStats:
    """闭环统计。用于验证「S1 干绝大部分、S2 只在关键帧醒」这一核心主张。"""
    frames: int = 0
    s1_frames: int = 0
    s2_calls: int = 0
    s2_llm: int = 0            # 真·LLM 决策次数（架构真相：区分真 LLM 与启发式兜底）
    s2_heuristic: int = 0      # LLM 不可用时的启发式兜底次数
    s2_reasoner: int = 0       # 本地大脑替身(reasoner)决策次数（离线演示用，不冒充真 LLM）
    experiences: int = 0       # 写入持久记忆的经验条数
    exp_hits: int = 0          # S1 命中历史经验、直接快决策的次数
    deaths: int = 0
    suppress: int = 0          # 空中抑制（防二段跳/防抖）
    restarts: int = 0
    plans: int = 0             # System2 制定/修订计划的次数（宏观调控次数）
    corrects: int = 0          # 死亡/失误时「及时纠错」次数（诊断+修正）

    def report(self) -> str:
        tot = max(1, self.frames)
        s1p = 100.0 * self.s1_frames / tot
        s2p = 100.0 * self.s2_calls / tot
        return (f"frames={self.frames}  S1={self.s1_frames}({s1p:.1f}%)  "
                f"S2={self.s2_calls}({s2p:.1f}%)  "
                f"[真LLM={self.s2_llm}/本地reasoner={self.s2_reasoner}/启发={self.s2_heuristic}]  "
                f"经验写入={self.experiences}  经验命中={self.exp_hits}  "
                f"撞死={self.deaths}  重开={self.restarts}  空中抑制={self.suppress}  "
                f"宏观调控={self.plans}  及时纠错={self.corrects}")
