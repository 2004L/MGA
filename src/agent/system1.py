"""
system1.py —— System1 小模型：快速经验性决策 + 即时执行
=======================================================
**不是纯物理/规则架构**，而是三层叠加的"小模型"：

    1. 物理基线   : MotionModel 纯数学估 vx → ETA（零神经网络，µs 级）
    2. 习得残差   : LearnedResidual —— System2 的每一条纠正回流训练它（知识蒸馏），
                   预测"触发时机补偿量"。解析物理为基，习得残差为进化。
    3. 经验检索   : 命中历史经验（System2 写入的）→ 直接快决策，置信度拉满
                   （像人的"本能"：见过就会，不用再想）

分工原则（核心）：
    · System1 **逐帧**运行，负责 99%+ 的帧，保证低延迟。
    · 遇到"没见过的情形"（经验检索未命中）→ 打 escalate 标记，交给 System2 学。
    · System2 学完后写经验 + 训练残差 → **下一次同样的情形 System1 自己就能快决策**。
      这就是"从 S2 思考 → 固化为 S1 习惯"的类脑学习闭环。

依赖：numpy + precheck/motion_model + predictive/frame（均为零框架）。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from predictive.frame import LearnedResidual
from precheck.motion_model import TrackerHub

from .types import Decision

# 习得残差的状态维度：[常数项, ETA, |vx|归一化, 是否飞鸟]
DIM = 4
# 残差输出的补偿量安全夹取（秒）：避免小模型早期乱跳
DELTA_CLAMP = 0.08


def _cx(obstacle) -> float:
    """取障碍中心 x（优先用 demo_dino_real 的精确 box，缺失则退回 x_rel）。"""
    try:
        from demo_dino_real import _obstacle_box
        return float(_obstacle_box(obstacle)[1])
    except Exception:
        return float(getattr(obstacle, "x_rel", 0.0))


class System1:
    """System1 小模型：快、经验性、逐帧。"""

    def __init__(self, dino_x: float = 44.0, reaction: float = 0.22,
                 airtime: float = 0.6, memory=None, s2_horizon: float = 1.2,
                 dim: int = DIM):
        self.hub = TrackerHub(dino_x=dino_x, max_age=5)
        self.reaction = reaction          # 物理基线触发阈值(秒)
        self.airtime = airtime            # 滞空估计：空中抑制窗，防二段跳/防抖
        self.memory = memory              # ExperienceMemory（可为 None）
        self.s2_horizon = s2_horizon      # 距离动作还有多久时，允许升级问 S2
        self.dim = dim
        self.residual = LearnedResidual(dim=dim)   # ★ 小模型：习得残差
        self.n_samples = 0                # 已从 System2 学到的纠正样本数

        self._fired = set()               # 已触发过的障碍 id（防重复跳）
        self._escalated = set()           # 已问过 S2 的障碍 id（每种情形只学一次）
        self._airborne_until = 0.0
        self.n_suppress = 0
        self.n_exp_hits = 0

    # ---------------- 情形键：经验的索引 ----------------
    @staticmethod
    def situation_key(is_bird: bool, vx: float, pair: bool = False) -> str:
        """把连续状态离散成"情形键"，作为经验检索的索引。
        例：cactus|v3|solo / bird|v5|pair"""
        bucket = int(abs(vx) // 150)
        return f"{'bird' if is_bird else 'cactus'}|v{min(bucket, 8)}|" \
               f"{'pair' if pair else 'solo'}"

    # ---------------- 状态向量 ----------------
    def state_vec(self, eta: float, vx: float, is_bird: bool) -> np.ndarray:
        return np.array([
            1.0,
            float(np.clip(eta, 0.0, 2.0)),
            float(np.clip(abs(vx) / 1000.0, 0.0, 2.0)),
            1.0 if is_bird else 0.0,
        ], dtype=float)

    # ---------------- 感知接入 ----------------
    def observe(self, obstacles: Sequence, t: float):
        """喂 MotionModel，返回 live=[(tid, mm, is_bird, cx), ...]。"""
        pairs = [(_cx(o), bool(getattr(o, "is_bird", False))) for o in obstacles]
        return self.hub.update(pairs, t)

    def reset(self):
        """一局结束/重开时清空跟踪状态（但**不清空**习得残差与经验——那是要留存的）。"""
        self._fired.clear()
        self._escalated.clear()
        self._airborne_until = 0.0
        try:
            self.hub.trackers.clear()
            self.hub.age.clear()
        except Exception:
            pass

    # ---------------- 决策（每帧调用，必须快） ----------------
    def decide(self, live: Sequence[Tuple], t: float) -> Decision:
        # 空中抑制：游戏无二段跳，滞空期内不重复发跳也不重复问 S2
        if t < self._airborne_until:
            self.n_suppress += 1
            return Decision(action="wait", source="S1", why="airborne-suppress")

        cands = []
        for (tid, mm, is_bird, cx) in live:
            if tid in self._fired:
                continue
            try:
                eta = float(mm.arrival_time(cx))
            except Exception:
                continue
            if not math.isfinite(eta):
                continue                      # 静止物(vx≈0)→ETA 无穷→自然过滤
            if eta < 0:
                # 已越过恐龙：再跳就是"迟到的无效跳"（真机实测会白白送命）。
                # 直接标记为已处理，后面不再考虑它。
                self._fired.add(tid)
                continue
            cands.append((eta, tid, mm, is_bird, cx))

        if not cands:
            return Decision(action="none", source="S1", why="no-moving-obstacle")

        cands.sort(key=lambda x: x[0])
        eta, tid, mm, is_bird, cx = cands[0]
        pair = len(cands) >= 2 and (cands[1][0] - eta) < 0.55

        key = self.situation_key(is_bird, mm.vx, pair)
        state = self.state_vec(eta, mm.vx, is_bird)

        # ★ 小模型：习得残差预测"触发时机补偿"
        try:
            delta = float(np.clip(self.residual.predict(state)[0],
                                  -DELTA_CLAMP, DELTA_CLAMP))
        except Exception:
            delta = 0.0

        # ★ 经验检索：命中历史经验 → 本能式快决策，置信度拉满
        hits = []
        if self.memory is not None:
            try:
                hits = self.memory.retrieve(key, top_k=1) or []
            except Exception:
                hits = []
        # 过滤掉宏观调控/计划类元经验（标题不含情形键 '|'），避免污染 S1 的情形检索
        hits = [h for h in hits
                if "|" in str(h.get("title", "")) or "|" in str(h.get("key", ""))]
        exp_hit = bool(hits)
        if exp_hit:
            self.n_exp_hits += 1

        react = max(0.05, self.reaction + delta)
        confidence = 0.5 + (0.25 if exp_hit else 0.0) + (0.25 if self.n_samples > 0 else 0.0)

        # 升级条件：没见过这种情形 + 还来得及问（eta 未到必须动作的临界）+ 该障碍没问过
        escalate = (not exp_hit) and (tid not in self._escalated) \
            and (react <= eta < max(react + 0.05, self.s2_horizon))

        action = "wait"
        if eta < react:
            action = "squat" if is_bird else "jump"

        return Decision(
            action=action, eta=eta, vx=float(getattr(mm, "vx", 0.0)), react=react,
            source="S1", confidence=confidence, escalate=escalate,
            why=("经验命中→本能快决策" if exp_hit else
                 ("新情形→请求S2" if escalate else "基线决策")),
            tid=tid, key=key,
            raw={"state": state, "delta": delta, "exp_hit": exp_hit,
                 "pair": pair, "exp": hits[0] if hits else None},
        )

    # ---------------- 执行回执 ----------------
    def mark_fired(self, tid):
        if tid is not None:
            self._fired.add(tid)

    def mark_airborne(self, t: float):
        self._airborne_until = t + self.airtime

    def mark_escalated(self, tid):
        if tid is not None:
            self._escalated.add(tid)

    # ---------------- 学习：System2 的纠正回流（知识蒸馏） ----------------
    def learn(self, state: np.ndarray, react_delta: float):
        """System2 给出"应该补偿多少"→ 训练习得残差，让 S1 下次自己就会。"""
        try:
            corr = np.zeros(self.dim, dtype=float)
            corr[0] = float(np.clip(react_delta, -DELTA_CLAMP, DELTA_CLAMP))
            self.residual.train_sample(state, corr)
            self.n_samples += 1
        except Exception:
            pass
