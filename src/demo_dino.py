"""
MGA · 端到端验证 Demo：Chrome 小恐龙（预判帧 System1）
======================================================
回到原始设计的核心创新演示：
- System1(预判帧) 用**纯数学**预测障碍物到达恐龙的剩余时间 ETA，无需任何神经网络/大模型。
- 仅当 ETA < 反应时间(或距离 < 阈值)才唤醒 System2(选 jump/squat)。
- 90% 帧 System1 静默(纯物理运算)，10% 关键帧 System2 决策 → 天然契合 token 经济
  (静默成本≈0，关键帧贵)。
- System2 的「正确反应余量」回流训练 System1 残差模型(RLS)——S2→S1 蒸馏，使「习惯」可习得。

复用：src/predictive/frame.py(Track / RLS 残差 / 校准窗口)
      src/economy/token.py(TokenEconomy 内在动机)
依赖：仅 numpy。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from predictive.frame import Track
from economy.token import TokenEconomy


@dataclass
class DinoFrame:
    trigger: bool
    eta: float
    dist: float
    info: str = ""


class DinoSystem1:
    """预判帧 System1：纯数学跟踪障碍 X，预测到达恐龙的 ETA。"""

    def __init__(self, dino_x: float = 50.0, reaction_time: float = 0.45,
                 arrive_th: float = 70.0, initial_vx: float = -320.0,
                 threshold: float = 0.1):
        self.dino_x = dino_x
        self.reaction_time = reaction_time      # ETA 阈值(秒)：小于即该跳/蹲
        self.arrive_th = arrive_th              # 距离阈值(px)：小于即触发
        self.threshold = threshold
        self.initial_vx = initial_vx
        # Track 跟踪障碍 X(1维)：初速度估计为向左的 vx，校准窗口反推真值。
        self.track = Track(pos=np.array([450.0]),
                           vel=np.array([initial_vx]), dim=1, threshold=threshold)
        self.t = 0.0
        self.vx = initial_vx

    def reset(self):
        """新障碍出现时重置 Track(保留习得残差)，避免跨障碍位置跳变污染速度估计。"""
        self.track.reset(self.initial_vx)

    def step(self, t: float, obs_x: float) -> DinoFrame:
        """喂实测障碍 X；返回是否触发 System2 + ETA。Track 更新由调用方决定(支持回流)。"""
        m = np.array([obs_x], dtype=float)
        if not self.track.feed(t, m):
            return DinoFrame(False, 1e9, 1e9, "calibrating")
        pred = self.track.forecast(t)
        vx = self.track.vel[0]
        eta = (self.dino_x - pred[0]) / (-vx) if vx < 0 else 1e9
        dist = abs(self.dino_x - pred[0])     # 绝对距离(障碍在右/左都按接近度判)
        # 仅当 ETA 落在 [0, 反应时间) 才唤醒 System2(障碍仍在右侧且即将到达)
        trigger = (0 <= eta < self.reaction_time) or (dist < self.arrive_th)
        return DinoFrame(bool(trigger), float(eta), float(dist),
                         f"pred_x={pred[0]:.1f}")

    def update(self, t: float, obs_x: float, sys2_correction: Optional[float] = None):
        """普通更新 / 带 System2 纠正回流(蒸馏残差)。"""
        m = np.array([obs_x], dtype=float)
        if sys2_correction is not None:
            self.track.update(t, m, sys2=np.array([sys2_correction], dtype=float))
        else:
            self.track.update(t, m)


def system2_decide(kind: str) -> str:
    """System2(视觉大脑)：地面障碍→跳，高空飞鸟→蹲。贵调用，仅 System1 触发时唤醒。"""
    return "squat" if kind == "bird" else "jump"


def simulate_env(kind: str, action: str) -> bool:
    """环境结果：动作与障碍类型匹配即躲过(成功)。"""
    return action == ("squat" if kind == "bird" else "jump")


def run_game(n_frames: int = 300, dt: float = 0.05, seed: int = 7):
    rng = np.random.default_rng(seed)
    sys1 = DinoSystem1()
    eco = TokenEconomy(init=1000.0, safe=400.0, cost_sys1=0.1, cost_sys2=100.0)

    W = 450.0
    vx = -320.0
    spawn_gap = (0.3, 0.6)
    next_spawn = 1.0
    obs_id = 0
    active = None                      # (obs_id, obs_x, kind, vx)
    fired_ids = set()                 # 已对当前障碍触发过 System2(避免重复扣费)
    sys1_ticks = sys2_ticks = 0
    hits = misses = 0

    print(f"== Chrome 小恐龙 Demo: System1 纯数学 ETA 预判 + System2 触发 ==")
    print(f"   帧数={n_frames} dt={dt}s 反应阈值={sys1.reaction_time}s 距离阈值={sys1.arrive_th}px")
    for i in range(n_frames):
        t = (i + 1) * dt
        sys1.t = t
        # 生成障碍
        if active is None and t >= next_spawn:
            obs_id += 1
            kind = "bird" if rng.random() < 0.3 else "ground"
            active = (obs_id, W, kind, vx)
            next_spawn = t + rng.uniform(*spawn_gap)
        if active is not None:
            oid, ox, kind, v = active
            ox += v * dt
            active = (oid, ox, kind, v)
            # 漏判：障碍掠过恐龙仍未触发 → 撞
            if ox < sys1.dino_x - 30 and oid not in fired_ids:
                misses += 1
                eco.act(use_sys2=False, env_reward=-50.0)
                active = None
                print(f"  t={t:5.2f} CRASH(漏判) {kind} {eco.report()}")
                continue
            r = sys1.step(t, ox)
            if r.trigger and oid not in fired_ids:
                fired_ids.add(oid)
                action = system2_decide(kind)
                ok = simulate_env(kind, action)
                reward = 30.0 if ok else -40.0
                hits += 1 if ok else 0
                eco.act(use_sys2=True, env_reward=reward)
                # S2→S1 蒸馏：残差目标 = 当前 ETA 与反应阈值之差，让接近时更灵敏
                sys1.update(t, ox, sys2_correction=float(r.eta - sys1.reaction_time))
                sys2_ticks += 1
                print(f"  t={t:5.2f} KEYFRAME(S2) eta={r.eta:6.3f} {kind:7s}→{action:5s} "
                      f"{'OK ' if ok else 'MISS'} {eco.report()}")
            else:
                sys1.update(t, ox)
                eco.act(use_sys2=False, env_reward=0.1)   # 静默小奖(物理预判成立)
                sys1_ticks += 1
                if i % 25 == 0:
                    print(f"  t={t:5.2f} SILENT(S1)  eta={r.eta:6.3f} {kind:7s} {eco.report()}")
            if ox < -50:
                active = None
        else:
            eco.act(use_sys2=False, env_reward=0.1)
            sys1_ticks += 1

    total = sys1_ticks + sys2_ticks
    print("\n== 结果 ==")
    print(f"  总帧={total}  System1静默={sys1_ticks}({100*sys1_ticks/total:.1f}%)  "
          f"System2关键帧={sys2_ticks}({100*sys2_ticks/total:.1f}%)")
    print(f"  触发决策={hits+misses}  成功躲过={hits}  漏判撞车={misses}")
    print(f"  [TokenEconomy] {eco.report()}")
    print(f"  校准速度 vx={sys1.track.vel[0]:.1f} (真值 {vx})")
    return dict(sys1=sys1_ticks, sys2=sys2_ticks, hits=hits, misses=misses,
                eff=eco.efficiency(), spent=eco.spent, earned=eco.earned)


if __name__ == "__main__":
    run_game()
