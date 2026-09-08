"""
MGA · Token 经济（内在动机）模块（src/economy/token.py）
========================================================
把强化学习的「生存成本 + 任务奖赏」合并成一个 token 经济系统：
- 每次推理消耗 token（生存成本）：System 2 贵，System 1 近乎零。
- 干好任务获得 token 配额（奖赏），只来自可验证环境结果，禁止自评分。
- 余额低 -> 焦虑 -> 优先走 System 1（物理预判），正好与预判帧机制自洽：
  生存压力逼着 agent 自己学会「能靠物理公式解决的，绝不动用大模型」。

对应报告 5.3 的「内在动机」补齐方案，且是 cost-sensitive RL 的可落地 reward。

依赖：仅标准库（零依赖）。
"""


class TokenEconomy:
    def __init__(self, init: float = 1000.0, safe: float = 400.0,
                 min_: float = 100.0, cost_sys1: float = 0.1,
                 cost_sys2: float = 100.0, novelty_bonus: float = 0.1):
        self.B = init
        self.init = init
        self.safe, self.min_ = safe, min_
        self.cost1, self.cost2 = cost_sys1, cost_sys2
        self.novelty_bonus = novelty_bonus
        # 成就仪表盘
        self.earned = 0.0          # 累计赚取（越强越赚）
        self.spent = 0.0           # 累计花费
        self.streak = 0            # 连续成功次数
        self.best_streak = 0
        self.tasks = 0

    def act(self, use_sys2: bool, env_reward: float,
            novelty: float = 0.0) -> str:
        """
        每步调用：扣生存成本，加任务奖赏，加探索 bonus，返回状态机档位。
        返回: "NORMAL" | "CONSERVE" | "STARVING"
        """
        cost = self.cost2 if use_sys2 else self.cost1
        self.B -= cost
        self.spent += cost
        self.B += env_reward
        if env_reward > 0:
            self.earned += env_reward
        if novelty > 0:
            self.B += self.novelty_bonus * novelty

        # 连胜 -> 解锁更高 safe 上限（信任额度提升）
        if env_reward > 0:
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            self.safe = min(self.safe + self.streak * 1.0, self.init * 0.9)
        else:
            self.streak = 0
            self.safe = max(self.safe - 20.0, self.min_ * 2.0)

        if self.B <= self.min_:
            return "STARVING"      # 强制 System 1，主动找简单任务回血
        if self.B <= self.safe:
            return "CONSERVE"      # 优先省钱
        return "NORMAL"

    def efficiency(self) -> float:
        """效率比：同样一件事花更少 token 完成 = 更聪明（成就指标）。"""
        return (self.earned / self.spent) if self.spent > 1e-9 else 0.0

    def alive(self) -> bool:
        return self.B > 0.0

    def report(self) -> str:
        return (f"[TokenEconomy] B={self.B:.1f} safe={self.safe:.1f} "
                f"earned={self.earned:.1f} spent={self.spent:.1f} "
                f"eff={self.efficiency():.3f} streak={self.streak}")


if __name__ == "__main__":
    eco = TokenEconomy()
    print("== Token 经济演示：余额低时被迫走 System 1，余额充足才唤醒 System 2 ==")
    scenario = [
        (False, 50.0, 0.0),   # 静默帧，干成小任务
        (False, 50.0, 0.0),
        (True,  0.0, 0.0),    # 关键帧，System2 调用（烧钱），任务没奖
        (False, 80.0, 0.0),
        (True,  120.0, 0.0),  # 关键帧但干成大任务，回血
        (False, 60.0, 0.0),
    ]
    for i, (sys2, reward, nov) in enumerate(scenario):
        mode = eco.act(sys2, reward, nov)
        print(f"step{i}: use_sys2={sys2} reward={reward:>5} -> {mode:10s} "
              f"{eco.report()}")
