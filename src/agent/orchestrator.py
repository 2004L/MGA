"""
orchestrator.py —— 编排器：把感知 / System1 / System2 / 记忆 / 执行串成闭环
==========================================================================
每帧流程（对应架构图）：

    感知(三级降级) → System1 小模型逐帧决策 ─┬─ 常规：直接执行（99%+ 的帧）
                                            └─ 新情形：升级 System2
                                                        ├─ LLM 快速推理 + 情景规划
                                                        ├─ 经验持久化写入存储
                                                        └─ 纠正回流训练 System1 残差
                                                                ↓
                                                   下次同类情形 S1 自己就会（习惯固化）
    执行 → 回执校验 → 死亡检测 → 自动重开（闭环永不断）

关键设计（诚实版）：
    · LLM 不在逐帧关键路径上（60fps 下 LLM 必卡死）。System2 的"主导"体现在
      **策略层**：它决定"这类情形该怎么处理"并写成经验、训练 System1。
    · 想让 S2 直接覆盖当帧动作，用 s2_override=True（默认关，安全优先）。
"""

from __future__ import annotations

import time
from typing import List, Optional

from .types import AgentStats, Decision
from .plan import Plan, Telemetry


class AgentOrchestrator:
    """分层决策智能体的主循环。"""

    def __init__(self, adapter, system1, system2, executor, memory=None,
                 dt: float = 1.0 / 60.0, max_gap_frames: int = 240,
                 auto_restart: bool = True, s2_override: bool = False,
                 verbose: bool = True):
        self.adapter = adapter
        self.s1 = system1
        self.s2 = system2
        self.exec = executor
        self.memory = memory
        self.dt = dt
        self.max_gap_frames = max_gap_frames
        self.auto_restart = auto_restart
        self.s2_override = s2_override
        self.verbose = verbose
        self.stats = AgentStats()
        self.t = 0.0
        self._gap = 0
        self._dino_refined = False
        self._last_mid: Optional[str] = None
        # ---- System2 宏观调控 / 纠错 支撑字段 ----
        self._tele = Telemetry()          # 执行遥测窗口（供 System2 复盘）
        self._plan_every = 10.0          # System2 重规划周期(秒)
        self._last_plan_t = -999.0
        self._cur_plan: Optional[Plan] = None
        self._trace: List[str] = []       # 近期决策轨迹（纠错时给 System2 看）
        self._last_key = ""
        self._last_action = "wait"
        self._last_eta = 0.0
        self._last_vx = 0.0
        self._last_state = None

    # ---------------- 启动：聚焦 + 把游戏唤醒 ----------------
    def bootstrap(self) -> bool:
        if self.exec is not None:
            self.exec.ensure_focus()
            time.sleep(0.3)
        # 空格既是"开始"也是"撞死后重开"；先按一次确保游戏在跑
        self.adapter.act("jump")
        time.sleep(0.4)
        if self.exec is not None and not self.adapter.sim:
            ok = self.exec.probe(self.adapter.grab)
            if not ok:
                print("  [启动] 按键似乎没生效，再聚焦重试一次…")
                self.exec.ensure_focus()
                time.sleep(0.3)
                self.adapter.act("jump")
                time.sleep(0.4)
                self.exec.probe(self.adapter.grab)
        return True

    # ---------------- 死亡/结束检测 ----------------
    def _dead(self, live) -> bool:
        # sim 模式：用适配器内部碰撞真值判死（离线也能驱动 System2 纠错闭环）
        if getattr(self.adapter, "sim", False):
            return bool(getattr(self.adapter, "_dead_flag", False))
        """用"是否有在移动的障碍"判死：GAME OVER 屏只有静止的分数文字等假障碍。"""
        has_moving = any(abs(getattr(mm, "vx", 0.0)) > 30.0 for (_, mm, _, _) in live)
        if has_moving:
            self._gap = 0
            return False
        self._gap += 1
        return self._gap > self.max_gap_frames and self.t > 4.0

    def _restart(self):
        self.stats.deaths += 1
        self._tele.deaths += 1
        # ★ 及时纠错：死亡瞬间让 System2 反思并修正（诊断 + 降权/重写误导经验 + 调旋钮 + 回流）
        if self.s2 is not None:
            self._correct()
        if self.auto_restart:
            if getattr(self.adapter, "sim", False):
                self.adapter.sim_reset()       # sim：清场重开
            else:
                self.adapter.act("jump")        # 真机：空格重开
            self.s1.reset()
            self._gap = 0
            self.stats.restarts += 1
            if self.verbose:
                print(f"  t={self.t:6.2f} [死亡] 自动重开（本局第 {self.stats.deaths} 次）")
        else:
            raise SystemExit("游戏结束（auto_restart=False）")

    # ---------------- System2 宏观调控：复盘遥测 → 修订计划 → 下发旋钮 ----------------
    def _make_plan(self, goal: str = "活下去并尽可能久地通关") -> Optional[Plan]:
        if self.s2 is None:
            return None
        ctx = {
            "goal": goal,
            "telemetry": self._tele.summary(),
            "current": {"reaction": self.s1.reaction, "airtime": self.s1.airtime,
                        "s2_horizon": self.s1.s2_horizon},
            "deaths": self._tele.deaths,
            "avg_speed": self._tele.avg_speed,
        }
        plan = self.s2.make_plan(ctx, t=self.t)
        self.stats.plans += 1
        applied = plan.apply_to(self.s1)        # ★ 宏观调控：把旋钮写进 System1
        self._cur_plan = plan
        self._tele.reset()
        if self.verbose:
            tail = f" → 生效 {applied}" if applied else ""
            print(f"  t={self.t:6.2f} [System2·宏观调控:{plan.source}] {plan.summary()}{tail}")
        return plan

    # ---------------- System2 及时纠错：诊断失败并修正（类人脑反思） ----------------
    def _correct(self) -> Optional[dict]:
        if self.s2 is None:
            return None
        st = self.stats
        st.corrects += 1
        last_exp = ""
        if self._last_mid and self.memory is not None:
            try:
                rows = self.memory.recent(limit=1) or []
                if rows:
                    last_exp = str(rows[0].get("summary", ""))[:120]
            except Exception:
                last_exp = ""
        ctx = {
            "key": self._last_key or "",
            "last_action": self._last_action,
            "eta": self._last_eta, "vx": self._last_vx,
            "last_experience": last_exp,
            "reaction": self.s1.reaction, "airtime": self.s1.airtime,
            "trace": self._recent_trace(),
        }
        corr = self.s2.correct_error(ctx)
        # ① 降权/重写误导经验（反经验主义 + 写入正确经验）
        # 注意：challenge 必须传经验 id(uuid)，不能用情形键字符串（否则静默无效）
        if self.memory is not None and corr.get("fix") in ("challenge", "rewrite"):
            if self._last_mid:
                self.memory.challenge(self._last_mid, success=False)
        if corr.get("rewrite"):
            self.s2.write_experience(corr.get("key") or self._last_key,
                                     corr.get("action", "jump"),
                                     corr.get("reason", ""),
                                     float(corr.get("react_delta", 0.0)),
                                     tags=["game", "dino", "s2", "corrected"])
            st.experiences += 1
        # ② 调旋钮（宏观调控）：System2 直接改 System1 的整体行为倾向
        if corr.get("fix") == "adjust_knobs" and corr.get("knobs"):
            p = Plan(knobs=corr["knobs"], reason="纠错调整：" + corr.get("reason", ""),
                     source=corr.get("source", "reasoner"))
            p.apply_to(self.s1)
            self._cur_plan = p
        # ③ 回流训练 System1 残差（用纠正后的 react_delta，让本能变正确）
        if self._last_state is not None:
            self.s1.learn(self._last_state, float(corr.get("react_delta", 0.0)))
        if self.verbose:
            print(f"  t={self.t:6.2f} [System2·及时纠错:{corr.get('source')}] "
                  f"诊断={corr.get('diagnosis','')[:46]} | fix={corr.get('fix')} "
                  f"→ {corr.get('reason','')[:46]}")
        return corr

    def _recent_trace(self) -> str:
        if not self._trace:
            return "(无显著决策)"
        return "\n".join(self._trace[-10:])

    # ---------------- 等游戏出现（避免"游戏没开"被误判成一直撞死） ----------------
    def wait_for_game(self, timeout: float = 12.0) -> bool:
        """轮询直到感知真的看到游戏元素（恐龙）。看不到就一直重试定位/聚焦。

        没有这一步，游戏没开时会表现为"每 240 帧撞死一次"的假死亡，
        统计全被污染（真机踩过：Chrome 被关掉后跑出 6 次假死亡）。
        """
        deadline = time.time() + timeout
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            if self.adapter.region is None:
                self.adapter.locate()
            try:
                scene = self.adapter.grab()
                s = self.adapter.perceive(scene)
                if s.dino_x is not None:
                    if self.verbose and attempt > 1:
                        print(f"  [等待] 第 {attempt} 次尝试后看到游戏（dino_x={s.dino_x:.0f}）")
                    self.s1.hub.dino_x = s.dino_x
                    self._dino_refined = True
                    return True
            except Exception as e:
                # 不再静默：等待游戏元素时出错若无声，会表现为"一直等不到"却查不出原因
                if self.verbose:
                    print(f"  [等待] ⚠️ 第 {attempt} 次探测异常：{type(e).__name__}: {e}")
            if self.exec is not None:
                self.exec.ensure_focus()
            time.sleep(1.0)
        print(f"  [等待] {timeout:.0f}s 内没看到游戏元素 —— 请确认 chrome://dino 已打开且可见")
        return False

    # ---------------- 主循环 ----------------
    def run(self, frames: int = 1000) -> AgentStats:
        self.bootstrap()
        if self.adapter.region is None:
            self.adapter.locate()
        if not self.adapter.sim and not self.wait_for_game():
            print("  [致命] 游戏不可见，停止（避免空转产生假死亡统计）")
            return self.stats

        # 开局先让 System2 定一份计划并下发宏观调控旋钮到 System1
        if self.s2 is not None:
            self._make_plan()
            self._last_plan_t = self.t

        for i in range(frames):
            f0 = time.perf_counter()
            self.adapter.tick(self.dt)

            scene = self.adapter.grab()
            s = self.adapter.perceive(scene)
            if s.dino_x is not None and not self._dino_refined:
                self.s1.hub.dino_x = s.dino_x          # 真机首帧校准恐龙位
                self._dino_refined = True

            live = self.s1.observe(s.obstacles, self.t)
            self.stats.frames += 1

            if self._dead(live):
                self._restart()
                self.t += self.dt
                continue

            d = self.s1.decide(live, self.t)
            self.stats.s1_frames += 1
            self._last_state = d.raw.get("state")     # 供死亡时回流训练 S1
            self._tele.observe(d.eta, d.vx)            # 遥测：喂 System2 宏观调控
            self._tele.frames += 1
            if d.raw.get("exp_hit"):
                self.stats.exp_hits += 1
                self._tele.exp_hits += 1

            # ---- 新情形 → 升级 System2（学习 + 写记忆 + 训练 S1）----
            if d.escalate and d.tid is not None:
                self._escalate(d)
                self._tele.escalations += 1

            # ---- 执行 ----
            if d.fires:
                self.adapter.act(d.action)
                self.s1.mark_fired(d.tid)
                self.s1.mark_airborne(self.t)
                # 记录最近决策（死亡时 System2 纠错要看）
                self._last_action = d.action
                self._last_eta = d.eta
                self._last_vx = d.vx
                self._last_key = d.key
                self._trace.append(f"t={self.t:6.2f} {('S2' if d.source=='S2' else 'S1')} "
                                   f"{d.action} eta={d.eta:.3f} key={d.key}")
                if self.verbose:
                    tag = "S2" if d.source == "S2" else "S1"
                    print(f"  t={self.t:6.2f} KEYFRAME({tag}) ETA={d.eta:6.3f}s "
                          f"react={d.react:.3f} "
                          f"{'bird' if d.action == 'squat' else 'cactus'}→{d.action}  "
                          f"{d.why}")

            self.t += self.dt
            # 宏观调控：按周期让 System2 复盘遥测并修订计划（下调 System1 增益）
            if self.t - self._last_plan_t >= self._plan_every:
                self._make_plan()
                self._last_plan_t = self.t
            # 保持近似实时节奏（真机：别把 CPU 跑满抢帧）
            el = time.perf_counter() - f0
            if not self.adapter.sim and el < self.dt:
                time.sleep(self.dt - el)

        # 汇总 S2 的真 LLM / 本地 reasoner / 启发式次数
        self.stats.s2_llm = getattr(self.s2, "n_llm", 0)
        self.stats.s2_heuristic = getattr(self.s2, "n_heuristic", 0)
        self.stats.s2_reasoner = getattr(self.s2, "n_reasoner", 0)
        self.stats.suppress = self.s1.n_suppress
        return self.stats

    # ---------------- 升级：让 System2 思考一次 ----------------
    def _escalate(self, d: Decision):
        st = self.stats
        st.s2_calls += 1
        st_raw = d.raw or {}
        sit = {
            "kind": "bird" if d.action == "squat" or ("bird" in d.key) else "cactus",
            "vx": d.vx, "eta": d.eta, "react": d.react,
            "pair": bool(st_raw.get("pair", False)), "key": d.key,
        }
        out = self.s2.think(sit)

        # ① 经验持久化写入存储（System2 的核心职责）
        mid = self.s2.write_experience(d.key, out["action"], out["reason"],
                                       out["react_delta"])
        if mid:
            st.experiences += 1
            self._last_mid = mid

        # ② 纠正回流训练 System1 的习得残差（知识蒸馏：S2 的习惯 → S1 的本能）
        if st_raw.get("state") is not None:
            self.s1.learn(st_raw["state"], out["react_delta"])

        self.s1.mark_escalated(d.tid)

        # ③ 可选：让 S2 直接覆盖当帧动作（默认关，安全优先）
        if self.s2_override and out["action"] in ("jump", "squat") and d.eta < d.react:
            d.action = out["action"]
            d.source = "S2"

        if self.verbose:
            print(f"  t={self.t:6.2f} [升级S2:{out['source']}] 情形={d.key} → "
                  f"{out['action']} Δ={out['react_delta']:+.3f}s | {out['reason'][:60]}")
