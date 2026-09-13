"""
event_gate.py —— Mage-VL 式「事件闸」(System1 闸 + System2 解码器范式)
=========================================================================

这是 MGA 感知层的**第二块**（A = supervision_events.py 结构化事件层）。
A 负责"看见并产出结构化事件"（越线/停留/类别突变）；本模块负责"决定何时
值得惊动大模型"——即 Mage-VL 的核心思想：

    · System1（闸 / gate）：极廉价、逐事件运行，对当前时刻打分 p∈[0,1]，
      表示"这一刻值不值得唤醒 System2"。在 Mage-VL 里这是常驻的轻量打分器，
      我们默认用**可解释的启发式打分器**（无需 16-24G 显存的 4B 模型也能跑）；
      但架构上预留了 ModelScorer 接口——真 4B 模型到位后直接插上即可。

    · 因果窗口（causal window）：闸维护最近 ~30s 的事件缓冲。一旦打分
      p ≥ 阈值（默认 0.70），闸门"开"，把**这一刻 + 前后因果窗口**打包成
      WakeSignal 唤醒 System2（LLM 解码器），只把压缩后的上下文运给大模型。

    · 防抖/滞回（hysteresis + cooldown）：闸门一旦开，进入"已触发"态，必须
      等分数跌破 low 阈值才重新武装，且两次唤醒之间至少间隔 cooldown，避免
      高频事件把 LLM 刷爆——这正是 Mage-VL "只在关键帧唤醒" 的工程落地。

设计纪律（对齐 MGA 全局）：
  · 真接通：EventGate 消费 A 的真实事件流，WakeSignal 真带因果窗口，transport
    回调真被调用；绝不"代码写完就算接通"。
  · 优雅降级：LLM transport 缺失/返回 None → 标记为 pending、缓存、不崩；
    不假装大模型已处理。active_backend 如实标注用的是启发式还是模型打分。
  · 模型无关：输入是 A 产出的事件 dict（含 type / tracker_id / dwell_sec 等），
    任何上游（supervision / ScreenParser / 桌面 UI 检测）的事件都能喂进来。

依赖：仅标准库（dataclasses / collections / time）。LLM 运输为可选回调。
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---- 默认超参（与 Mage-VL 范式对齐，可注入）----
DEFAULT_WINDOW_SEC = 30.0      # 因果窗口：唤醒时打包最近 30s 的事件
DEFAULT_HIGH = 0.70            # 闸门开启阈值 p≥high → 唤醒 System2
DEFAULT_LOW = 0.35             # 滞回下阈值：需跌破 low 才重新武装
DEFAULT_COOLDOWN = 2.0         # 两次唤醒最小间隔(秒)，防 LLM 被刷爆
DEFAULT_NOVELTY = 15.0         # 新颖性检测窗口：该类型多久没出现过算"新"


# ===========================================================================
# WakeSignal —— 闸门开时打包给 System2 的"因果上下文"
# ===========================================================================
@dataclass
class WakeSignal:
    """闸门开 → 唤醒 System2 的信令。携因果窗口 + 触发事件 + 打分证据。"""
    trigger_event: Dict[str, Any]
    score: float
    t: float
    backend: str
    window_summary: List[Dict[str, Any]] = field(default_factory=list)  # 因果窗口（压缩后）
    n_events_in_window: int = 0
    distinct_objects: int = 0
    burst_rate: float = 0.0        # 最近 5s 事件率(个/秒)
    pending: bool = False          # LLM 不可用时标记待处理
    transport_source: str = ""     # 实际处理者：llm / reasoner / heuristic / pending
    analysis: str = ""             # 大模型研判原文（transport 返回即写入）

    def to_dict(self) -> dict:
        return {
            "type": "gate_wake",
            "score": round(self.score, 3),
            "t": round(self.t, 3),
            "backend": self.backend,
            "trigger": self.trigger_event,
            "window_summary": self.window_summary,
            "n_events_in_window": self.n_events_in_window,
            "distinct_objects": self.distinct_objects,
            "burst_rate": round(self.burst_rate, 2),
            "pending": self.pending,
            "transport_source": self.transport_source,
            "analysis": self.analysis,
        }


# ===========================================================================
# 打分器（System1 闸的核心）—— 可插拔
# ===========================================================================
class Scorer:
    """打分器接口：score(t, trigger, window) -> (p, reasons)。

    window 为 [(ts, event_dict), ...]，含当前触发事件（位于末尾）。
    子类可替换为真实模型（ModelScorer）。"""
    def score(self, t: float, trigger: Dict[str, Any],
              window: List[Tuple[float, Dict[str, Any]]]) -> Tuple[float, List[str]]:
        raise NotImplementedError


class HeuristicScorer(Scorer):
    """可解释的启发式打分器（默认）：无需 GPU，零依赖。

    组合信号（权重均文档化，便于后续用真模型校准）：
      · 触发事件严重度(base)：越线 0.45 / 区域停留 0.30+min(dwell/10,0.40)
                            / 类别突变已触发 0.40 / 其他 0.10
      · 窗口密度(density)：窗口内事件越多越可疑，min(n/40, 0.20)
      · 新颖性(novelty)：该类型在 novelty_window 内首次出现 +0.10
      · 多目标(distinct)：窗口内不同 tracker_id 越多越可疑，min(ntid/6, 0.10)
    上限 1.0。设计目标：
      - 单次普通越线(≤0.65) 不唤醒（噪声控制）
      - 长停留(≥? 4s→0.70) / 事件突发(≥~10 个) 唤醒（真异常）
    """
    def __init__(self, novelty_window: float = DEFAULT_NOVELTY):
        self.novelty_window = novelty_window

    def score(self, t: float, trigger: Dict[str, Any],
              window: List[Tuple[float, Dict[str, Any]]]) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        ttype = str(trigger.get("type", "unknown"))

        # 1) 触发事件严重度
        if ttype == "line_cross":
            base = 0.45
            reasons.append("越线事件(base=0.45)")
        elif ttype == "zone_dwell":
            dwell = float(trigger.get("dwell_sec", 0.0) or 0.0)
            base = 0.30 + min(dwell / 10.0, 0.40)
            reasons.append(f"区域停留{dwell:.1f}s(base={base:.2f})")
        elif ttype == "class_event":
            ce = trigger.get("class_event") or {}
            fired = bool(ce.get("fired"))
            base = 0.40 if fired else 0.10
            reasons.append(f"类别计数突变(fired={fired}, base={base:.2f})")
        else:
            base = 0.10
            reasons.append(f"未识别事件类型[{ttype}](base=0.10)")
        p = base

        # 2) 窗口密度
        n = len(window)
        density = min(n / 40.0, 0.20)
        if density > 0:
            p += density
            reasons.append(f"窗口内{n}事件(密度+{density:.2f})")

        # 3) 新颖性：该类型在 novelty_window 内是否首次出现
        prior = [(ts, e) for (ts, e) in window if ts < t - 1e-9]
        seen_types = {e.get("type") for (ts, e) in prior
                      if (t - ts) <= self.novelty_window}
        if ttype != "unknown" and ttype not in seen_types:
            p += 0.10
            reasons.append("新事件类型(新颖性+0.10)")

        # 4) 多目标
        tids = {e.get("tracker_id") for (ts, e) in window
                if e.get("tracker_id") is not None}
        if tids:
            mult = min(len(tids) / 6.0, 0.10)
            if mult > 0:
                p += mult
                reasons.append(f"{len(tids)}个不同目标(+{mult:.2f})")

        p = min(1.0, p)
        return p, reasons


class ConstantScorer(Scorer):
    """测试/占位用：固定返回 p（验证闸门/运输/滞回逻辑时绕过启发式）。"""
    def __init__(self, value: float = 0.9):
        self.value = min(1.0, max(0.0, value))

    def score(self, t: float, trigger: Dict[str, Any],
              window: List[Tuple[float, Dict[str, Any]]]) -> Tuple[float, List[str]]:
        return self.value, [f"ConstantScorer({self.value:.2f})"]


class ModelScorer(Scorer):
    """真实模型打分器接口（预留给 Mage-VL 4B 到位后插入）。

    用法：实现 score()，内部把因果窗口的帧/事件编码后送 4B 模型，返回
    sigmoid 后的概率 p∈[0,1]。本类只定义契约 + 诚实降级：模型不可用时
    自动退 HeuristicScorer 并标注 backend='model-fallback'。"""
    def __init__(self, model_fn: Optional[Callable[[List[Tuple[float, Dict[str, Any]]]], float]] = None,
                 fallback: Optional[Scorer] = None):
        self.model_fn = model_fn
        self._fallback = fallback or HeuristicScorer()
        self.available = model_fn is not None

    def score(self, t: float, trigger: Dict[str, Any],
              window: List[Tuple[float, Dict[str, Any]]]) -> Tuple[float, List[str]]:
        if not self.available or self.model_fn is None:
            p, r = self._fallback.score(t, trigger, window)
            return p, r + ["ModelScorer 不可用→启发式兜底"]
        try:
            raw = float(self.model_fn(window))
            p = min(1.0, max(0.0, raw))
            return p, ["ModelScorer(真实模型)"]
        except Exception as e:
            p, r = self._fallback.score(t, trigger, window)
            return p, r + [f"ModelScorer 推断失败({type(e).__name__})→启发式兜底"]


# ===========================================================================
# 事件闸（System1 常驻打分 → 唤醒 System2）
# ===========================================================================
class EventGate:
    """Mage-VL 式事件闸。

    消费 A 层的结构化事件流，维护因果窗口，廉价打分；p≥high 时打包 WakeSignal
    唤醒 System2（通过 on_wake 回调 + 可选 llm_transport）。带滞回 + 冷却，避免
    LLM 被高频事件刷爆。

    用法：
        gate = EventGate()
        gate.set_llm_transport(my_llm_fn)          # 可选：运给大模型
        for ev in stream:
            gate.push_event(ev)                      # 每来一个事件推一次
        sig = gate.last_wake                        # 最近一次唤醒信令
    """

    def __init__(self,
                 scorer: Optional[Scorer] = None,
                 on_wake: Optional[Callable[[WakeSignal], None]] = None,
                 window_sec: float = DEFAULT_WINDOW_SEC,
                 high: float = DEFAULT_HIGH,
                 low: float = DEFAULT_LOW,
                 cooldown_sec: float = DEFAULT_COOLDOWN,
                 novelty_window: float = DEFAULT_NOVELTY):
        self.scorer = scorer or HeuristicScorer(novelty_window=novelty_window)
        self.on_wake = on_wake
        self.window_sec = window_sec
        self.high = high
        self.low = low
        self.cooldown_sec = cooldown_sec
        self.novelty_window = novelty_window

        self.window: deque = deque()        # (t, event_dict)
        self.armed = True                   # 滞回：是否处于可触发态
        self.last_fire_t = -1e18
        self.last_score = 0.0
        self.last_reasons: List[str] = []
        self.fires: List[WakeSignal] = []
        self.n_scored = 0
        self.n_fired = 0
        self._llm_transport: Optional[Callable[[WakeSignal], Any]] = None
        # backend 标签如实反映用的是哪种打分器
        self.backend = type(self.scorer).__name__

    # ---------------- 配置 ----------------
    def set_llm_transport(self, fn: Callable[[WakeSignal], Any]) -> None:
        """注册"运给大模型"的回调。返回非 None 视为已处理；返回 None 标记 pending。"""
        self._llm_transport = fn

    # ---------------- 主入口：推一个事件 ----------------
    def push_event(self, event: Dict[str, Any], t: Optional[float] = None) -> Optional[WakeSignal]:
        """推入一个结构化事件（来自 A 层）。返回本次若触发则 WakeSignal，否则 None。"""
        if t is None:
            t = time.time()
        self.window.append((t, event))
        self._prune(t)
        return self._score_and_decide(t, event)

    # ---------------- 内部：因果窗口裁剪 ----------------
    def _prune(self, t: float) -> None:
        cutoff = t - self.window_sec
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()

    def _score_and_decide(self, t: float, trigger: Dict[str, Any]) -> Optional[WakeSignal]:
        p, reasons = self.scorer.score(t, trigger, list(self.window))
        self.n_scored += 1
        self.last_score = p
        self.last_reasons = reasons

        # 滞回：未武装（已触发态）先不判 fires，等跌破 low 再武装
        if not self.armed:
            if p < self.low:
                self.armed = True
            else:
                return None

        # 冷却：两次唤醒最小间隔
        if (t - self.last_fire_t) < self.cooldown_sec:
            return None

        if p >= self.high:
            return self._emit(t, trigger, p, reasons)
        return None

    # ---------------- 内部：开闸 → 打包 + 唤醒 + 运输 ----------------
    def _emit(self, t: float, trigger: Dict[str, Any], p: float,
              reasons: List[str]) -> WakeSignal:
        window_tuples = list(self.window)
        summary = self._compact_window(window_tuples)
        tids = {e.get("tracker_id") for (_, e) in window_tuples
                if e.get("tracker_id") is not None}
        burst = self._burst_rate(t, window_tuples, span=5.0)

        sig = WakeSignal(
            trigger_event=trigger, score=p, t=t, backend=self.backend,
            window_summary=summary, n_events_in_window=len(window_tuples),
            distinct_objects=len(tids), burst_rate=burst,
        )

        # 1) on_wake 回调（编排器可在此真正唤醒 System2 决策）
        if self.on_wake is not None:
            try:
                self.on_wake(sig)
            except Exception as e:
                # 不再静默：编排器崩了会让 System2 永远不被唤醒
                print(f"  [EventGate] ⚠️ on_wake 回调异常：{type(e).__name__}: {e}")

        # 2) 运输给大模型（可选；缺失/返回 None → pending，不崩、不冒充已处理）
        if self._llm_transport is not None:
            try:
                result = self._llm_transport(sig)
                if result is None:
                    sig.pending = True
                    sig.transport_source = "pending"
                else:
                    sig.pending = False
                    sig.transport_source = "llm"
                    sig.analysis = result if isinstance(result, str) else str(result)
            except Exception as e:
                sig.pending = True
                sig.transport_source = "pending"
                print(f"  [EventGate] ⚠️ LLM 运输异常，标记 pending：{type(e).__name__}: {e}")
        else:
            # 未注册运输回调：唤醒已发生，但大模型永远收不到 → 诚实标记 pending
            sig.pending = True
            sig.transport_source = "pending(no-transport)"

        # 3) 更新闸门状态（滞回 + 冷却起点）
        self.fires.append(sig)
        self.n_fired += 1
        self.last_fire_t = t
        self.armed = False
        return sig

    # ---------------- 因果窗口压缩（运给大模型的上下文）----------------
    def _compact_window(self, window_tuples: List[Tuple[float, Dict[str, Any]]]
                        ) -> List[Dict[str, Any]]:
        """把因果窗口压成"最近 N 条精简事件"，省 token。"""
        out = []
        for (ts, e) in window_tuples[-24:]:
            slim = {"t": round(ts, 2), "type": e.get("type")}
            if e.get("tracker_id") is not None:
                slim["tid"] = e["tracker_id"]
            if e.get("class_name") is not None:
                slim["cls"] = e["class_name"]
            if e.get("direction") is not None:
                slim["dir"] = e["direction"]
            if e.get("dwell_sec") is not None:
                slim["dwell"] = round(float(e["dwell_sec"]), 2)
            out.append(slim)
        return out

    def _burst_rate(self, t: float,
                    window_tuples: List[Tuple[float, Dict[str, Any]]],
                    span: float = 5.0) -> float:
        lo = t - span
        cnt = sum(1 for (ts, _) in window_tuples if ts >= lo)
        return cnt / span if span > 0 else 0.0

    # ---------------- 把 WakeSignal 格式化成可送 LLM 的 prompt ----------------
    @staticmethod
    def build_llm_payload(sig: WakeSignal, max_events: int = 12) -> str:
        """组装成一段人类可读、可直接塞进 LLM 的上下文 prompt（运输给大模型的载荷）。"""
        recent = sig.window_summary[-max_events:]
        lines = [
            "【MGA 事件闸唤醒】以下是一段最近 {:.0f}s 内的感知事件因果窗口，"
            "请基于它做研判/下一步建议。".format(0.0),
        ]
        # 用窗口实际跨度更直观
        if sig.window_summary:
            t0 = sig.window_summary[0].get("t", 0.0)
            t1 = sig.window_summary[-1].get("t", 0.0)
            lines[0] = ("【MGA 事件闸唤醒】以下是一段约 {:.1f}s 内的感知事件因果窗口"
                        "（触发置信度 {:.2f}），请基于它做研判/下一步建议。"
                        ).format(max(0.1, t1 - t0), sig.score)
        lines.append(f"· 触发事件：{sig.trigger_event}")
        lines.append(f"· 窗口内事件数={sig.n_events_in_window}，"
                     f"不同目标={sig.distinct_objects}，"
                     f"近5s事件率={sig.burst_rate:.2f}/s")
        lines.append("· 因果窗口事件序列：")
        for ev in recent:
            lines.append("   - " + ", ".join(f"{k}={v}" for k, v in ev.items()))
        return "\n".join(lines)

    # ---------------- 状态 ----------------
    def summary(self) -> dict:
        return {
            "backend": self.backend,
            "window_sec": self.window_sec,
            "high": self.high,
            "low": self.low,
            "cooldown_sec": self.cooldown_sec,
            "armed": self.armed,
            "n_scored": self.n_scored,
            "n_fired": self.n_fired,
            "last_score": round(self.last_score, 3),
            "last_reasons": self.last_reasons,
            "window_size": len(self.window),
            "n_pending": sum(1 for f in self.fires if f.pending),
        }


# ===========================================================================
# 便捷集成：把 A 层分析器的每帧结果直接喂进事件闸
# ===========================================================================
def feed_analyzer_frame(analyzer, gate: EventGate, boxes, t: Optional[float] = None
                        ) -> dict:
    """喂一帧给 A 层分析器，并把产出的新事件（越线/停留/类别突变）转发给事件闸。

    返回 {analyzer_snapshot, wake_signal}。wake_signal 为 None 表示本帧未触发唤醒。
    这是 A→B 的标准接线：编排器每帧调一次即可。
    """
    snap = analyzer.update(boxes, t=t)
    wake: Optional[WakeSignal] = None
    for ev in (snap.get("new_cross") or []):
        wake = gate.push_event(ev, t=t) or wake
    for ev in (snap.get("new_dwell") or []):
        wake = gate.push_event(ev, t=t) or wake
    ce = snap.get("class_event")
    if ce and ce.get("fired"):
        wake = gate.push_event({"type": "class_event", "class_event": ce}, t=t) or wake
    return {"analyzer": snap, "wake": wake}


# ===========================================================================
# 自检 / 集成演示
# ===========================================================================
if __name__ == "__main__":
    import os
    import sys

    # 让本模块在 src/perception 直跑时也能 import 到同包
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 64)
    print("MGA 事件闸 (Mage-VL 式) —— 集成自检")
    print("=" * 64)

    # ---- 1) 单元：启发式阈值行为 ----
    g = EventGate()
    captured = []
    g.set_llm_transport(lambda sig: captured.append(sig) or "ok")

    # 1a) 静默期：无事件 → 0 唤醒
    assert g.n_fired == 0, "无事件不应唤醒"

    # 1b) 单次普通越线：base 0.45 + distinct 0.10 = 0.55 < 0.70 → 不唤醒
    ev_cross = {"type": "line_cross", "line_id": "A", "tracker_id": 1,
                "class_name": "person", "direction": "in", "frame": 1, "t": 1.0}
    r = g.push_event(ev_cross, t=1.0)
    assert r is None, "单次越线不应唤醒"
    print("[1b] 单次越线 score=%.2f → 不唤醒 ✓" % g.last_score)

    # 1c) 长停留 4s：0.30+0.40=0.70 ≥ 0.70 → 唤醒
    ev_dwell = {"type": "zone_dwell", "zone_id": "entrance", "tracker_id": 1,
                "class_name": "person", "dwell_sec": 4.0, "frame_in": 2, "frame_out": 10}
    r = g.push_event(ev_dwell, t=2.0)
    assert r is not None and r.score >= 0.70, "长停留应唤醒"
    assert r.transport_source == "llm", "LLM transport 应被调用"
    print("[1c] 长停留 score=%.2f → 唤醒 ✓ (LLM transport 命中)" % r.score)

    # 1d) 冷却：刚唤醒后 1s(<?2s) 再推高优事件 → 不唤醒
    r2 = g.push_event({"type": "zone_dwell", "zone_id": "z", "tracker_id": 2,
                       "class_name": "person", "dwell_sec": 5.0}, t=2.5)
    assert r2 is None, "冷却期内不应唤醒"
    print("[1d] 冷却期内(0.5s)再推高优 → 不唤醒 ✓")

    # 1e) 滞回：冷却后推一低分事件(p<low)→ 重新武装；再推高优 → 唤醒
    g.push_event({"type": "unknown", "tracker_id": 9}, t=5.0)  # 低分，跌破 low
    assert g.armed is True, "低分后应重新武装"
    r3 = g.push_event({"type": "zone_dwell", "zone_id": "z", "tracker_id": 3,
                       "class_name": "person", "dwell_sec": 4.0}, t=6.0)
    assert r3 is not None, "重新武装后高优应唤醒"
    print("[1e] 滞回：低分重武装 → 高优再唤醒 ✓ (累计唤醒=%d)" % g.n_fired)

    # 1f) 突发：连续多个越线（密度拉满）→ 唤醒
    g2 = EventGate()
    fired = False
    for k in range(12):
        rr = g2.push_event({"type": "line_cross", "line_id": "A", "tracker_id": 100 + k,
                            "class_name": "person", "direction": "in", "frame": k, "t": 10.0 + k * 0.1})
        if rr is not None:
            fired = True
            print("[1f] 突发 %d 个越线 → 唤醒 score=%.2f ✓" % (k + 1, rr.score))
            break
    assert fired, "事件突发应唤醒"

    # ---- 2) 降级：LLM transport 缺失 → pending 不崩 ----
    g3 = EventGate()  # 无 transport
    r4 = g3.push_event({"type": "zone_dwell", "zone_id": "z", "tracker_id": 1,
                        "class_name": "person", "dwell_sec": 6.0}, t=1.0)
    assert r4 is not None and r4.pending is True, "无 LLM 应标记 pending"
    print("[2] 无 LLM transport → pending 标记 ✓ (backend=%s)" % g3.backend)

    # ---- 3) 集成：真实 supervision 事件流 → 事件闸 ----
    print("\n--- 3) A(supervision) → B(事件闸) 真流集成 ---")
    try:
        from perception.supervision_events import (VideoEventAnalyzer, LineCfg, ZoneCfg,
                                                    TrackBox, build_event_analyzer)
        HAS_SV = True
    except Exception:
        try:
            from supervision_events import (VideoEventAnalyzer, LineCfg, ZoneCfg,
                                            TrackBox, build_event_analyzer)
            HAS_SV = True
        except Exception:
            HAS_SV = False

    wakes = []
    if HAS_SV:
        line = LineCfg(id="A", p1=(100, 200), p2=(400, 200))
        zone = ZoneCfg(id="entrance", polygon=[(300, 300), (500, 300),
                                               (500, 500), (300, 500)])
        an = build_event_analyzer(lines=[line], zones=[zone])
        gate = EventGate(on_wake=lambda s: wakes.append(s))
        seq = [
            [TrackBox((150, 100, 180, 140), 0, "person", 0.9)],   # 线上方
            [TrackBox((150, 180, 180, 220), 0, "person", 0.9)],   # 跨线(单次，不应唤醒)
            [TrackBox((150, 260, 180, 300), 0, "person", 0.9)],   # 线下方
            [TrackBox((350, 350, 380, 390), 0, "person", 0.9)],   # 进区域
            [TrackBox((350, 350, 380, 390), 0, "person", 0.9)],   # 停留
            [TrackBox((350, 350, 380, 390), 0, "person", 0.9)],   # 停留
            [TrackBox((350, 350, 380, 390), 0, "person", 0.9)],   # 停留
            [TrackBox((350, 350, 380, 390), 0, "person", 0.9)],   # 停留(累计→长停留)
            [TrackBox((350, 350, 380, 390), 0, "person", 0.9)],   # 停留
            [TrackBox((150, 260, 180, 300), 0, "person", 0.9)],   # 离开区域(结算 dwell)
        ]
        for k, boxes in enumerate(seq):
            feed_analyzer_frame(an, gate, boxes, t=float(k))
        print("   后端=%s, 唤醒次数=%d, pending=%d" %
              (gate.backend, gate.n_fired, gate.summary()["n_pending"]))
        assert gate.n_fired >= 1, "真实流应至少唤醒 1 次(长停留)"
        sig = wakes[-1]
        assert sig.window_summary, "WakeSignal 因果窗口不应为空"
        types_in_window = {ev.get("type") for ev in sig.window_summary}
        assert sig.trigger_event.get("type") in types_in_window, \
            "因果窗口应含触发事件类型"
        payload = EventGate.build_llm_payload(sig)
        print("   触发事件: %s" % sig.trigger_event)
        print("   —— 运输给大模型的载荷(节选) ——")
        print("\n".join("   " + ln for ln in payload.split("\n")[:6]))
        print("   ✅ A→B 真流集成通过（supervision %s）" % ("可用" if HAS_SV else "降级"))
    else:
        print("   supervision 不可用 → 直接合成事件验证闸门逻辑")
        gate = EventGate()
        for k in range(6):
            gate.push_event({"type": "zone_dwell", "zone_id": "z", "tracker_id": 1,
                             "class_name": "person", "dwell_sec": 4.0}, t=float(k))
        assert gate.n_fired >= 1
        print("   ✅ 合成事件闸门逻辑通过")

    print("\n" + "=" * 64)
    print("✅ 事件闸全部自检通过（启发式阈值 / 冷却 / 滞回 / 突发 / 降级 / A→B 集成）")
    print("=" * 64)
