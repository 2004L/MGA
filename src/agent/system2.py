"""
system2.py —— System2 高层大脑：快速推理 + 情景规划 + 经验记忆写入
=================================================================
只在「关键帧」被唤醒（System1 遇到没见过的情形 / 置信度低时升级上来）。职责三件：

    1. 快速推理思考：结合当前局面与检索到的历史经验，给出动作与"时机补偿量"。
    2. 情景规划    ：不只看最近一个障碍，而是把接下来 2~3 个障碍的间距纳入考虑
                    （双障碍要不要早跳、飞鸟要不要更早蹲）。
    3. 经验持久化  ：把这次的结论**写入存储**（memory_system / sqlite），
                    并回流训练 System1 的习得残差 → 下次 System1 自己就会（习惯固化）。

LLM 接入：
    通过 perception/llm_bridge.build_llm 构造（mock / oracle / api / ultralytics）。
    **架构真相**：stats 会如实区分 n_llm（真 LLM 决策）与 n_heuristic（LLM 不可用时的
    启发式兜底）——绝不把启发式冒充成 LLM。

零依赖兜底：任何 LLM 后端缺失/超时/返回非 JSON → 自动退化为启发式，闭环不中断。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .types import Decision
from .plan import Plan, Telemetry, _clamp_knob

def _first_json(text: str):
    r"""抠出第一个完整（支持嵌套）的 JSON 对象。比简单正则更稳：能处理 knobs 等嵌套结构。

    真 LLM / reasoner 的纠错/计划 JSON 常含 {"knobs":{"reaction":...}} 这类嵌套，
    简单正则 \{[^{}]*\} 会匹配失败 → 退回启发式。这里用括号计数+字符串转义感知提取。"""
    if not text:
        return None
    s = text.find("{")
    if s < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(s, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[s:i + 1]
    return None

# 允许 LLM 输出的动作（白名单，防越权/幻觉）
_ALLOWED = ("jump", "squat", "wait")


class System2:
    """System2：LLM 大脑。低频、贵、但会学习并把经验固化。"""

    def __init__(self, memory=None, backend: str = "mock", model: Optional[str] = None,
                 verbose: bool = False):
        self.memory = memory
        self.verbose = verbose
        self.bridge = None
        self.backend_name = "heuristic"
        self.n_llm = 0
        self.n_heuristic = 0
        self.n_reasoner = 0     # 本地大脑替身（离线演示用，is_real=False，不冒充真 LLM）
        self.n_plans = 0        # 制定/修订计划的次数（宏观调控次数）
        self.n_correct = 0      # 死亡/失误时「及时纠错」次数（诊断+修正）
        try:
            from perception.llm_bridge import build_llm
            self.bridge = build_llm(backend=backend, model=model)
            self.backend_name = type(self.bridge).__name__
            print(f"  [System2] LLM 后端已挂载：{self.backend_name} (backend={backend})")
        except Exception as e:
            print(f"  [System2] LLM 不可用，降级启发式：{type(e).__name__}: {e}")
            self.bridge = None

    # ---------------- 提示词：把局面说清楚 + 情景规划 ----------------
    def _prompt(self, sit: Dict[str, Any], memory_ctx: str = "") -> str:
        return (
            "你是游戏 AI 的高层决策大脑(System2)，负责在关键帧做判断并把经验固化。\n"
            f"当前局面：障碍类型={sit.get('kind')}，水平速度={sit.get('vx', 0):.0f}px/s，"
            f"预计到达时间 ETA={sit.get('eta', 0):.3f}s，"
            f"当前触发阈值={sit.get('react', 0.22):.3f}s，"
            f"后续是否紧跟障碍={sit.get('pair', False)}。\n"
            f"历史经验：{memory_ctx or '（无，这是首次遇到该情形）'}\n"
            "请只输出一行 JSON，不要任何解释：\n"
            '{"action":"jump|squat|wait","react_delta":<-0.08~0.08 秒的时机补偿，'
            '负数=更早跳>,"reason":"一句话说明理由"}\n'
            "规则：地面障碍(cactus)用 jump；飞鸟(bird)用 squat；"
            "若后续紧跟障碍，可给负的 react_delta 让它更早起跳以覆盖两个障碍。"
        )

    # ---------------- 解析 LLM 输出（健壮：兼容代码块/前后废话） ----------------
    def _parse(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        text = raw.strip()
        obj = _first_json(text)
        if not obj:
            return None
        try:
            obj = json.loads(obj)
        except Exception:
            return None
        act = str(obj.get("action", "")).strip().lower()
        if act not in _ALLOWED:
            return None
        try:
            delta = float(obj.get("react_delta", 0.0))
        except Exception:
            delta = 0.0
        return {"action": act, "react_delta": delta,
                "reason": str(obj.get("reason", ""))[:200]}

    # ---------------- 启发式兜底（LLM 不可用时的"常识"） ----------------
    def _heuristic(self, sit: Dict[str, Any]) -> Dict[str, Any]:
        kind = sit.get("kind", "cactus")
        pair = bool(sit.get("pair", False))
        return {
            "action": "squat" if kind == "bird" else "jump",
            "react_delta": -0.03 if pair else 0.0,
            "reason": "启发式兜底(LLM不可用)：地面障碍跳、飞鸟蹲；双障碍略早跳",
        }

    # ---------------- 主入口：思考一次 ----------------
    def think(self, sit: Dict[str, Any]) -> Dict[str, Any]:
        """返回 {action, react_delta, reason, source}。"""
        mem_ctx = ""
        if self.memory is not None:
            try:
                hits = self.memory.retrieve(sit.get("key", ""), top_k=2) or []
                if hits:
                    mem_ctx = " | ".join(str(h.get("summary", ""))[:80] for h in hits)
            except Exception:
                mem_ctx = ""

        out = None
        if self.bridge is not None:
            try:
                raw = self.bridge.respond(self._prompt(sit, mem_ctx))
                out = self._parse(raw)
                if out is not None:
                    if getattr(self.bridge, "is_real", False):
                        out["source"] = "llm"
                        self.n_llm += 1
                    else:
                        out["source"] = "reasoner"
                        self.n_reasoner += 1
            except Exception as e:
                if self.verbose:
                    print(f"  [System2] LLM 调用失败，转启发式：{type(e).__name__}: {e}")
                out = None

        if out is None:
            out = self._heuristic(sit)
            out["source"] = "heuristic"
            self.n_heuristic += 1
        return out

    # ---------------- 经验持久化写入（核心职责） ----------------
    def write_experience(self, key: str, action: str, reason: str,
                         react_delta: float, tags=None) -> str:
        if self.memory is None:
            return ""
        summary = (f"情形={key} → 动作={action}，时机补偿={react_delta:+.3f}s；"
                   f"理由：{reason}")
        mid = self.memory.write(key, summary, tags=list(tags or ["game", "dino", "s2"]))
        if self.verbose and mid:
            print(f"  [System2] 经验已写入存储：{key} → {action} ({mid})")
        return mid

    # ==================================================================
    #  计划与宏观调控（System2 握方向盘：定计划 + 调 System1 的全局旋钮）
    # ==================================================================
    def _plan_prompt(self, ctx: Dict[str, Any], memory_ctx: str = "") -> str:
        cur = ctx.get("current", {})
        return (
            "你是游戏 AI 的高层大脑(System2)，负责**制定未来一段时间的计划并对底层做宏观调控**。\n"
            f"目标：{ctx.get('goal','')}\n"
            f"上一段执行遥测：{ctx.get('telemetry','')}\n"
            f"当前生效旋钮：reaction={cur.get('reaction',0.22):.3f} "
            f"airtime={cur.get('airtime',0.6):.3f} "
            f"s2_horizon={cur.get('s2_horizon',1.2):.3f}\n"
            f"历史经验：{memory_ctx or '（暂无）'}\n"
            "旋钮含义：reaction=触发阈值(越大越早跳，0.15~0.32)；"
            "airtime=空中抑制窗(越小越允许快速补跳，0.40~0.75)；"
            "s2_horizon=多早请示高层(0.70~1.80)。\n"
            "只输出一行 JSON，不要解释：\n"
            '{"steps":["未来这段时间打算怎么应对，1~3 条"],'
            '"reaction":0.22,"airtime":0.60,"s2_horizon":1.20,'
            '"horizon":8.0,"reason":"为什么这样调控"}'
        )

    def _parse_plan(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        obj = _first_json(raw)
        if not obj:
            return None
        try:
            obj = json.loads(obj)
        except Exception:
            return None
        steps = obj.get("steps")
        if not isinstance(steps, list):
            return None
        try:
            knobs = {
                "reaction": _clamp_knob("reaction", float(obj.get("reaction", 0.22))),
                "airtime": _clamp_knob("airtime", float(obj.get("airtime", 0.60))),
                "s2_horizon": _clamp_knob("s2_horizon", float(obj.get("s2_horizon", 1.20))),
            }
            horizon = float(obj.get("horizon", 8.0))
        except Exception:
            return None
        return {"steps": [str(s)[:120] for s in steps[:3]],
                "knobs": knobs, "horizon": max(2.0, min(30.0, horizon)),
                "reason": str(obj.get("reason", ""))[:200]}

    def _heuristic_plan(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 不可用时的"宏观调控常识"（启发式，如实标记 source）。"""
        cur = ctx.get("current", {})
        react = float(cur.get("reaction", 0.22))
        airtime = float(cur.get("airtime", 0.60))
        horizon_k = float(cur.get("s2_horizon", 1.20))
        deaths = int(ctx.get("deaths", 0))
        speed = float(ctx.get("avg_speed", 0.0))

        reason = []
        if deaths > 0:
            # 实测主失效模式是"跳太早落地即撞"→ 死过就稍晚一点跳
            react = _clamp_knob("reaction", react - 0.01)
            airtime = _clamp_knob("airtime", airtime - 0.05)   # 缩短抑制窗，允许更快补跳
            reason.append(f"窗口内死亡{deaths}次→略晚跳/允许更快补跳")
        if speed > 600:
            horizon_k = _clamp_knob("s2_horizon", horizon_k + 0.10)
            reason.append(f"速度{speed:.0f}偏快→更早请示高层")
        if not reason:
            reason.append("执行平稳→维持当前旋钮")

        return {
            "steps": ["维持节奏：地面障碍跳、飞鸟蹲",
                      "遇到没见过的情形交给高层学习",
                      "速度变快时提前请示高层"],
            "knobs": {"reaction": react, "airtime": airtime, "s2_horizon": horizon_k},
            "horizon": 8.0,
            "reason": "启发式宏观调控：" + "；".join(reason),
        }

    def make_plan(self, ctx: Dict[str, Any], t: float = 0.0) -> Plan:
        """制定/修订计划：LLM 优先，失败则启发式；并把计划写入记忆（持久化）。"""
        mem_ctx = ""
        if self.memory is not None:
            try:
                hits = self.memory.retrieve("计划", top_k=2) or []
                if hits:
                    mem_ctx = " | ".join(str(h.get("summary", ""))[:100] for h in hits)
            except Exception:
                mem_ctx = ""

        out = None
        if self.bridge is not None:
            try:
                raw = self.bridge.respond(self._plan_prompt(ctx, mem_ctx))
                out = self._parse_plan(raw)
                if out:
                    out["source"] = "llm" if getattr(self.bridge, "is_real", False) else "reasoner"
                    if out["source"] == "llm":
                        self.n_llm += 1
                    else:
                        self.n_reasoner += 1
            except Exception as e:
                if self.verbose:
                    print(f"  [System2] 计划 LLM 调用失败，转启发式：{type(e).__name__}: {e}")
                out = None
        if out is None:
            out = self._heuristic_plan(ctx)
            out["source"] = "heuristic"
            self.n_heuristic += 1

        plan = Plan(goal=ctx.get("goal", "活下去并尽可能久地通关"),
                    horizon=out["horizon"], created_t=t, steps=out["steps"],
                    knobs=out["knobs"], reason=out["reason"], source=out["source"])
        self.n_plans += 1
        self.write_plan(plan)
        return plan

    def write_plan(self, plan: Plan) -> str:
        """把计划（含宏观调控旋钮）持久化写入记忆，便于下局/下阶段复用。"""
        if self.memory is None:
            return ""
        kn = " ".join(f"{k}={v:.3f}" for k, v in plan.knobs.items())
        return self.memory.write(
            "计划", f"[{plan.source}] 步骤：{' → '.join(plan.steps)}；旋钮 {kn}；"
                    f"理由：{plan.reason}", tags=["plan", "宏观调控"])

    # ==================================================================
    #  及时纠错（System2 的元认知）：死亡/失误瞬间反思并修正
    # ==================================================================
    def _correct_prompt(self, ctx: Dict[str, Any], memory_ctx: str = "") -> str:
        return (
            "你是游戏 AI 的**反思/纠错模块**（System2 的元认知）。智能体刚刚**撞死/失误**了，\n"
            "请基于下面信息诊断「哪一步想错了」，并给出纠正方案（类比真实大脑的反思学习）。\n\n"
            f"【死亡时情形】\n"
            f"情形键={ctx.get('key','')}\n"
            f"撞死前最近动作={ctx.get('last_action','wait')}（ETA={ctx.get('eta',0):.3f}s，"
            f"速度={ctx.get('vx',0):.0f}px/s）\n"
            f"导致该局的最后一条经验={ctx.get('last_experience','（无）')}\n"
            f"当前宏观调控旋钮：reaction={ctx.get('reaction',0.22):.3f} "
            f"airtime={ctx.get('airtime',0.6):.3f}\n\n"
            f"【近期决策轨迹】\n{ctx.get('trace','(无)')}\n\n"
            f"历史经验：{memory_ctx or '（暂无）'}\n\n"
            "只输出一行 JSON，不要解释：\n"
            '{"diagnosis":"一句话诊断错在哪","fix":"challenge|rewrite|adjust_knobs|new_plan",'
            '"key":"要修正的经验情形键(可空)","action":"jump|squat|wait(纠正后动作，空则不改)",'
            '"react_delta":<-0.08~0.08,"knobs":{"reaction":0.20,"airtime":0.55}'
            '(仅 adjust_knobs 时用),"rewrite":true/false,"reason":"纠正理由"}\n'
            "规则：\n"
            " - 若经验把动作带偏(该跳时蹲)→ fix=rewrite 并给正确 action；\n"
            " - 若只是时机偏晚→ fix=adjust_knobs 或 react_delta 负值；\n"
            " - 若经验完全不可信→ fix=challenge（降权）。"
        )

    def _parse_correct(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        obj = _first_json(raw)
        if not obj:
            return None
        try:
            obj = json.loads(obj)
        except Exception:
            return None
        fix = str(obj.get("fix", "")).strip().lower()
        if fix not in ("challenge", "rewrite", "adjust_knobs", "new_plan"):
            return None
        knobs = obj.get("knobs") or {}
        if not isinstance(knobs, dict):
            knobs = {}
        safe_knobs = {}
        for k in ("reaction", "airtime", "s2_horizon"):
            if k in knobs:
                try:
                    safe_knobs[k] = float(knobs[k])
                except Exception:
                    pass
        return {
            "diagnosis": str(obj.get("diagnosis", ""))[:200],
            "fix": fix,
            "key": str(obj.get("key", ""))[:80],
            "action": str(obj.get("action", ""))[:12],
            "react_delta": float(obj.get("react_delta", 0.0) or 0.0),
            "knobs": safe_knobs,
            "rewrite": bool(obj.get("rewrite", False)),
            "reason": str(obj.get("reason", ""))[:200],
        }

    def _heuristic_correct(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 不可用时的「纠错常识」（启发式，如实标记 source），逻辑与 reasoner 同构。"""
        kind = "bird" if "bird" in (ctx.get("key", "") or "") else "cactus"
        return {
            "diagnosis": f"启发式纠错：{kind} 情形触发时机偏晚或被误导经验带偏",
            "fix": "challenge",
            "key": ctx.get("key", ""),
            "action": "jump" if kind != "bird" else "squat",
            "react_delta": -0.02,
            "knobs": {"reaction": 0.20, "airtime": 0.55},
            "rewrite": True,
            "reason": "死亡后反思：降权误导经验并写入更早发现时机，略早跳兜底",
        }

    def correct_error(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """死亡/失误瞬间调用：System2 反思诊断 + 给出修正方案（不在此处落库，由编排器应用）。

        返回 {diagnosis, fix, key, action, react_delta, knobs, rewrite, reason, source}。
        """
        self.n_correct += 1
        mem_ctx = ""
        if self.memory is not None:
            try:
                hits = self.memory.retrieve(ctx.get("key", ""), top_k=2) or []
                if hits:
                    mem_ctx = " | ".join(str(h.get("summary", ""))[:80] for h in hits)
            except Exception:
                mem_ctx = ""

        out = None
        if self.bridge is not None:
            try:
                raw = self.bridge.respond(self._correct_prompt(ctx, mem_ctx))
                out = self._parse_correct(raw)
                if out is not None:
                    out["source"] = "llm" if getattr(self.bridge, "is_real", False) else "reasoner"
            except Exception as e:
                if self.verbose:
                    print(f"  [System2] 纠错 LLM 调用失败，转启发式：{type(e).__name__}: {e}")
                out = None
        if out is None:
            out = self._heuristic_correct(ctx)
            out["source"] = "heuristic"
        return out
