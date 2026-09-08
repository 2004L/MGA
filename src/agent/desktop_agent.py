"""
desktop_agent.py —— 桌面双系统闭环（PCD loop）
============================================================================
把 M1–M5 造好的零件真正接成「能跑的桌面智能体」，而不是一堆孤立模块：

    perceive(三级降级) → plan(S1模板 / S2 LLM) → gate(IntentGate 外层语义闸)
                      → act(DesktopAdapter→ComputerUse) → verify(失败则 S2 反思)

双系统设计（对应架构图）：
  · System1（快）：模板直出步骤序列，不调 LLM，直接执行（99% 的常规任务）。
  · System2（慢）：LLM 规划开放目标 + 失败反思纠错；只在「没模板 / 步骤失败」时醒。
  · 经验记忆：每步写 ExperienceMemory；失败 → S2 反思 → 纠正后重试。
  · 安全门禁：IntentGate 在「最外层」——delete/send 直接 BLOCK 整任务，
              payment 卡人工确认点；过闸后才进 ComputerUse 的动作级物理闸。

诚实边界：
  · 开放目标（无模板）规划需要真 LLM；无 LLM 时诚实返回 no-plan，不假装能规划。
  · 桌面操作默认关：adapter.enabled=False 时 act() 直接拒绝（硬约束保留）。
  · LLM 不可用 / 超时 / 非 JSON → DesktopBrain 退化为启发式或返回 None，闭环不中断。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PERCEPTION = os.path.join(_ROOT, "perception")
_SAFETY = os.path.join(_ROOT, "safety")
for _p in (_ROOT, _HERE, _PERCEPTION, _SAFETY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .types import Scene
from .desktop_tasks import Step, build_task, TEMPLATES
from safety.intent_gate import IntentGate


# ------------------------------------------------------------------ #
#  结果结构
# ------------------------------------------------------------------ #
@dataclass
class DesktopResult:
    goal: str = ""
    status: str = "init"          # done | blocked | need_confirm | no-plan | failed
    plan_source: str = ""         # S1-template | S2
    active_backend: str = "unknown"
    steps: List[Step] = field(default_factory=list)
    steps_done: List[Step] = field(default_factory=list)
    blocked_step: int = -1
    blocked_note: str = ""
    need_confirm_step: int = -1
    pending_id: Optional[str] = None
    trace: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"[DesktopAgent] 目标={self.goal!r} 状态={self.status} "
                 f"规划源={self.plan_source} 感知={self.active_backend}"]
        if self.trace:
            lines.append("  轨迹:")
            for t in self.trace:
                lines.append(f"    - {t}")
        lines.append(f"  完成 {len(self.steps_done)}/{len(self.steps)} 步")
        return "\n".join(lines)


@dataclass
class DesktopStats:
    s1_calls: int = 0        # 模板快路径（不调 LLM）
    s2_calls: int = 0        # LLM 规划
    s2_reflect: int = 0      # 失败反思纠正
    executed: int = 0
    blocked: int = 0
    need_confirm: int = 0

    def report(self) -> str:
        return (f"S1(模板)={self.s1_calls}  S2(LLM规划)={self.s2_calls}  "
                f"S2(反思)={self.s2_reflect}  执行={self.executed}  "
                f"拦截={self.blocked}  待确认={self.need_confirm}")


# ------------------------------------------------------------------ #
#  System2 大脑：桌面规划 + 反思（复用 build_llm，启发式兜底）
# ------------------------------------------------------------------ #
def _first_json(text: str):
    """抠出第一个完整 JSON 对象（兼容嵌套 steps）。"""
    if not text:
        return None
    s = text.find("[")
    if s < 0:
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
            elif c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return text[s:i + 1]
    return None


class DesktopBrain:
    """System2（慢）：开放目标规划 + 失败反思。"""

    def __init__(self, memory=None, backend: str = "mock", model: Optional[str] = None,
                 verbose: bool = False):
        self.memory = memory
        self.verbose = verbose
        self.bridge = None
        self.backend_name = "heuristic"
        try:
            from perception.llm_bridge import build_llm
            self.bridge = build_llm(backend=backend, model=model)
            self.backend_name = type(self.bridge).__name__
            if self.verbose:
                print(f"  [DesktopBrain] LLM 后端：{self.backend_name} (backend={backend})")
        except Exception as e:
            print(f"  [DesktopBrain] LLM 不可用，降级启发式：{type(e).__name__}: {e}")
            self.bridge = None

    # ---------------- 规划 ----------------
    def _prompt(self, goal: str, elements: List[Any]) -> str:
        labels = ", ".join(
            f"{getattr(e, 'label', '?')}({getattr(e, 'conf', 0):.2f})"
            for e in (elements or [])[:12]) or "（无 detected 元素）"
        return (
            "你是桌面自动化智能体(System2)。把用户目标拆成具体的操作步骤。\n"
            f"目标：{goal}\n"
            f"当前检测到界面元素：{labels}\n"
            "规则：只输出 JSON 数组，每个元素 {action: 紧凑命令(如 'click 530 320'), "
            "note: 人类可读且含语义的动作说明}。\n"
            "禁止生成删除/发送/支付类操作（会被安全门禁拦截）。\n"
            '示例：[{"action":"click 530 320","note":"点击用户名输入框"},'
            '{"action":"type 张三","note":"填入姓名"}]'
        )

    def _parse_steps(self, raw: str) -> Optional[List[Step]]:
        obj = _first_json(raw)
        if not obj:
            return None
        try:
            data = json.loads(obj)
        except Exception:
            return None
        arr = data if isinstance(data, list) else data.get("steps")
        if not isinstance(arr, list):
            return None
        out = []
        for it in arr:
            if not isinstance(it, dict):
                continue
            act = str(it.get("action", "")).strip()
            note = str(it.get("note", act)).strip()
            if not act:
                continue
            out.append(Step(act, note, risk="write"))
        return out or None

    def plan(self, goal: str, elements: List[Any], ctx: dict = None) -> Optional[List[Step]]:
        """返回步骤序列；LLM 不可用 / 解析失败 → 启发式 / None（诚实不造假）。"""
        if self.bridge is not None:
            try:
                raw = self.bridge.respond(self._prompt(goal, elements))
                steps = self._parse_steps(raw)
                if steps:
                    return steps
            except Exception as e:
                if self.verbose:
                    print(f"  [DesktopBrain] 规划 LLM 失败，转启发式：{type(e).__name__}: {e}")
        # 启发式：按关键词选已验证模板（需要 spec，自由目标无法补全 → 返回 None）
        return self._heuristic_template(goal)

    def _heuristic_template(self, goal: str) -> Optional[List[Step]]:
        g = (goal or "").lower()
        try:
            if "填表" in g or "表单" in g:
                return None  # 需 spec，交上层报错提示
            if "搬运" in g or "复制" in g:
                return None
            if "整理" in g or "移动" in g:
                return None
        except Exception:
            pass
        return None

    # ---------------- 失败反思 ----------------
    def reflect(self, failed: Step, elements: List[Any], goal: str) -> Optional[Step]:
        """步骤执行失败后反思纠正。LLM 不可用 → 启发式（同一动作重试，note 标记重试）。"""
        if self.bridge is not None:
            try:
                prompt = (f"桌面步骤执行失败，请纠正。目标={goal}\n"
                          f"失败步骤：{failed.action} | {failed.note}\n"
                          f"界面元素：{', '.join(getattr(e,'label','?') for e in (elements or [])[:8])}\n"
                          '只输出一行 JSON：{"action":"修正后的紧凑命令","note":"修正说明"}')
                raw = self.bridge.respond(prompt)
                steps = self._parse_steps(raw)
                if steps:
                    return steps[0]
            except Exception:
                pass
        # 启发式兜底：原样重试（保留语义，交给动作级物理闸再判一次）
        return Step(failed.action, failed.note + "（S2 反思重试）", risk=failed.risk)


# ------------------------------------------------------------------ #
#  编排器：桌面双系统 PCD 闭环
# ------------------------------------------------------------------ #
class DesktopAgent:
    """桌面 Computer Use 双系统智能体（perceive→plan→gate→act→verify）。"""

    def __init__(self, adapter, intent_gate: IntentGate = None, brain: DesktopBrain = None,
                 memory=None, planner=None, dry_run: bool = True, verbose: bool = True):
        self.adapter = adapter
        self.gate = intent_gate or IntentGate()
        self.brain = brain
        self.memory = memory
        self.planner = planner          # 注入点（测试用）；默认用 brain.plan
        self.verbose = verbose
        self.stats = DesktopStats()

    # ---------------- 主入口：执行一个目标 ----------------
    def run_goal(self, goal: str, template: str = None, spec: dict = None,
                 max_steps: int = 40, human_confirm: bool = True) -> DesktopResult:
        res = DesktopResult(goal=goal)
        if not goal:
            res.status = "no-plan"
            return res

        # 1) perceive（三级降级；active_backend 透传，绝不谎报）
        try:
            scene = self.adapter.perceive(goal=goal)
        except Exception as e:
            res.status = "no-plan"
            res.trace.append(f"感知失败：{type(e).__name__}: {e}")
            return res
        res.active_backend = scene.meta.get("active_backend", "unknown")
        res.steps = []  # 先占位，下面填充

        # 2) plan：S1 模板快路径 / S2 LLM
        try:
            if template:
                if template not in TEMPLATES:
                    res.status = "no-plan"
                    res.trace.append(f"未知模板 {template}")
                    return res
                steps = build_task(template, spec or {})
                self.stats.s1_calls += 1
                res.plan_source = "S1-template"
            else:
                planned = (self.planner(scene, goal) if self.planner
                           else (self.brain.plan(goal, scene.elements) if self.brain else None))
                if not planned:
                    res.status = "no-plan"
                    res.trace.append("无 LLM / 无匹配模板 → 无法规划开放目标（诚实返回）")
                    return res
                steps = planned
                self.stats.s2_calls += 1
                res.plan_source = "S2"
        except ValueError as e:
            # 模板编译期 IntentGate BLOCK（含 delete/send）
            res.status = "blocked"
            res.blocked_note = str(e)
            res.trace.append(f"BLOCK(编译期): {e}")
            self.stats.blocked += 1
            return res

        res.steps = steps

        # 3) 执行循环：每步过最外层语义闸
        for i, step in enumerate(steps):
            if i >= max_steps:
                res.trace.append(f"超过 max_steps={max_steps}，暂停")
                break
            d = self.gate.gate(step.note)
            if d.blocked():
                res.status = "blocked"
                res.blocked_step = i
                res.blocked_note = d.reason
                res.trace.append(f"步骤{i+1} 被 IntentGate 拦截: {step.note} → {d.reason}")
                self.stats.blocked += 1
                return res
            if d.need_confirm():
                res.pending_id = d.pending_id
                res.need_confirm_step = i
                res.trace.append(f"步骤{i+1} 需人工确认(支付): {step.note}")
                self.stats.need_confirm += 1
                if not human_confirm:
                    res.status = "need_confirm"
                    return res
                # 生产路径：真人确认后才继续
                if not self.gate.confirm(d.pending_id):
                    res.status = "need_confirm"
                    return res

            # 执行
            try:
                self.adapter.act(step.action)
                res.steps_done.append(step)
                self.stats.executed += 1
                if self.memory is not None:
                    self.memory.write("desktop-step",
                                      f"目标={goal} 步骤={step.action} 备注={step.note}",
                                      tags=["desktop", "step"])
            except Exception as e:
                res.trace.append(f"步骤{i+1} 执行失败：{type(e).__name__}: {e}")
                corr = self.brain.reflect(step, scene.elements, goal) if self.brain else None
                if corr is not None:
                    res.trace.append(f"S2 反思纠正 → {corr.action} ({corr.note})")
                    try:
                        self.adapter.act(corr.action)
                        res.steps_done.append(corr)
                        self.stats.s2_reflect += 1
                        self.stats.executed += 1
                    except Exception as e2:
                        res.trace.append(f"反思后重试仍失败：{type(e2).__name__}: {e2}")
                        res.status = "failed"
                        return res
                else:
                    res.status = "failed"
                    return res

        res.status = "done"
        return res


if __name__ == "__main__":
    print("desktop_agent.py 模块加载正常（直接运行请用 *_selftest.py）")
