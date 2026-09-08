"""
desktop_agent_selftest.py —— 桌面双系统闭环沙箱自检（不碰真实桌面）
============================================================================
用 MockBackend + 注入 planner + 假 perceive，验证：
  T1 正常填表（S1 模板）→ 跑通、步骤记录、active_backend 透传
  T2 删除意图（运行时 gate）→ 整任务 BLOCK、零执行
  T3 支付意图（运行时 gate）→ 标 need_confirm、零执行
  T4 S2 反思纠错 → 失败步触发 reflect、纠正后跑通
  T5 S2 正常规划（开放目标）→ s2_calls 统计、跑通
  T6 编译期 BLOCK（模板含高危）→ build_task 直接抛 ValueError
全部 Exit 0 即双系统闭环接线正确。
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PERCEPTION = os.path.join(_ROOT, "perception")
_SAFETY = os.path.join(_ROOT, "safety")
for _p in (_ROOT, _HERE, _PERCEPTION, _SAFETY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.desktop_adapter import DesktopAdapter
from agent.desktop_agent import DesktopAgent, DesktopBrain
from agent.desktop_tasks import Step
from agent.types import Scene
from safety.intent_gate import IntentGate


# ----------------------------- Mock 后端 ----------------------------- #
class MockBackend:
    """记录每次真实注入调用（对齐 ComputerUse 实际调用的后端接口）。

    注意：ComputerUse._run 会吞掉后端异常并返回 False，所以「执行失败→S2 反思」
    只能在 DesktopAgent.act 层捕获——T4 用 monkeypatch adapter.act 来模拟失败。
    """

    def __init__(self):
        self.name = "MockBackend"
        self.calls = []

    def _rec(self, name, *a):
        self.calls.append((name, a))

    def click(self, x, y, **kw):
        self._rec("click", x, y)

    def right_click(self, x, y, **kw): self._rec("right_click", x, y)
    def double_click(self, x, y, **kw): self._rec("double_click", x, y)
    def move(self, x, y, **kw): self._rec("move", x, y)
    def drag(self, x1, y1, x2, y2, **kw): self._rec("drag", x1, y1, x2, y2)
    def scroll(self, n, **kw): self._rec("scroll", n)
    def press(self, k, **kw): self._rec("press", k)
    def key_down(self, k, **kw): self._rec("kd", k)
    def key_up(self, k, **kw): self._rec("ku", k)
    def hotkey(self, *ks, **kw): self._rec("hotkey", *ks)
    def type_unicode(self, text, **kw):
        self._rec("type", text)
        return True
    def type(self, text, **kw):
        self._rec("type", text)
        return True
    def mouse_down(self, **kw): self._rec("down", None)
    def mouse_up(self, **kw): self._rec("up", None)
    def position(self, **kw): return (0, 0)
    def screen_size(self, **kw): return (1707, 960)
    def screenshot(self, **kw): return "mock.png"


class StubBrain:
    """只提供 reflect（T4 用），其余不依赖真实 LLM。"""
    def reflect(self, failed: Step, elements, goal):
        return Step("click 530 320", f"{failed.note}（S2 反思重试）", risk=failed.risk)


# ----------------------------- 工具 ----------------------------- #
def _dummy_scene(backend="screenparser"):
    return Scene(frame=np.zeros((100, 100, 3), np.uint8),
                 elements=[SimpleNamespace(label="Button", conf=0.5)],
                 meta={"active_backend": backend, "confidence": 0.5})


def _new_agent():
    backend = MockBackend()
    # dry_run=False 才会真正路由到后端（MockBackend 是 hermetic 的，不真动键鼠，
    # 用来验证「执行链路真的接通」；DRY 不注入的语义已由 M3 验收覆盖）
    a = DesktopAdapter(dry_run=False, backend=backend)
    a.enable()  # 沙箱显式开启（尊重默认关）
    # 中和真实窗口操作，保持测试 hermetic
    a.cu.focus_window = lambda t: 123
    a.cu.wm.focus_by_title = lambda t: 123
    a.grab = lambda: np.zeros((100, 100, 3), np.uint8)
    a.perceive = lambda goal="": _dummy_scene()
    return a, backend


_pass = 0
_fail = 0


def _check(name, cond, extra=""):
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  [PASS] {name} {extra}")
    else:
        _fail += 1
        print(f"  [FAIL] {name} {extra}")


# ----------------------------- 用例 ----------------------------- #
def test_t1_formfill_s1():
    print("\n[T1] 正常填表（S1 模板快路径）")
    a, backend = _new_agent()
    agent = DesktopAgent(a, intent_gate=IntentGate(), dry_run=True)
    res = agent.run_goal("填报销单", template="form_fill",
                         spec={"window_title": "报销单",
                               "fields": [{"label": "姓名", "value": "张三", "x": 200, "y": 120},
                                          {"label": "金额", "value": "88.5"}]})
    print(res.report())
    _check("状态=done", res.status == "done", f"→ {res.status}")
    _check("S1 模板路径", res.plan_source == "S1-template")
    _check("active_backend 透传", res.active_backend == "screenparser")
    _check("步骤全部执行", len(res.steps_done) == len(res.steps),
           f"done={len(res.steps_done)}/{len(res.steps)}")
    _check("未触发拦截", agent.stats.blocked == 0)
    _check("未触发确认", agent.stats.need_confirm == 0)
    _check("真实注入调用已记录", len(backend.calls) > 0,
           f"calls={[c[0] for c in backend.calls]}")


def test_t2_delete_blocked():
    print("\n[T2] 删除意图（运行时外层 gate → BLOCK）")
    a, backend = _new_agent()
    planner = lambda scene, goal: [Step("click 999 999", "删除桌面临时文件")]
    agent = DesktopAgent(a, intent_gate=IntentGate(), planner=planner, dry_run=True)
    res = agent.run_goal("清理桌面")
    _check("状态=blocked", res.status == "blocked", f"→ {res.status}")
    _check("零执行", len(res.steps_done) == 0)
    _check("拦截计数+1", agent.stats.blocked == 1)
    _check("无注入调用", len(backend.calls) == 0,
           f"calls={[c[0] for c in backend.calls]}")


def test_t3_payment_confirm():
    print("\n[T3] 支付意图（运行时 gate → need_confirm，不执行）")
    a, backend = _new_agent()
    planner = lambda scene, goal: [
        Step("type 100", "支付 100 元"),
        Step("click 1 1", "点击确认支付"),
    ]
    agent = DesktopAgent(a, intent_gate=IntentGate(), planner=planner, dry_run=True)
    res = agent.run_goal("付款", human_confirm=False)
    _check("状态=need_confirm", res.status == "need_confirm", f"→ {res.status}")
    _check("有 pending_id", bool(res.pending_id))
    _check("零执行（支付不自动做）", len(res.steps_done) == 0)
    _check("确认计数+1", agent.stats.need_confirm == 1)
    _check("无注入调用", len(backend.calls) == 0,
           f"calls={[c[0] for c in backend.calls]}")


def test_t4_s2_reflect():
    print("\n[T4] S2 反思纠错（失败步触发 reflect）")
    a, backend = _new_agent()
    planner = lambda scene, goal: [Step("click 999 999", "点击提交")]
    brain = StubBrain()
    # 模拟「执行失败」：ComputerUse._run 会吞后端异常，所以失败要在 adapter.act 层抛
    orig_act = a.act
    def _failing_act(action):
        if str(action).startswith("click 999"):
            raise RuntimeError("模拟点击失败（目标不可达）")
        return orig_act(action)
    a.act = _failing_act
    agent = DesktopAgent(a, intent_gate=IntentGate(), planner=planner, brain=brain, dry_run=True)
    res = agent.run_goal("提交表单")
    _check("状态=done", res.status == "done", f"→ {res.status}")
    _check("S2 反思计数+1", agent.stats.s2_reflect == 1)
    _check("纠正后步骤已执行", any(c[0] == "click" and c[1] == (530, 320)
                                    for c in backend.calls),
           f"calls={[c[0] for c in backend.calls]}")
    _check("未触发拦截/确认", agent.stats.blocked == 0 and agent.stats.need_confirm == 0)


def test_t5_s2_plan():
    print("\n[T5] S2 开放目标规划（正常）→ s2 统计")
    a, backend = _new_agent()
    planner = lambda scene, goal: [
        Step("click 1 1", "点击用户名框"),
        Step("type hello", "输入用户名"),
    ]
    agent = DesktopAgent(a, intent_gate=IntentGate(), planner=planner, dry_run=True)
    res = agent.run_goal("登录系统")
    _check("状态=done", res.status == "done", f"→ {res.status}")
    _check("S2 规划路径", res.plan_source == "S2")
    _check("s2_calls=1", agent.stats.s2_calls == 1)
    _check("两步均执行", len(res.steps_done) == 2)


def test_t6_compile_block():
    print("\n[T6] 编译期 BLOCK（模板含高危 → build_task 直接抛）")
    a, backend = _new_agent()
    agent = DesktopAgent(a, intent_gate=IntentGate(), dry_run=True)
    # FormFillTemplate 本身不含高危；这里用一个会触发 gate 的自定义 planner 走模板分支不易，
    # 故直接验证：含删除 note 的步骤经 gate 在编译期即被发现（等价 build_task._screen 行为）
    from agent.desktop_tasks import TaskTemplate
    tpl = TaskTemplate()
    try:
        tpl._screen([Step("click 1 1", "删除全部数据")])
        _check("编译期应拦截", False, "未抛 ValueError")
    except ValueError as e:
        _check("编译期 BLOCK 抛 ValueError", True, f"→ {e}")


if __name__ == "__main__":
    test_t1_formfill_s1()
    test_t2_delete_blocked()
    test_t3_payment_confirm()
    test_t4_s2_reflect()
    test_t5_s2_plan()
    test_t6_compile_block()
    print(f"\n=== 桌面双系统闭环自检：PASS={_pass} FAIL={_fail} ===")
    raise SystemExit(1 if _fail else 0)
