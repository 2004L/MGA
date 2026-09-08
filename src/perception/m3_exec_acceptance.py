"""
m3_exec_acceptance.py —— M3 执行工具集验收（沙箱可跑，不碰真实桌面）
============================================================================

目标：证明「执行工具齐全」这件事在代码层与沙箱层都成立，而非只停留在设计。

验收维度（每条都给 pass/fail）：
  1. 动作路由齐全：DesktopAdapter.ACTIONS 里每个动作都能从 act() 一路路由到
     真实输入后端（MockBackend 记录到调用），无 stub、无断链。
  2. 注册表同源：ToolRegistry 暴露的工具名 ⊇ 动作集 + 几何工具，保证 System2
     function calling「看到的能力」与「实际能执行的能力」一致。
  3. dry_run 安全：dry_run=True 时任何动作都不触达后端（MockBackend 调用为空），
     证明「预览/试跑」不会真动鼠标键盘。
  4. 几何接线：window_rect/client_rect/foreground_window/is_minimized 已暴露且
     非 Windows 下安全降级（返回 None/""/False，不崩）。
  5. 坐标换算：to_screen(rel) 用 client_rect 正确换算成屏幕绝对坐标（focus→rect→换算→输入）。
  6. 中文输入：type 动作经 type_unicode 把中文原样送达后端。

延迟验收门（需真 Windows 桌面，不在本沙箱判定，诚实标注）：
  · 真机点击/输入注入成功率 >95%（SendInput 在目标窗口实际生效，含 UIPI/DPI 边界）。
  · 几何还原召回：真实窗口 rect 与 GetWindowRect 一致率（多屏/最小化/管理员窗口）。
  · 拟人化轨迹落点误差（jitter 末端收敛后坐标偏差）。
"""

from __future__ import annotations

import os
import sys

# ---- 路径 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PERCEPTION = _HERE
for _p in (_ROOT, _PERCEPTION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exec.backends import InputBackend                         # noqa: E402
from exec.computer_use import ComputerUse, SafetyGuard, ExecPolicy  # noqa: E402
from agent.desktop_adapter import DesktopAdapter               # noqa: E402


# ---------------------------------------------------------------------------
# Mock 后端：记录每一次真实注入调用（不碰系统）
# ---------------------------------------------------------------------------
class MockBackend(InputBackend):
    name = "mock"

    def __init__(self):
        self.calls = []

    def move(self, x, y):
        self.calls.append(("move", x, y)); return True

    def click(self, x, y, button="left", clicks=1):
        self.calls.append(("click", x, y, button, clicks)); return True

    def mouse_down(self, button="left"):
        self.calls.append(("down", button)); return True

    def mouse_up(self, button="left"):
        self.calls.append(("up", button)); return True

    def scroll(self, clicks):
        self.calls.append(("scroll", clicks)); return True

    def key_down(self, key):
        self.calls.append(("kd", key)); return True

    def key_up(self, key):
        self.calls.append(("ku", key)); return True

    def type_unicode(self, text):
        self.calls.append(("type", text)); return True

    def position(self):
        return (0, 0)

    def screen_size(self):
        return (1920, 1080)


def _new_cu(dry_run: bool, backend=None) -> ComputerUse:
    pol = ExecPolicy(dry_run=dry_run)
    return ComputerUse(SafetyGuard.from_policy(policy=pol), backend=backend or MockBackend())


# ---------------------------------------------------------------------------
# 验收主流程
# ---------------------------------------------------------------------------
_FAILS = []


def _check(name: str, cond: bool, detail: str = ""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)


def main():
    print("=" * 64)
    print("M3 执行工具集验收")
    print("=" * 64)

    # ---- 1. 动作路由齐全 ----
    print("\n[1] 动作路由齐全（act → 真实后端调用）")
    da = DesktopAdapter(dry_run=False, backend=MockBackend(), enabled=True)
    da.enable()
    # (动作命令, 期望后端记录的关键字)
    routing = [
        ("click 530 320", "click"),
        ("double_click 530 320", "click"),     # clicks=2 经 click
        ("right_click 530 320", "click"),      # button=right 经 click
        ("move 530 320", "move"),
        ("type 你好世界", "type"),
        ("drag 100 100 400 300", "down"),      # drag 必含 mouse_down
        ("scroll 3", "scroll"),
        ("press enter", "kd"),
        ("hotkey ctrl s", "kd"),
    ]
    for cmd, expect in routing:
        da.cu.backend.calls.clear()
        da.act(cmd)
        got = [c[0] for c in da.cu.backend.calls]
        _check(f"route: {cmd}", expect in got,
               f"backend.calls={got}")
    # focus 不产生后端调用（走 wm），只验证不崩、路由存在
    try:
        da.cu.backend.calls.clear()
        da.act("focus 记事本")
        _check("route: focus 记事本（走 wm，无后端调用）", True)
    except Exception as e:  # noqa
        _check("route: focus 记事本", False, f"异常={type(e).__name__}: {e}")

    # ---- 2. 注册表同源 ----
    print("\n[2] 注册表同源（System2 看到的能力 = 实际能力）")
    cu = _new_cu(False)
    names = set(cu.registry.names())
    required = {
        "click", "double_click", "type_text", "press_key", "hotkey", "drag",
        "scroll", "move", "screenshot", "list_windows", "focus_window",
        "screen_size", "cursor_position", "right_click", "window_rect",
        "client_rect", "foreground_window", "is_minimized",
    }
    missing = required - names
    _check("registry ⊇ 动作集+几何工具", not missing,
           f"缺={sorted(missing)}" if missing else f"共 {len(names)} 个工具")

    # ---- 3. dry_run 安全 ----
    print("\n[3] dry_run 安全（预览/试跑不真注入）")
    mock = MockBackend()
    cu_dry = ComputerUse(SafetyGuard.from_policy(ExecPolicy(dry_run=True)),
                         backend=mock)
    cu_dry.click(530, 320)
    cu_dry.type("你好 hello")
    cu_dry.drag(100, 100, 400, 300)
    cu_dry.hotkey("ctrl", "s")
    _check("dry_run 下后端调用为空", mock.calls == [],
           f"意外调用={mock.calls}")

    # ---- 4. 几何接线 ----
    print("\n[4] 几何工具接线（真 Windows 可用；找不到窗口安全降级）")
    cu_g = _new_cu(False)
    for m in ("window_rect", "client_rect", "foreground_window", "is_window_minimized"):
        _check(f"方法存在: {m}", hasattr(cu_g, m))
    _no = "__no_such_window_xyz__"
    _check("window_rect 找不到→None", cu_g.window_rect(_no) is None)
    _check("client_rect 找不到→None", cu_g.client_rect(_no) is None)
    _check("foreground_window 返回 str", isinstance(cu_g.foreground_window(), str))
    _check("is_window_minimized 找不到→False", cu_g.is_window_minimized(_no) is False)

    # ---- 5. 坐标换算 ----
    print("\n[5] 坐标换算（focus → rect → 换算 → 输入）")
    da2 = DesktopAdapter(enabled=True, backend=MockBackend())
    da2.enable()
    da2.cu.client_rect = lambda title: (100, 200, 500, 400)   # 客户区屏幕绝对坐标
    sx, sy = da2.to_screen(10, 20, "记事本")
    _check("to_screen 相对→绝对", (sx, sy) == (110, 220), f"得=({sx},{sy})")
    # 无标题/无 focus_title → 原样返回
    da2.focus_title = ""
    sx2, sy2 = da2.to_screen(10, 20)
    _check("to_screen 无目标→原样", (sx2, sy2) == (10, 20), f"得=({sx2},{sy2})")
    # locate 真正返回窗口客户区矩形（消除 M3 TODO 占位）
    _check("locate 返回真实矩形", da2.locate("记事本") == (100, 200, 500, 400))

    # ---- 6. 中文输入 ----
    print("\n[6] 中文输入路径（type_unicode 原样送达）")
    da3 = DesktopAdapter(dry_run=False, backend=MockBackend(), enabled=True)
    da3.enable()
    da3.cu.backend.calls.clear()
    da3.act("type 你好世界")
    types = [c for c in da3.cu.backend.calls if c[0] == "type"]
    # 拟人化逐字输入：每个字符一次 type_unicode，拼回应等于原文
    _check("中文经 type_unicode 逐字送达且完整",
           types and "".join(c[1] for c in types) == "你好世界",
           f"calls={types}")

    # ---- 验收结论 ----
    print("\n" + "=" * 64)
    if _FAILS:
        print(f"M3 验收：FAIL（{len(_FAILS)} 项未过：{_FAILS}）")
        print("=" * 64)
        return 1
    print("M3 执行工具集验收：✅ 沙箱层全过（动作齐全/非stub/dry_run安全/几何接线/坐标换算/中文输入）")
    print("延迟验收门（需真 Windows 桌面，本沙箱不判定，诚实标注）：")
    print("  · 真机注入成功率 >95%（含 UIPI/DPI/多屏边界）")
    print("  · 窗口几何还原召回率（真实 rect 一致性）")
    print("  · 拟人化轨迹落点误差")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
