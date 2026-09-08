"""
desktop_adapter.py —— 桌面适配器：把"桌面 Computer Use"接到通用智能体
============================================================================

复用：
  · GameAdapter 基类接口（locate/grab/perceive/act/tick）—
    智能体核心 orchestrator/system1/system2 一行不用改。
  · 执行层 src/exec/computer_use.py 的 ComputerUse（已真接通：键鼠/窗口/拟人化/
    审计/急停 + SafetyGuard 物理闸）。**不重造执行层**。
  · 感知层 src/perception/desktop_perceive.py 的三级降级管线。

本期只做"能跑通的接口骨架"（沙箱可验，不碰真实桌面）：
  · locate/grab/perceive 接真实模块；
  · act 把紧凑命令（"click 530 320" / "type 你好" / ...）路由给 ComputerUse；
  · 桌面操作默认关（enabled=False）：未人工开启前 act() 直接拒绝，符合硬约束。
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PERCEPTION = os.path.join(_ROOT, "perception")
for _p in (_ROOT, _HERE, _PERCEPTION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from .game_adapter import GameAdapter
from .types import Scene

# 感知 + 执行（exec 是包，必须 exec.computer_use；perception 非包，顶层 import）
from desktop_perceive import perceive_pipeline, DesktopScene   # noqa: E402
from exec.computer_use import ComputerUse, SafetyGuard, ExecPolicy   # noqa: E402


class DesktopAdapter(GameAdapter):
    """桌面 Computer Use 适配器（Phase 3）。"""

    name = "desktop"
    # 暴露给 S2 function calling 的动作集（非高危；删除/发送由 IntentGate 挡在更外层）
    ACTIONS = ("click", "double_click", "right_click", "move", "type",
               "drag", "scroll", "press", "hotkey", "focus")

    def __init__(self, region=None, l2_bridge=None, enabled: bool = False,
                 dry_run: bool = True, focus_title: str = "", verbose: bool = True,
                 backend=None):
        self.region = region
        self.l2_bridge = l2_bridge        # S2 兜底 grounding（L2 多模态 LLM）
        self.enabled = enabled            # 桌面操作默认关（硬约束）
        self.verbose = verbose
        self.focus_title = focus_title
        # 执行层：复用 ComputerUse + SafetyGuard（动作级物理闸）
        # 注意：dry_run 在 SafetyGuard 上，ComputerUse 不吃该 kwarg
        # backend 可选注入（验收测试用 MockBackend；默认走真实 sendinput/pyautogui）
        pol = ExecPolicy(dry_run=dry_run, focus_title=focus_title)
        self.cu = ComputerUse(SafetyGuard.from_policy(policy=pol), backend=backend)

    # ---------------- 启用（人工 opt-in 后才接真实桌面）----------------
    def enable(self):
        self.enabled = True

    def _ensure_enabled(self):
        if not self.enabled:
            raise RuntimeError("桌面操作默认关闭：请先人工 enable() 再执行动作")

    # ---------------- 定位桌面窗口 ----------------
    def locate(self, title_contains: str = ""):
        """返回操作区域矩形 (left,top,right,bottom)。

        M3 对齐：有标题时真正取窗口**客户区**矩形（window.py 的 rect 接口），
        不再返回固定 region 占位；找不到窗口则回退到固定 region（可能为 None）。
        """
        title = title_contains or self.focus_title
        if self.region:
            return self.region
        if title:
            rect = self.cu.client_rect(title)   # 客户区屏幕绝对坐标
            if rect:
                return rect
        return self.region

    # ---------------- 坐标换算（focus → rect → 换算 → 输入）----------------
    def to_screen(self, rel_x: float, rel_y: float,
                  title: str = "") -> Tuple[float, float]:
        """把窗口内相对坐标换算成屏幕绝对坐标。

        桌面自动化基本纪律（window.py 文档强调）：先聚焦目标窗口，取客户区矩形，
        把相对坐标换算成屏幕绝对坐标后才点，避免点到错误位置（尤其多窗口/扩展屏）。
        找不到窗口 / 未指定标题 / 非 Windows → 原样返回（安全降级，不报错）。
        """
        title = title or self.focus_title
        if not title:
            return rel_x, rel_y
        rect = self.cu.client_rect(title)
        if not rect:
            return rel_x, rel_y
        l, t, _r, _b = rect
        return l + rel_x, t + rel_y

    # ---------------- 截一帧 ----------------
    def grab(self) -> np.ndarray:
        from PIL import Image
        path = self.cu.screenshot()
        return np.array(Image.open(path).convert("RGB"))

    # ---------------- 感知（三级降级）----------------
    def perceive(self, frame: np.ndarray = None, goal: str = "") -> Scene:
        if frame is None:
            frame = self.grab()
        dscene: DesktopScene = perceive_pipeline(
            frame, goal=goal, l2_bridge=self.l2_bridge)
        # 转成通用 Scene（智能体核心只读 scene.elements/frame/meta）
        return Scene(frame=frame, elements=dscene.elements,
                     meta={"active_backend": dscene.active_backend,
                           "confidence": dscene.confidence})

    # ---------------- 执行（紧凑命令 → ComputerUse）----------------
    def act(self, action: str) -> None:
        self._ensure_enabled()
        parts = action.split()
        if not parts:
            return
        op = parts[0].lower()
        if op == "click" and len(parts) >= 3:
            self.cu.click(float(parts[1]), float(parts[2]))
        elif op == "double_click" and len(parts) >= 3:
            self.cu.double_click(float(parts[1]), float(parts[2]))
        elif op == "right_click" and len(parts) >= 3:
            self.cu.right_click(float(parts[1]), float(parts[2]))
        elif op == "move" and len(parts) >= 3:
            self.cu.move(float(parts[1]), float(parts[2]))
        elif op == "type" and len(parts) >= 2:
            self.cu.type(" ".join(parts[1:]))
        elif op == "drag" and len(parts) >= 5:
            self.cu.drag(*(float(p) for p in parts[1:5]))
        elif op == "scroll" and len(parts) >= 2:
            self.cu.scroll(int(parts[1]))
        elif op == "press" and len(parts) >= 2:
            self.cu.press(parts[1])
        elif op == "hotkey" and len(parts) >= 2:
            self.cu.hotkey(*parts[1:])
        elif op == "focus" and len(parts) >= 2:
            self.cu.focus_window(" ".join(parts[1:]))
        else:
            raise ValueError(f"未知/参数不足的动作: {action}")

    def tick(self, dt: float) -> None:
        return None  # 真机无需推进世界


if __name__ == "__main__":
    # 沙箱自检：enabled 默认关 → act 拒绝；enable + dry_run → 路由到 ComputerUse(DRY)
    a = DesktopAdapter(dry_run=True)
    try:
        a.act("click 530 320")
        print("FAIL: 未启用却执行了")
    except RuntimeError as e:
        print(f"[桌面默认关] OK 拒绝: {e}")
    a.enable()
    a.act("click 530 320")     # dry_run → 只打印 [CU][DRY]
    a.act("type 你好 hello")
    print("桌面适配骨架自检通过：默认关 + 启用后路由执行层（DRY）")
