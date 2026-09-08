"""
exec/computer_use.py —— MGA ⑥ 执行层：Computer Use 编排

分层（参考 computer-use-mcp / cc-haha 的多工具抽象后重整）：

    ComputerUse（本文件·编排层）
        ├─ 策略闸 SafetyGuard   ：放不放行（config.exec，默认全开 + 急停）
        ├─ 拟人化              ：随机延迟 / 抖动轨迹（可关，关=确定性执行）
        ├─ 工具注册表 ToolRegistry：统一调用入口 + JSONL 审计（经验层数据源）
        ├─ 窗口管理 WindowManager：先聚焦再输入（消除"按键去哪了"）
        └─ 输入后端 InputBackend  ：SendInputBackend(原生) → PyAutoGUIBackend(兜底)

大脑架构保持不变：执行层只负责"做"，什么时候做、做什么由
System1(预判帧,µs) / System2(深度推理) 决定。执行层向上只暴露
`registry.schema()` 给 System2 做 function calling，保证能力同源。
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List

from .backends import build_backend, InputBackend, IS_WINDOWS
from .tools import ToolRegistry, Tool, DEFAULT_AUDIT
from .window import WindowManager

# 项目根（src/exec/computer_use.py → src/exec → src → 根）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 危险动作文本（默认**不启用**；策略 block_dangerous_text=true 时才拦截）
DANGEROUS_PATTERNS = (
    "shutdown", "restart", "reboot", "format", "mkfs",
    "rm -rf", "del /f", "rd /s", "regedit", "taskkill",
    "powershell", "cmd /c", "sudo", "chmod -r", "kill -9",
)


# ---------------------------------------------------------------------------
# 执行策略：所有约束都是开关，默认全放开
# ---------------------------------------------------------------------------
@dataclass
class ExecPolicy:
    dry_run: bool = False
    require_confirm: bool = False
    allowed_region: Optional[Tuple[int, int, int, int]] = None
    block_dangerous_text: bool = False
    block_dangerous_action: List[str] = field(default_factory=list)

    # 随机化/拟人化
    human_like: bool = True
    random_delay: Tuple[float, float] = (0.02, 0.12)
    mouse_steps: Tuple[int, int] = (3, 8)
    mouse_jitter: float = 3.0
    type_jitter: float = 0.03

    # 后端与窗口
    backend: str = "auto"          # auto / sendinput / pyautogui
    focus_before_input: bool = True  # 输入前先聚焦目标窗口
    focus_title: str = ""            # 目标窗口标题子串；空=用当前前台窗口

    # 急停与审计
    failsafe: bool = True
    stop_file: bool = True
    audit: bool = True
    audit_path: str = DEFAULT_AUDIT

    @classmethod
    def from_config(cls, path: Optional[str] = None) -> "ExecPolicy":
        path = path or os.path.join(_PROJECT_ROOT, "config.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f).get("exec", {})
        except Exception:
            return cls()
        pol = cls()
        for k, v in (cfg or {}).items():
            if not hasattr(pol, k):
                continue
            if k in ("random_delay", "mouse_steps") and isinstance(v, (list, tuple)):
                setattr(pol, k, tuple(v))
            elif k == "allowed_region" and isinstance(v, (list, tuple)) and len(v) == 4:
                setattr(pol, k, tuple(v))
            else:
                setattr(pol, k, v)
        return pol


class SafetyGuard:
    """策略化护栏：默认全放开，只留急停。"""

    def __init__(self, dry_run: bool = False, require_confirm: bool = False,
                 confirm_cb: Optional[Callable[[str], bool]] = None,
                 allowed_region: Optional[Tuple[int, int, int, int]] = None,
                 policy: Optional[ExecPolicy] = None):
        self.policy = policy or ExecPolicy()
        self.dry_run = dry_run
        self.require_confirm = require_confirm
        self.confirm_cb = confirm_cb
        if allowed_region is not None:
            self.policy.allowed_region = allowed_region
        self._stopped = False
        self.blocked_log: list = []

    @classmethod
    def from_policy(cls, policy: Optional[ExecPolicy] = None,
                    dry_run: Optional[bool] = None,
                    require_confirm: Optional[bool] = None) -> "SafetyGuard":
        pol = policy or ExecPolicy.from_config()
        g = cls(policy=pol)
        g.dry_run = dry_run if dry_run is not None else pol.dry_run
        g.require_confirm = (require_confirm if require_confirm is not None
                             else pol.require_confirm)
        return g

    # ---- 急停 ----
    def emergency_stop(self):
        self._stopped = True

    def clear_stop(self):
        self._stopped = False

    def _stop_file_triggered(self) -> bool:
        if not self.policy.stop_file:
            return False
        try:
            if os.path.exists(os.path.join(_PROJECT_ROOT, "STOP")):
                self._stopped = True
                return True
        except Exception:
            pass
        return False

    # ---- 闸门 ----
    def check(self, action: str, x: Optional[float] = None,
              y: Optional[float] = None, text: Optional[str] = None
              ) -> Tuple[bool, str]:
        if self._stopped:
            return False, "EMERGENCY_STOP"
        if self._stop_file_triggered():
            return False, "STOP_FILE"
        if action in (self.policy.block_dangerous_action or []):
            return False, f"POLICY_BLOCKED_ACTION:{action}"
        region = self.policy.allowed_region
        if (x is not None and y is not None) and region:
            x1, y1, x2, y2 = region
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                return False, "OUT_OF_REGION"
        if self.policy.block_dangerous_text:
            low = (text or action).lower()
            for pat in DANGEROUS_PATTERNS:
                if pat in low:
                    return False, f"DANGEROUS:{pat}"
        if self.require_confirm and self.confirm_cb is not None:
            if not self.confirm_cb(f"{action} @({x},{y}) text={text}"):
                return False, "HUMAN_DENIED"
        if self.dry_run:
            return True, "DRY_RUN"
        return True, "OK"


class ComputerUse:
    """编排层：过策略 → 拟人化 → 调后端 → 记审计。"""

    def __init__(self, guard: Optional[SafetyGuard] = None,
                 screenshot_dir: str = "blobs/shots",
                 backend: Optional[InputBackend] = None,
                 audit_path: Optional[str] = None):
        self.guard = guard or SafetyGuard.from_policy()
        self.policy = self.guard.policy
        self.screenshot_dir = screenshot_dir
        os.makedirs(screenshot_dir, exist_ok=True)

        self.backend = backend or build_backend(prefer=self.policy.backend,
                                                failsafe=self.policy.failsafe)
        self.wm = WindowManager()
        # auto_audit=False：审计统一由 _run 记录（覆盖「直接调方法」和
        # 「经 registry.call」两条路径），避免重复记账导致经验层数据翻倍。
        self.registry = ToolRegistry(
            audit_path=audit_path or self.policy.audit_path,
            backend_name=self.backend.name,
            audit=self.policy.audit, auto_audit=False)
        self._shot_count = 0
        self._register_tools()

    # ---- 随机化/拟人化 ----
    def _rand_delay(self):
        lo, hi = self.policy.random_delay
        if hi > 0:
            time.sleep(random.uniform(lo, hi))

    def _move_human(self, x: float, y: float):
        if not self.policy.human_like:
            self.backend.move(int(x), int(y))
            return
        sx, sy = self.backend.position()
        lo, hi = self.policy.mouse_steps
        steps = random.randint(max(1, lo), max(1, hi))
        j = self.policy.mouse_jitter
        for i in range(1, steps + 1):
            t = i / steps
            damp = (1.0 - t) ** 0.5            # 末端收敛，保证落点准
            self.backend.move(int(sx + (x - sx) * t + random.uniform(-j, j) * damp),
                              int(sy + (y - sy) * t + random.uniform(-j, j) * damp))
        self.backend.move(int(x), int(y))

    def _ensure_focus(self) -> bool:
        """输入前聚焦目标窗口（策略可关）。失败不阻断，只提示。"""
        if not self.policy.focus_before_input or not self.policy.focus_title:
            return True
        if not IS_WINDOWS:
            return True
        hwnd = self.wm.focus_by_title(self.policy.focus_title)
        if hwnd is None:
            print(f"  [CU] 警告：未找到/无法聚焦窗口 '{self.policy.focus_title}'"
                  f"（目标若为管理员启动，UIPI 会拒绝聚焦与输入）")
            return False
        return True

    # ---- 截图 ----
    def screenshot(self, path: Optional[str] = None,
                   all_screens: bool = True) -> str:
        from PIL import ImageGrab
        if path is None:
            self._shot_count += 1
            path = os.path.join(self.screenshot_dir, f"shot_{self._shot_count}.png")
        img = ImageGrab.grab(all_screens=all_screens)
        img.save(path)
        return path

    # ---- 统一执行入口：策略闸 → 急停/DRY → 执行 → 审计 ----
    # 所有动作都过这里，保证「直接调方法」和「经 registry.call」两条路径
    # 都被审计（审计是经验层的数据源，漏记 = 经验有洞）。
    def _run(self, action: str, args: dict, fn, *, x=None, y=None,
             text=None) -> bool:
        ok, reason = self.guard.check(action, x, y, text=text)
        if not ok:
            print(f"  [CU] 拦截 {action} {args}: {reason}")
            self.guard.blocked_log.append((action, reason))
            self.registry.log_manual(action, args, ok=False, error=reason)
            return False
        self._rand_delay()
        if self.guard.dry_run:
            print(f"  [CU][DRY] {action} {args}")
            self.registry.log_manual(action, args, ok=True, error="DRY_RUN")
            return True
        self._ensure_focus()
        t0 = time.perf_counter()
        try:
            r = bool(fn())
            self.registry.log_manual(action, args, ok=r,
                                     ms=(time.perf_counter() - t0) * 1000)
            return r
        except Exception as e:
            self.registry.log_manual(action, args, ok=False,
                                     error=f"{type(e).__name__}: {e}",
                                     ms=(time.perf_counter() - t0) * 1000)
            return False

    # ---- 动作 ----
    def click(self, x, y, button: str = "left", clicks: int = 1,
              human: bool = True) -> bool:
        def _do():
            if human:
                self._move_human(x, y)
            return self.backend.click(int(x), int(y), button=button, clicks=clicks)
        args = {"x": x, "y": y, "button": button, "clicks": clicks}
        return self._run("click", args, _do, x=x, y=y)

    def double_click(self, x, y) -> bool:
        return self.click(x, y, clicks=2)

    def right_click(self, x, y) -> bool:
        return self.click(x, y, button="right")

    def move(self, x, y, human: bool = True) -> bool:
        def _do():
            if human:
                self._move_human(x, y)
            else:
                self.backend.move(int(x), int(y))
            return True
        return self._run("move", {"x": x, "y": y}, _do, x=x, y=y)

    def drag(self, x1, y1, x2, y2, human: bool = True) -> bool:
        args = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        ok2, reason2 = self.guard.check("drag", x2, y2)     # 终点也要过区域闸
        if not ok2:
            self.guard.blocked_log.append(("drag", reason2))
            self.registry.log_manual("drag", args, ok=False, error=reason2)
            return False

        def _do():
            if human:
                self._move_human(x1, y1)
                j = self.policy.mouse_jitter
                self.backend.mouse_down()
                steps = random.randint(6, 14)
                for i in range(1, steps + 1):
                    t = i / steps
                    damp = (1.0 - t) ** 0.5
                    self.backend.move(
                        int(x1 + (x2 - x1) * t + random.uniform(-j, j) * damp),
                        int(y1 + (y2 - y1) * t + random.uniform(-j, j) * damp))
                    time.sleep(0.01)
                self.backend.move(int(x2), int(y2))
            else:
                self.backend.move(int(x1), int(y1))
                self.backend.mouse_down()
                self.backend.move(int(x2), int(y2))
            self.backend.mouse_up()
            return True
        return self._run("drag", args, _do, x=x1, y=y1)

    def type(self, text: str) -> bool:
        """输入文本（走 Unicode 后端，支持中文）。"""
        def _do():
            j = self.policy.type_jitter
            if j > 0 and self.policy.human_like:
                ok_all = True
                for ch in text:              # 逐字 + 抖动，更像人
                    ok_all = self.backend.type_unicode(ch) and ok_all
                    time.sleep(random.uniform(0, j))
                return ok_all
            return self.backend.type_unicode(text)
        return self._run("type", {"text": text}, _do, text=text)

    def press(self, key: str) -> bool:
        return self._run("press", {"key": key},
                         lambda: self.backend.press(key), text=key)

    def hotkey(self, *keys) -> bool:
        def _do():
            r = all(self.backend.key_down(k) for k in keys)
            return all(self.backend.key_up(k) for k in reversed(keys)) and r
        return self._run("hotkey", {"keys": list(keys)}, _do, text=" ".join(keys))

    def scroll(self, clicks: int, x: Optional[float] = None,
               y: Optional[float] = None) -> bool:
        def _do():
            if x is not None and y is not None:
                self.backend.move(int(x), int(y))
            return self.backend.scroll(clicks)
        return self._run("scroll", {"clicks": clicks, "x": x, "y": y}, _do,
                         x=x, y=y)

    # ---- 窗口 ----
    def focus_window(self, title_contains: str) -> Optional[int]:
        return self.wm.focus_by_title(title_contains)

    def list_windows(self) -> List[dict]:
        return self.wm.list_windows()

    # ---- 窗口几何（坐标换算用，M3 补齐 window.py 的 rect 接口）----
    # 桌面自动化基本纪律：focus → 取客户区矩形 → 把相对坐标换算成屏幕绝对坐标 → 输入。
    # 这些方法把 window.py 的几何能力接到执行层，让「按标题定位 + 坐标换算」可端到端走通。
    def window_rect(self, title_contains: str) -> Optional[Tuple[int, int, int, int]]:
        """窗口整体矩形 (left, top, right, bottom)，含标题栏/边框。"""
        hwnd = self.wm.find(title_contains)
        return self.wm.rect(hwnd) if hwnd else None

    def client_rect(self, title_contains: str) -> Optional[Tuple[int, int, int, int]]:
        """窗口**客户区**屏幕绝对坐标 (left, top, right, bottom)，去掉标题栏。

        可直接喂截屏与坐标换算；找不到窗口/非 Windows 返回 None（上层应降级）。
        """
        hwnd = self.wm.find(title_contains)
        return self.wm.client_screen_rect(hwnd) if hwnd else None

    def foreground_window(self) -> str:
        """当前前台窗口标题（空串=无/非 Windows）。"""
        return self.wm.foreground_title()

    def is_window_minimized(self, title_contains: str) -> bool:
        hwnd = self.wm.find(title_contains)
        return self.wm.is_minimized(hwnd) if hwnd else False

    # ---- 工具注册（供 System2 function calling）----
    def _register_tools(self):
        p = lambda t, d, req=False: {"type": t, "desc": d, "required": req}
        self.registry.register(Tool(
            "click", "在屏幕坐标点击（left/right/middle）",
            lambda x, y, button="left", clicks=1: self.click(x, y, button, clicks),
            {"x": p("number", "横坐标", True), "y": p("number", "纵坐标", True),
             "button": p("string", "按键 left/right/middle"),
             "clicks": p("number", "点击次数")}, risk="write"))
        self.registry.register(Tool(
            "double_click", "双击",
            lambda x, y: self.double_click(x, y),
            {"x": p("number", "横坐标", True), "y": p("number", "纵坐标", True)},
            risk="write"))
        self.registry.register(Tool(
            "type_text", "输入文本（支持中文）",
            lambda text: self.type(text),
            {"text": p("string", "要输入的内容", True)}, risk="write"))
        self.registry.register(Tool(
            "press_key", "按单个键，如 enter/space/down/f5",
            lambda key: self.press(key),
            {"key": p("string", "键名", True)}, risk="write"))
        self.registry.register(Tool(
            "hotkey", "组合键，如 ctrl+c、alt+tab",
            lambda keys: self.hotkey(*keys.split("+")),
            {"keys": p("string", "用 + 连接的组合键，如 ctrl+s", True)}, risk="write"))
        self.registry.register(Tool(
            "drag", "从一点拖到另一点",
            lambda x1, y1, x2, y2: self.drag(x1, y1, x2, y2),
            {"x1": p("number", "起点x", True), "y1": p("number", "起点y", True),
             "x2": p("number", "终点x", True), "y2": p("number", "终点y", True)},
            risk="write"))
        self.registry.register(Tool(
            "scroll", "滚动滚轮（正数向上）",
            lambda clicks, x=None, y=None: self.scroll(clicks, x, y),
            {"clicks": p("number", "格数，正上负下", True),
             "x": p("number", "可选横坐标"), "y": p("number", "可选纵坐标")},
            risk="write"))
        self.registry.register(Tool(
            "move", "移动鼠标不点击",
            lambda x, y: self.move(x, y),
            {"x": p("number", "横坐标", True), "y": p("number", "纵坐标", True)},
            risk="write"))
        self.registry.register(Tool(
            "screenshot", "截图返回路径",
            lambda: self.screenshot(), {}, risk="read"))
        self.registry.register(Tool(
            "list_windows", "列出可见窗口（标题/pid）",
            lambda: self.list_windows(), {}, risk="read"))
        self.registry.register(Tool(
            "focus_window", "聚焦标题含指定文字的窗口",
            lambda title: bool(self.focus_window(title)),
            {"title": p("string", "窗口标题子串", True)}, risk="write"))
        self.registry.register(Tool(
            "right_click", "右键点击",
            lambda x, y: self.right_click(x, y),
            {"x": p("number", "横坐标", True), "y": p("number", "纵坐标", True)},
            risk="write"))
        self.registry.register(Tool(
            "window_rect", "取窗口整体矩形(left,top,right,bottom)屏幕绝对坐标",
            lambda title: self.window_rect(title),
            {"title": p("string", "窗口标题子串", True)}, risk="read"))
        self.registry.register(Tool(
            "client_rect", "取窗口客户区屏幕绝对坐标（去掉标题栏）",
            lambda title: self.client_rect(title),
            {"title": p("string", "窗口标题子串", True)}, risk="read"))
        self.registry.register(Tool(
            "foreground_window", "当前前台窗口标题",
            lambda: self.foreground_window(), {}, risk="read"))
        self.registry.register(Tool(
            "is_minimized", "窗口是否最小化（最小化时矩形是伪坐标）",
            lambda title: self.is_window_minimized(title),
            {"title": p("string", "窗口标题子串", True)}, risk="read"))
        self.registry.register(Tool(
            "screen_size", "获取屏幕尺寸",
            lambda: self.backend.screen_size(), {}, risk="read"))
        self.registry.register(Tool(
            "cursor_position", "获取当前鼠标坐标",
            lambda: self.backend.position(), {}, risk="read"))


# ---------------------------------------------------------------------------
# 自检（python -m exec.computer_use，DRY_RUN 不真动鼠标）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pol = ExecPolicy.from_config()
    print("== 执行策略（config.json exec 段）==")
    for k in ("dry_run", "require_confirm", "allowed_region", "backend",
              "focus_before_input", "human_like", "failsafe", "stop_file", "audit"):
        print(f"  {k:20s} = {getattr(pol, k)}")

    g = SafetyGuard.from_policy(dry_run=True)     # 自检强制 dry_run
    cu = ComputerUse(g, audit_path=os.path.join("blobs", "exec_audit_selftest.jsonl"))

    print(f"\n== 输入后端 ==")
    print(f"  后端 = {cu.backend.name}   屏幕 = {cu.backend.screen_size()}")
    if IS_WINDOWS and cu.backend.name == "sendinput":
        print(f"  进程完整性级别 IL = {cu.backend.integrity_level()}"
              f"（medium 进程无法向 high IL 管理员程序发输入）")

    print("\n== 工具清单（registry.schema() 可直接喂 System2）==")
    print(cu.registry.describe())
    print(f"  共 {len(cu.registry.names())} 个工具")

    print("\n== 放开验证（DRY_RUN）==")
    cu.click(530, 320)
    cu.type("你好 hello")            # Unicode 路径
    cu.drag(100, 100, 400, 300)
    cu.hotkey("ctrl", "s")

    print("\n== 急停验证 ==")
    g.emergency_stop()
    cu.click(530, 320)
    g.clear_stop()
    cu.click(530, 320)

    print("\n== 审计记录（经验层数据源）==")
    for r in cu.registry.recent(6):
        print(f"  {r['tool']:14s} ok={str(r['ok']):5s} {r['ms']:6.2f}ms "
              f"backend={r['backend']} err={r['error']}")
