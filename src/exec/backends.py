"""
exec/backends.py —— 输入后端抽象（可插拔）

参考 neeetman/windows-input-mcp 与 computer-use-mcp 对 SendInput 的封装，补齐
pyautogui 单独使用时踩不到的几个真实问题：

1. **UIPI 权限隔离**（最致命，也是"按键无效"的常见根因）
   Windows 的 UIPI(User Interface Privilege Isolation) 会阻止 **medium IL 进程
   向 high IL 进程** 发送 SendInput。以管理员权限启动的程序（游戏、部分启动器）
   会**完全忽略**非提升进程发来的输入。
   → 本模块提供 `integrity_level()` 检测自身 IL，并在输入疑似被吞时给出明确诊断，
     而不是让上层以为是"坐标算错了"。

2. **虚拟屏幕坐标归一化**（多显示器 / 负原点）
   SendInput 的 MOUSEEVENTF_ABSOLUTE 要求坐标归一化到**整个虚拟屏幕**
   （含负坐标的副屏），不是主屏分辨率。直接按主屏 1920x1080 归一化，
   在副屏或扩展屏上会点到错误位置。

3. **DPI 感知**
   进程未声明 DPI aware 时，Windows 会做坐标虚拟化（GetWindowRect 返回缩放后坐标）。
   需要先 SetProcessDPIAware / PER_MONITOR_AWARE_V2，否则截图坐标与 SendInput 坐标
   对不上（这正是 dino 真机 region 量偏的潜在原因之一）。

4. **Unicode vs 扫描码**
   - 打中文/任意字符：KEYEVENTF_UNICODE（绕过键盘布局）
   - 打游戏键（WASD/空格/方向）：用虚拟键码 VK_*，游戏读扫描码而非字符

后端：SendInputBackend（Windows 原生 ctypes，零第三方依赖）→ 失败降级 PyAutoGUIBackend。
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from ctypes import wintypes
from typing import Optional, Tuple

IS_WINDOWS = (platform.system() == "Windows")

# ---- 常量 ----
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

# 虚拟键码（常用）
VK = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D,
    "shift": 0x10, "ctrl": 0x11, "alt": 0x12, "pause": 0x13,
    "capslock": 0x14, "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, "pageup": 0x21, "pagedown": 0x22, "end": 0x23,
    "home": 0x24, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "printscreen": 0x2C, "insert": 0x2D, "delete": 0x2E,
    "win": 0x5B, "f1": 0x70,
}
VK.update({f"f{i}": 0x6F + i for i in range(1, 25)})   # f1..f24


def _vk(key: str) -> int:
    """键名 → 虚拟键码。单字符用 ASCII 大写，其余查表。"""
    k = (key or "").lower()
    if k in VK:
        return VK[k]
    if len(k) == 1:
        return ord(k.upper())
    return VK.get(k, 0)


# ---- 结构体 ----
if IS_WINDOWS:
    ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR))

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = (("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", ULONG_PTR))

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = (("uMsg", wintypes.DWORD),
                    ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD))

    class _INPUTUNION(ctypes.Union):
        _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))
else:  # 非 Windows：不定义结构体，SendInputBackend 直接不可用
    INPUT = None


class InputBackend:
    """输入后端接口。所有方法返回 bool（是否成功）。"""

    name = "base"

    def move(self, x: int, y: int) -> bool:
        raise NotImplementedError

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        raise NotImplementedError

    def mouse_down(self, button: str = "left") -> bool:
        raise NotImplementedError

    def mouse_up(self, button: str = "left") -> bool:
        raise NotImplementedError

    def scroll(self, clicks: int) -> bool:
        raise NotImplementedError

    def key_down(self, key: str) -> bool:
        raise NotImplementedError

    def key_up(self, key: str) -> bool:
        raise NotImplementedError

    def press(self, key: str) -> bool:
        return self.key_down(key) and self.key_up(key)

    def type_unicode(self, text: str) -> bool:
        """输入任意字符（含中文）。默认退化为逐字符 press。"""
        return all(self.press(ch) for ch in text)

    def position(self) -> Tuple[int, int]:
        return (0, 0)

    def screen_size(self) -> Tuple[int, int]:
        return (1920, 1080)


# ---------------------------------------------------------------------------
# Windows 原生后端：SendInput
# ---------------------------------------------------------------------------
class SendInputBackend(InputBackend):
    """Windows user32.SendInput 原生封装（ctypes，零第三方依赖）。"""

    name = "sendinput"

    def __init__(self, dpi_aware: bool = True):
        if not IS_WINDOWS:
            raise RuntimeError("SendInputBackend 仅支持 Windows")
        self.user32 = ctypes.windll.user32
        if dpi_aware:
            self._set_dpi_aware()

    # -- DPI / 屏幕信息 --
    def _set_dpi_aware(self):
        """声明 DPI 感知，避免坐标虚拟化导致截图坐标与输入坐标不一致。"""
        try:
            # PER_MONITOR_AWARE_V2 = -4（Win10 1703+）；失败退回老 API
            self.user32.SetProcessDpiAwarenessContext(-4)
        except Exception:
            try:
                self.user32.SetProcessDPIAware()
            except Exception:
                pass

    def _virtual_screen(self) -> Tuple[int, int, int, int]:
        """虚拟屏幕 (x, y, w, h)，含负原点的副屏。"""
        u = self.user32
        vx = u.GetSystemMetrics(76)    # SM_XVIRTUALSCREEN
        vy = u.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN
        vw = u.GetSystemMetrics(78)    # SM_CXVIRTUALSCREEN
        vh = u.GetSystemMetrics(79)    # SM_CYVIRTUALSCREEN
        return vx, vy, vw, vh

    def screen_size(self) -> Tuple[int, int]:
        _, _, vw, vh = self._virtual_screen()
        return (vw, vh)

    def _norm(self, x: int, y: int) -> Tuple[int, int]:
        """屏幕坐标 → SendInput 绝对坐标（归一化到虚拟屏幕 0..65535）。"""
        vx, vy, vw, vh = self._virtual_screen()
        nx = int((x - vx) * 65535 / max(1, vw - 1))
        ny = int((y - vy) * 65535 / max(1, vh - 1))
        return nx, ny

    def _send(self, *inputs) -> bool:
        n = len(inputs)
        arr = (INPUT * n)(*inputs)
        sent = self.user32.SendInput(n, ctypes.byref(arr), ctypes.sizeof(INPUT))
        return sent == n

    def _mouse_input(self, flags: int, x: Optional[int] = None,
                     y: Optional[int] = None, data: int = 0) -> INPUT:
        inp = INPUT()
        inp.type = INPUT_MOUSE
        if x is not None and y is not None:
            nx, ny = self._norm(x, y)
            inp.mi.dx, inp.mi.dy = nx, ny
            flags |= MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE
        inp.mi.dwFlags = flags
        inp.mi.mouseData = data
        return inp

    def _key_input(self, vk: int, up: bool = False, unicode: bool = False,
                   scan: int = 0) -> INPUT:
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.ki.wVk = 0 if unicode else vk
        inp.ki.wScan = vk if unicode else scan
        inp.ki.dwFlags = KEYEVENTF_KEYUP if up else 0
        if unicode:
            inp.ki.dwFlags |= KEYEVENTF_UNICODE
        return inp

    # -- 动作 --
    def move(self, x: int, y: int) -> bool:
        return self._send(self._mouse_input(0, x, y))

    def mouse_down(self, button: str = "left") -> bool:
        flags = {"left": MOUSEEVENTF_LEFTDOWN, "right": MOUSEEVENTF_RIGHTDOWN,
                 "middle": MOUSEEVENTF_MIDDLEDOWN}.get(button, MOUSEEVENTF_LEFTDOWN)
        return self._send(self._mouse_input(flags))

    def mouse_up(self, button: str = "left") -> bool:
        flags = {"left": MOUSEEVENTF_LEFTUP, "right": MOUSEEVENTF_RIGHTUP,
                 "middle": MOUSEEVENTF_MIDDLEUP}.get(button, MOUSEEVENTF_LEFTUP)
        return self._send(self._mouse_input(flags))

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        if not self.move(x, y):
            return False
        ok = True
        for _ in range(max(1, clicks)):
            ok = self.mouse_down(button) and self.mouse_up(button) and ok
        return ok

    def scroll(self, clicks: int) -> bool:
        # 一格 = 120（WHEEL_DELTA），正负表示方向
        return self._send(self._mouse_input(MOUSEEVENTF_WHEEL, data=int(clicks * 120)))

    def key_down(self, key: str) -> bool:
        return self._send(self._key_input(_vk(key)))

    def key_up(self, key: str) -> bool:
        return self._send(self._key_input(_vk(key), up=True))

    def type_unicode(self, text: str) -> bool:
        """KEYEVENTF_UNICODE：绕过键盘布局，能打中文/任意字符。"""
        ok = True
        for ch in text:
            down = self._key_input(ord(ch), unicode=True)
            up = self._key_input(ord(ch), up=True, unicode=True)
            ok = self._send(down, up) and ok
        return ok

    def position(self) -> Tuple[int, int]:
        pt = wintypes.POINT()
        if self.user32.GetCursorPos(ctypes.byref(pt)):
            return (pt.x, pt.y)
        return (0, 0)

    # -- 诊断：完整性级别 --
    def integrity_level(self) -> str:
        """返回自身进程完整性级别：low / medium / high / system / unknown。

        high IL 的目标（管理员启动的 Chrome/游戏）**不会**接收 medium IL 进程的
        SendInput —— 这是"按键发不进去"最容易被误判为坐标 bug 的根因。
        """
        try:
            import subprocess
            out = subprocess.run(
                ["whoami", "/groups"], capture_output=True, text=True,
                shell=True, timeout=5).stdout.lower()
            if "s-1-16-16384" in out or "system" in out and "mandatory" in out:
                return "system"
            if "s-1-16-12288" in out:
                return "high"
            if "s-1-16-8192" in out:
                return "medium"
            if "s-1-16-4096" in out:
                return "low"
        except Exception:
            pass
        return "unknown"


# ---------------------------------------------------------------------------
# pyautogui 后端（跨平台兜底）
# ---------------------------------------------------------------------------
class PyAutoGUIBackend(InputBackend):
    """pyautogui 后端：跨平台，缺 SendInput 或降级时使用。"""

    name = "pyautogui"

    def __init__(self, failsafe: bool = True):
        import pyautogui
        pyautogui.FAILSAFE = failsafe
        self._pg = pyautogui

    def move(self, x: int, y: int) -> bool:
        self._pg.moveTo(x, y)
        return True

    def mouse_down(self, button: str = "left") -> bool:
        self._pg.mouseDown(button=button)
        return True

    def mouse_up(self, button: str = "left") -> bool:
        self._pg.mouseUp(button=button)
        return True

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        self._pg.click(x, y, button=button, clicks=clicks)
        return True

    def scroll(self, clicks: int) -> bool:
        self._pg.scroll(clicks)
        return True

    def key_down(self, key: str) -> bool:
        self._pg.keyDown(key)
        return True

    def key_up(self, key: str) -> bool:
        self._pg.keyUp(key)
        return True

    def type_unicode(self, text: str) -> bool:
        # pyautogui.typewrite 不支持非 ASCII，用剪贴板粘贴兜底
        try:
            import pyperclip
            pyperclip.copy(text)
            self._pg.hotkey("ctrl", "v")
            return True
        except Exception:
            return all(self.press(ch) for ch in text)

    def position(self) -> Tuple[int, int]:
        p = self._pg.position()
        return (int(p.x), int(p.y))

    def screen_size(self) -> Tuple[int, int]:
        s = self._pg.size()
        return (int(s.width), int(s.height))


# ---------------------------------------------------------------------------
# 工厂：优先 SendInput，失败降级 pyautogui
# ---------------------------------------------------------------------------
def build_backend(prefer: str = "auto", failsafe: bool = True) -> InputBackend:
    """prefer: auto / sendinput / pyautogui。auto = Windows 优先原生，失败降级。"""
    order = []
    if prefer == "pyautogui":
        order = ["pyautogui", "sendinput"]
    else:
        order = ["sendinput", "pyautogui"]

    last_err = None
    for name in order:
        try:
            if name == "sendinput":
                if not IS_WINDOWS:
                    continue
                return SendInputBackend()
            if name == "pyautogui":
                return PyAutoGUIBackend(failsafe=failsafe)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"无可用输入后端（最后错误：{last_err}）")
