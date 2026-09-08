"""
gui_agent.py — 大模型驱动的 GUI 桌面访问（Computer Use Agent）
=============================================================
用户要的「computer use 中的 GUI 桌面访问能力」到底指什么：

    不是"截个图丢给大模型让它猜坐标"，而是三件事串起来：
      ① 看得见窗口   —— WindowManager 枚举真实窗口(标题/矩形)，能聚焦
      ② 看得见元素   —— ScreenParser(YOLO11-L, 55 类 UI, 深度学习) + OCR + UIA
                        给出**带编号的结构化元素清单**(深度学习检测，不是颜色阈值)
      ③ 会动手       —— ToolRegistry → SendInput 原生注入（含审计/策略闸/拟人化）

    大模型在这条链里只干它擅长的：**看着元素清单推理"该点哪个"**。
    坐标由 ScreenParser 的框给出，模型不需要猜像素——这正是
    MGA 架构里「毫秒级框 UI 元素 → 结构化 bbox 注入 LLM 上下文 → LLM 只推理」的落地。

    对照 CV 方案：颜色阈值分不清任务栏和游戏地面（已翻车）；
    ScreenParser 是深度学习检测器，输出的是语义类别(Button/Text/Icon…)，跨主题稳。

依赖全部懒加载 + 缺失降级，遵循"满血运行"：
    ultralytics 没装 → 元素级感知降级为空，靠 Vision LLM 兜底
    UIA 不可用(非 Windows) → 跳过，窗口级仍可用
    密钥没配 → 决策降级为"人工指令模式"(仍可执行预设动作)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# --- 感知层（懒加载，缺失即降级） ---
try:
    from perception.detector import (Element, ScreenParserBackend, OCRBackend,
                                     UIALocator, TriLevelLocator)
    _HAS_DETECTOR = True
except Exception:
    _HAS_DETECTOR = False

try:
    from perception.vision_locate import VisionLocator, _load_dotenv, _cfg_runtime
    _HAS_VISION = True
except Exception:
    _HAS_VISION = False

# --- 执行层 ---
try:
    from exec.tools import ToolRegistry, DEFAULT_AUDIT
    _HAS_TOOLS = True
except Exception:
    _HAS_TOOLS = False


# ===========================================================================
# 观察（一次"看屏幕"的结果）
# ===========================================================================
@dataclass
class GUIObservation:
    frame: Optional[np.ndarray] = None          # 截图 (H,W,3) RGB
    windows: List[dict] = field(default_factory=list)   # 窗口清单
    elements: List[Any] = field(default_factory=list)   # Element 列表(深度学习检出)
    element_source: str = "none"                # screenparser / uia / vision / none
    vision_note: str = ""                       # 大模型对画面的描述(可选)
    t_shot: float = 0.0
    t_detect: float = 0.0

    def context(self, max_elements: int = 60) -> str:
        """把观察压缩成给大模型的上下文（省 token：只列前 N 个元素）。"""
        lines: List[str] = []
        if self.windows:
            lines.append("【窗口】")
            for w in self.windows[:12]:
                title = (w.get("title") or "")[:48]
                r = w.get("rect")
                rect = f" rect={r}" if r else ""
                lines.append(f"  - hwnd={w.get('hwnd')} 「{title}」{rect}")
        if self.elements:
            lines.append(f"【界面元素】(来源={self.element_source}，共 {len(self.elements)} 个，"
                         f"列出前 {max_elements} 个)")
            for i, e in enumerate(self.elements[:max_elements]):
                lines.append("  " + (e.to_context(i) if hasattr(e, "to_context")
                                     else f"{i}. {e}"))
        else:
            lines.append(f"【界面元素】无（来源={self.element_source}）")
        if self.vision_note:
            lines.append(f"【画面理解】{self.vision_note}")
        return "\n".join(lines)


# ===========================================================================
# GUI Computer Use Agent
# ===========================================================================
class GUIComputerUse:
    """大模型驱动的 GUI 桌面访问：感知 → 决策 → 执行 闭环。"""

    SYSTEM_PROMPT = """你是一个 GUI 桌面操作助手（Computer Use）。

你会拿到：当前窗口清单 + 界面上可交互元素的编号清单（含类别和坐标）+ 用户的目标。
你的任务：只输出一行 JSON，选择**下一步**动作。不要解释、不要 markdown。

可选动作：
  {"action":"click_element","idx":<元素编号>,"thought":"为什么点它"}
  {"action":"click","x":<像素>,"y":<像素>,"thought":"..."}
  {"action":"type","text":"要输入的内容","thought":"..."}
  {"action":"press","key":"space|enter|tab|down","thought":"..."}
  {"action":"hotkey","keys":["ctrl","t"],"thought":"..."}
  {"action":"scroll","clicks":-3,"thought":"..."}
  {"action":"focus_window","title":"窗口标题关键字","thought":"..."}
  {"action":"done","summary":"目标已达成，简述结果"}
  {"action":"fail","reason":"做不到，简述原因"}

规则：
- 优先用 click_element（坐标由检测模型给出，比你自己猜像素准）
- 一次只做一个动作；不确定就先 focus_window 或 screenshot 观察
- 目标已完成立刻输出 done，不要多余动作
"""

    def __init__(self,
                 goal: str = "",
                 use_screenparser: bool = True,
                 use_ocr: bool = False,
                 use_uia: bool = True,
                 use_vision: bool = False,
                 model: str = "",
                 dry_run: bool = False,
                 imgsz: int = 640,
                 audit_path: Optional[str] = None):
        self.goal = goal
        self.use_screenparser = use_screenparser and _HAS_DETECTOR
        self.use_ocr = use_ocr and _HAS_DETECTOR
        self.use_uia = use_uia and _HAS_DETECTOR
        self.use_vision = use_vision and _HAS_VISION
        self.model = model or ""
        self.dry_run = dry_run
        # 实测坑：torch CPU 在本机 imgsz>=1280 + 多线程会段错误(进程直接被杀，
        # 不是 Python 异常)。imgsz=320~640 + 单线程稳定（2s 左右/帧）。
        self.imgsz = imgsz
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        # 执行层：工具注册表（自带审计 + 策略闸）
        if _HAS_TOOLS:
            self.tools = ToolRegistry(audit_path=audit_path or DEFAULT_AUDIT)
            if dry_run:
                for t in list(self.tools._tools.values()):
                    t.enabled = False
        else:
            self.tools = None

        # 感知层：懒加载占位
        self._locator = None
        self._vision = None
        self._wm = None
        self.history: List[dict] = []

    # ---------------------------------------------------------------- 感知
    @property
    def window_manager(self):
        if self._wm is None:
            from exec.window import WindowManager
            self._wm = WindowManager()
        return self._wm

    def _ensure_locator(self):
        """三级定位器：ScreenParser(深度学习) → OCR(文字) → UIA(语义树)。"""
        if self._locator is not None or not _HAS_DETECTOR:
            return self._locator
        visual = (ScreenParserBackend(imgsz=self.imgsz) if self.use_screenparser
                  else None)
        ocr = OCRBackend() if (self.use_ocr and visual) else None
        uia = None
        if self.use_uia:
            try:
                uia = UIALocator()
                if not uia.available():
                    uia = None
            except Exception:
                uia = None
        if visual is None and uia is not None:
            # 没有视觉检测器时，用 UIA 当主通道（仍是结构化元素，不是颜色阈值）
            self._locator = uia
        elif visual is not None:
            self._locator = TriLevelLocator(visual=visual, ocr=ocr, uia=uia)
        return self._locator

    def screenshot(self) -> Optional[np.ndarray]:
        """mss 直读帧缓冲（比 pyautogui 快）。"""
        try:
            import mss
            with mss.MSS() as sct:
                shot = sct.grab(sct.monitors[0])
                arr = np.frombuffer(shot.rgb, dtype=np.uint8)
                arr = arr.reshape(shot.height, shot.width, 3)[:, :, ::-1].copy()
                return arr
        except Exception:
            pass
        try:
            import pyautogui
            return np.array(pyautogui.screenshot().convert("RGB"))
        except Exception:
            return None

    def observe(self, with_elements: bool = True) -> GUIObservation:
        """看一次屏幕：窗口清单 + 深度学习元素检测 + (可选)大模型画面理解。"""
        obs = GUIObservation()
        t0 = time.perf_counter()
        obs.frame = self.screenshot()
        obs.t_shot = time.perf_counter() - t0

        # ① 窗口级：真实的 GUI 桌面访问能力
        try:
            obs.windows = self.window_manager.list_windows(visible_only=True)
        except Exception as e:
            obs.windows = []
            print(f"  [GUI] 窗口枚举失败：{e}")

        # ② 元素级：深度学习检测（ScreenParser / UIA）
        if with_elements and obs.frame is not None:
            loc = self._ensure_locator()
            if loc is not None:
                try:
                    t1 = time.perf_counter()
                    obs.elements = loc.detect(obs.frame)
                    obs.t_detect = time.perf_counter() - t1
                    obs.element_source = ("screenparser" if self.use_screenparser
                                          else "uia")
                except Exception as e:
                    print(f"  [GUI] 元素检测失败：{e}")

        # ③ 语义级（可选）：让视觉模型描述画面，补 ScreenParser 看不见的 Canvas 内容
        if self.use_vision and obs.frame is not None:
            try:
                if self._vision is None:
                    self._vision = VisionLocator(model=self.model or None)
                # 借用 perceive 的通用图文通道，只取原始回答
                img, _ = self._vision._resize(obs.frame)
                b64 = self._vision._to_b64(img)
                obs.vision_note = self._vision_text(
                    "用一句话描述这个屏幕上正在显示什么、用户接下来该点哪里。不要输出 JSON。",
                    b64)
            except Exception as e:
                print(f"  [GUI] 视觉理解失败：{e}")
        return obs

    def _vision_text(self, prompt: str, b64: str) -> str:
        import urllib.request
        import urllib.error
        if self._vision is None:
            return ""
        payload = {"model": self._vision.model,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": prompt},
                       {"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
                   "max_tokens": 200}
        req = urllib.request.Request(
            f"{self._vision.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._vision.api_key}"})
        with urllib.request.urlopen(req, timeout=self._vision.timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        return (d["choices"][0]["message"]["content"] or "").strip()[:300]

    # ---------------------------------------------------------------- 决策
    def decide(self, obs: GUIObservation, goal: str = "") -> Dict[str, Any]:
        """大模型看着元素清单决定下一步。失败/无密钥时返回空动作。"""
        goal = goal or self.goal
        user = f"目标：{goal}\n\n当前屏幕：\n{obs.context()}\n\n只输出一行 JSON 动作。"
        try:
            raw = self._llm_chat(self.SYSTEM_PROMPT, user)
            act = self._parse_action(raw)
            if act is None:
                print(f"  [GUI] 模型输出无法解析：{raw[:150]}")
                return {"action": "fail", "reason": "模型输出无法解析"}
            return act
        except Exception as e:
            print(f"  [GUI] 决策失败：{e}")
            return {"action": "fail", "reason": str(e)[:120]}

    def _llm_chat(self, system: str, user: str) -> str:
        import urllib.request
        model = self.model or os.getenv("MGA_LLM_MODEL") or "hy3"
        base = (os.getenv("MGA_LLM_BASE_URL")
                or (_cfg_runtime().get("llm_base_url") if _HAS_VISION else "")
                or "").rstrip("/")
        key = os.getenv("MGA_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if _HAS_VISION:
            _load_dotenv()
            key = key or os.getenv("MGA_LLM_API_KEY") or ""
        if not base:
            raise RuntimeError("未配置 llm_base_url（config.json runtime.llm_base_url）")
        if not key:
            raise RuntimeError("未配置密钥（.env.local 的 MGA_LLM_API_KEY）")
        payload = {"model": model, "temperature": 0.0, "max_tokens": 300,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode("utf-8"))
        return d["choices"][0]["message"]["content"] or ""

    @staticmethod
    def _parse_action(raw: str) -> Optional[Dict[str, Any]]:
        try:
            from perception.vision_locate import _parse_json
            return _parse_json(raw)
        except Exception:
            import re
            s = raw.strip()
            a, b = s.find("{"), s.rfind("}")
            if a < 0 or b <= a:
                return None
            try:
                return json.loads(s[a:b + 1])
            except Exception:
                return None

    # ---------------------------------------------------------------- 执行
    def act(self, action: Dict[str, Any], obs: Optional[GUIObservation] = None
            ) -> Dict[str, Any]:
        """执行一个动作。click_element 会查元素表拿坐标（模型不用猜像素）。"""
        name = str(action.get("action", "")).lower()
        rec = {"action": name, "ok": False, "detail": "", "thought":
               str(action.get("thought", ""))[:160]}

        if name in ("done", "fail"):
            rec["ok"] = True
            rec["detail"] = str(action.get("summary") or action.get("reason") or "")
            return rec

        if self.dry_run:
            rec["ok"] = True
            rec["detail"] = f"[dry-run] {action}"
            return rec

        try:
            if name == "click_element":
                idx = int(action.get("idx", -1))
                els = (obs.elements if obs else []) or []
                if not (0 <= idx < len(els)):
                    rec["detail"] = f"元素编号越界 idx={idx} 共{len(els)}个"
                    return rec
                cx, cy = els[idx].center()
                rec["detail"] = f"点击元素#{idx} {els[idx].label} @({cx},{cy})"
                rec["ok"] = bool(self._call_tool("click", x=float(cx), y=float(cy)))
            elif name == "click":
                rec["ok"] = bool(self._call_tool(
                    "click", x=float(action.get("x", 0)), y=float(action.get("y", 0))))
            elif name == "type":
                rec["ok"] = bool(self._call_tool("type", text=str(action.get("text", ""))))
            elif name == "press":
                rec["ok"] = bool(self._call_tool("press", key=str(action.get("key", "space"))))
            elif name == "hotkey":
                keys = [str(k) for k in action.get("keys", [])]
                rec["ok"] = bool(self._call_tool("hotkey", keys=keys))
            elif name == "scroll":
                rec["ok"] = bool(self._call_tool("scroll", clicks=int(action.get("clicks", 0))))
            elif name == "focus_window":
                title = str(action.get("title", ""))
                hwnd = self.window_manager.focus_by_title(title)
                rec["ok"] = hwnd is not None
                rec["detail"] = f"聚焦窗口「{title}」→ {hwnd}"
            else:
                rec["detail"] = f"未知动作 {name}"
        except Exception as e:
            rec["detail"] = f"执行异常：{type(e).__name__}: {e}"
        return rec

    def _call_tool(self, name, **kwargs):
        if self.tools is None:
            print(f"  [GUI] 执行层不可用，跳过 {name}")
            return None
        res = self.tools.call(name, **kwargs)
        if not res.ok:
            print(f"  [GUI] {name} 失败：{res.error}")
        return res.ok

    # ---------------------------------------------------------------- 闭环
    def run(self, goal: str = "", max_steps: int = 12, sleep: float = 0.8,
            verbose: bool = True) -> dict:
        """感知→决策→执行 循环，直到 done/fail 或超步数。"""
        goal = goal or self.goal
        if not goal:
            return {"ok": False, "reason": "没有目标"}

        for step in range(1, max_steps + 1):
            obs = self.observe()
            if verbose:
                print(f"\n--- step {step} ---")
                print(f"  窗口 {len(obs.windows)} 个，元素 {len(obs.elements)} 个"
                      f"({obs.element_source})，截图 {obs.t_shot*1000:.0f}ms"
                      f"，检测 {obs.t_detect*1000:.0f}ms")
            action = self.decide(obs, goal)
            rec = self.act(action, obs)
            self.history.append({"step": step, "action": action, "result": rec})
            if verbose:
                print(f"  思考：{rec.get('thought','')}")
                print(f"  动作：{rec['action']}  {rec['detail']}  "
                      f"{'OK' if rec['ok'] else 'FAIL'}")
            if action.get("action") in ("done", "fail"):
                return {"ok": action.get("action") == "done",
                        "steps": step, "summary": rec["detail"]}
            time.sleep(sleep)
        return {"ok": False, "steps": max_steps, "summary": "达到最大步数未完成任务"}


# ===========================================================================
# 便捷：找 Chrome 窗口（用户步骤①「用 ScreenParser 识别 Chrome 窗口的位置」）
# 说明：窗口位置最可靠的来源不是屏幕像素，而是 OS 的窗口管理器（零误差）。
#       ScreenParser 负责窗口**内部元素**，两者配合才是完整的 GUI 桌面访问。
# ===========================================================================
def find_chrome_window(title_key: str = "chrome") -> Optional[dict]:
    """枚举真实窗口，返回 Chrome 窗口 {hwnd,title,rect}；找不到返回 None。"""
    try:
        from exec.window import WindowManager
        wm = WindowManager()
        for w in wm.list_windows(visible_only=True):
            title = (w.get("title") or "").lower()
            if title_key.lower() in title:
                hwnd = w.get("hwnd")
                try:
                    w["rect"] = wm.rect(hwnd)
                except Exception:
                    pass
                return w
    except Exception as e:
        print(f"  [GUI] 窗口枚举失败：{e}")
    return None


def locate_game_window(verbose: bool = True, do_focus: bool = True,
                       title_key: str = "chrome"
                       ) -> Optional[Tuple[int, int, int, int]]:
    """完整的 GUI 桌面访问：找 Chrome 窗口 → 最小化则还原 → 聚焦 → 取客户区屏幕矩形。

    返回 (x, y, w, h) —— 可直接喂给 mss 截屏。
    这比"在像素里猜游戏区"可靠得多：坐标来自 OS，零误差，且顺手解决了
    「窗口最小化导致 CV 怎么找都找不到」的顽疾。
    """
    try:
        from exec.window import WindowManager
    except Exception as e:
        if verbose:
            print(f"  [GUI] WindowManager 不可用：{e}")
        return None

    wm = WindowManager()
    w = None
    for cand in wm.list_windows(visible_only=True):
        if title_key.lower() in (cand.get("title") or "").lower():
            w = cand
            break
    if not w:
        if verbose:
            print(f"  [GUI] 没找到含「{title_key}」的窗口（请先打开 chrome://dino）")
        return None

    hwnd = w["hwnd"]
    title = (w.get("title") or "")[:44]

    # 最小化 → 还原 + 聚焦（focus 内部已处理 SW_RESTORE）
    if do_focus:
        if wm.is_minimized(hwnd):
            if verbose:
                print(f"  [GUI] 窗口最小化，正在还原：{title}")
            wm.focus(hwnd)
            time.sleep(0.6)             # 等还原动画结束，否则截到半透明残影
        else:
            wm.focus(hwnd)
            time.sleep(0.15)

    r = wm.client_screen_rect(hwnd)
    if not r:
        return None
    x, y, x2, y2 = r
    region = (x, y, max(1, x2 - x), max(1, y2 - y))
    if verbose:
        print(f"  [GUI] 窗口「{title}」 客户区屏幕矩形={region}")
    return region
