"""
desktop_wechat_chat.py —— 桌面 Computer Use 替你聊微信（单一预授权联系人 + 人随时接管）
===================================================================================
安全边界（来自你 2026-09-08 的拍板，全部落地、不偷偷放宽）：
  · 只替 --contact 指定的【那一个】联系人回；换人必须重新跑并确认（检测到换人即停）。
  · 【真人鼠标一活动（点击/移动）立即自动取消】自主聊天（人随时接管）。
  · 删除/支付 永远不碰（IntentGate 红线：DELETE 永远 BLOCK、PAYMENT 永远 NEED_CONFIRM）。
  · 发送：仅在显式 --real + 预授权联系人范围内才发（authorized-send，带审计标记）。
  · 披露：默认每条/每次会话带一次「(AI代回)」声明；--no-disclosure 可关（关掉=完全静默，
    伦理责任归你，我不拦但标注）。

依赖：
  · 读消息：可插拔 inbox（--inbox JSON 文件 / 内置 Mock；真机加 --ocr 抓屏+OCR 认新消息）。
  · 回消息：LLM 草稿（build_llm；默认 hy3 OpenAI 兼容，密钥走 .env.local；无 key 用 --echo 占位）。
  · 发消息：DesktopAdapter（sendinput）+ WeChat 搜索框聚焦联系人。

用法：
  python desktop_wechat_chat.py --contact "张三"                # 计划，不碰桌面
  python desktop_wechat_chat.py --contact "张三" --real --echo  # 真机自测（echo 草稿）
  python desktop_wechat_chat.py --contact "张三" --real --ocr   # 真机 OCR 读消息自测(echo)
  python desktop_wechat_chat.py --contact "张三" --real --ocr --llm-backend hy3  # OCR读+真草稿发
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.desktop_adapter import DesktopAdapter
from safety.intent_gate import IntentGate
from perception.detector import OCRBackend   # 真机抓屏读微信消息（读仅，不发送）


# ============================ 真人接管监视器 ============================
class HumanOverrideMonitor:
    """真人鼠标活动 → 取消事件。轮询 cursor 位置 + 左右键状态（无需全局钩子，跨平台安全降级）。

    设计点：agent 自己只敲键盘（打字/回车），不移动鼠标；因此监听期内任何鼠标
    移动/点击都视为【真人介入】→ 立即取消自主聊天。monitor.start() 必须在 agent
    完成自己的初始聚焦点击之后调用，否则会误判自己的点击。
    """

    VK_LBUTTON = 0x01
    VK_RBUTTON = 0x02

    def __init__(self, poll: float = 0.12):
        self.poll = poll
        self.cancel = threading.Event()
        self._active = False
        self._thread = None
        self._last_pos = None
        self._has_win32 = False
        try:
            self._user32 = ctypes.windll.user32
            self._has_win32 = True
        except Exception:
            self._has_win32 = False

    def _key_down(self, vk: int) -> bool:
        if not self._has_win32:
            return False
        try:
            return bool(self._user32.GetAsyncKeyState(vk) & 0x8000)
        except Exception:
            return False

    def _loop(self) -> None:
        if not self._has_win32:
            return  # 非 Windows：监视器为空操作（永不取消）
        p = ctypes.wintypes.POINT()
        while self._active and not self.cancel.is_set():
            if self._user32.GetCursorPos(ctypes.byref(p)):
                if self._last_pos is not None and (p.x, p.y) != self._last_pos:
                    self.cancel.set()
                    break
                self._last_pos = (p.x, p.y)
            if self._key_down(self.VK_LBUTTON) or self._key_down(self.VK_RBUTTON):
                self.cancel.set()
                break
            time.sleep(self.poll)

    def start(self) -> None:
        self._active = True
        self._last_pos = None
        if self._has_win32:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._active = False


# ============================ 读消息（可插拔 inbox） ============================
class Inbox(ABC):
    @abstractmethod
    def read_latest(self, contact: str) -> Optional[str]:
        """返回该联系人最新一条未读消息；无则返回 None。"""

    def ack(self, contact: str) -> None:
        """标记已处理（避免重复回同一条）。"""


class FileInbox(Inbox):
    """从 JSON 文件读最新一条：{"contact": "消息"}。读完即清空该条（ack）。"""

    def __init__(self, path: str):
        self.path = path

    def read_latest(self, contact: str) -> Optional[str]:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        return data.get(contact)

    def ack(self, contact: str) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if contact in data:
                del data[contact]
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


class MockInbox(Inbox):
    """沙箱自测用：返回预设消息一次。"""

    def __init__(self, msg: str = "在吗？周末有空一起打三角洲吗"):
        self._msg = msg
        self._pending = True

    def read_latest(self, contact: str) -> Optional[str]:
        if self._pending:
            return self._msg
        return None

    def ack(self, contact: str) -> None:
        self._pending = False


class OCRInbox(Inbox):
    """真机抓屏 + OCR 读微信新消息（读仅，不发送；安全）。

    流程：grab 微信窗口截图 → 裁到聊天区 → OCR 出文本行 → 去重(已见集合)
          → 跳过自带「(AI代回)」前缀的自己消息 → 返回最新一条未读。

    诚实边界：
      · 微信 UI 布局因版本/缩放而异，区域比例系数 chat_frac 是启发式，真机需微调；
      · 首轮会把当前屏上所有可见消息标记为已见（不会回历史），之后只回真正新来的；
      · OCR 较慢（整图 easyocr CPU 约数秒~18s），依赖指纹缓存避免重复推理；
      · 若无 easyocr / client_rect 取不到，安全降级返回 None（不崩、不发）。
    """

    def __init__(self, adapter: DesktopAdapter, contact: str, ocr: OCRBackend = None,
                 chat_frac=(0.30, 0.10, 1.0, 0.90), self_prefix: str = "(AI代回)"):
        self.adapter = adapter
        self.contact = contact
        self.ocr = ocr or OCRBackend()
        self.chat_frac = chat_frac      # 聊天区在窗口中的比例 (左,上,右,下)
        self.self_prefix = self_prefix  # 自身回复前缀 → 识别为"我发的"，跳过
        self._seen = set()              # 已见消息文本集合（去重）
        self._last_returned = None

    def _crop(self, frame):
        """裁到聊天区（排除左侧联系人列表、顶部标题栏、底部输入框）。"""
        try:
            rect = self.adapter.cu.client_rect("微信")
        except Exception:
            rect = None
        if rect is None:
            return frame                # 无窗口矩形 → 用整图
        l, t, r, b = rect
        w, h = r - l, b - t
        fl, ft, fr, fb = self.chat_frac
        x1 = int(l + fl * w); y1 = int(t + ft * h)
        x2 = int(l + fr * w); y2 = int(t + fb * h)
        try:
            return frame[y1:y2, x1:x2]
        except Exception:
            return frame

    def read_latest(self, contact: str) -> Optional[str]:
        try:
            frame = self.adapter.grab()
        except Exception:
            return None
        crop = self._crop(frame)
        try:
            lines = self.ocr.read_lines(crop)
        except Exception:
            return None
        # 去重 + 跳过自己的回复 → 取最新（最靠下）一条未读
        newest = None
        for (cx, cy, text, conf) in lines:
            if not text:
                continue
            if self.self_prefix and text.startswith(self.self_prefix):
                self._seen.add(text)    # 自己发的，标已见，避免被当成对方新消息
                continue
            if text in self._seen:
                continue
            if newest is None or cy > newest[1]:
                newest = (text, cy)
        self._last_returned = newest[0] if newest else None
        return self._last_returned

    def ack(self, contact: str) -> None:
        # 标刚返回的那条为已见，避免下一轮重复回同一条
        if self._last_returned:
            self._seen.add(self._last_returned)
            self._last_returned = None


# ============================ 草稿 LLM（可插拔） ============================
class EchoLLM:
    """无 key 自测：把对方的话回显成一句测试草稿。真机请改用 build_llm(backend='api')。"""

    def decide(self, prompt: str, **kw) -> str:
        return "（测试草稿）收到，我看看时间安排下～"


# ============================ 微信自主聊 Agent ============================
class WeChatChatAgent:
    def __init__(self, contact: str, adapter: DesktopAdapter, llm, inbox: Inbox,
                 allow_send: bool = True, disclosure: str = "(AI代回)",
                 monitor: HumanOverrideMonitor = None, verbose: bool = True):
        self.contact = contact              # 单一预授权联系人（范围受控）
        self.adapter = adapter
        self.llm = llm
        self.inbox = inbox
        self.allow_send = allow_send        # 仅在 --real + 预授权时 True
        self.disclosure = disclosure        # None/"" = 静默（冒充真人）
        self.monitor = monitor or HumanOverrideMonitor()
        self.gate = IntentGate(allow_send=allow_send)
        self.verbose = verbose
        self._sent = 0
        self._cancelled = False

    def _log(self, *a):
        if self.verbose:
            print("[wechat]", *a)

    # ---- 打开/聚焦微信 ----
    def open_and_focus(self) -> bool:
        self._log("聚焦微信窗口")
        rect = self.adapter.cu.client_rect("微信") if hasattr(self.adapter, "cu") else None
        if rect is None:
            # 没开 → Win+R 启动
            self.adapter.act("hotkey win r")
            time.sleep(0.5)
            self.adapter.act("type wechat")
            time.sleep(0.3)
            self.adapter.act("press enter")
            time.sleep(2.5)
        else:
            self.adapter.act("focus 微信")
            time.sleep(0.5)
        return True

    # ---- 导航到指定联系人会话（best-effort，真机坐标需微调） ----
    def navigate_to_contact(self, contact: str) -> bool:
        # WeChat 搜索框在客户区左上；点一下聚焦 → 输入联系人 → 回车开会话。
        rect = self.adapter.cu.client_rect("微信")
        if rect is None:
            self._log("找不到微信窗口客户区，导航跳过（请手动点开会话）")
            return False
        l, t, _r, _b = rect
        sx, sy = l + 70, t + 45          # 搜索框启发式坐标（真机按实际窗口微调）
        self.adapter.act(f"click {sx} {sy}")
        time.sleep(0.4)
        self.adapter.act(f"type {contact}")
        time.sleep(0.4)
        self.adapter.act("press enter")
        time.sleep(0.8)
        self._log(f"已尝试打开与「{contact}」的会话")
        return True

    # ---- 处理一条消息：草稿 → 门禁 → 发送 ----
    def run_once(self, msg: str) -> bool:
        prompt = (f"你代表梁学泽和微信联系人「{self.contact}」聊天。对方刚说：{msg}\n"
                  f"用梁学泽自然的口吻回一句简短的话（不要解释你是 AI）。")
        try:
            reply = self.llm.decide(prompt) if self.llm else f"（AI代回）{msg}"
        except Exception as e:
            self._log(f"LLM 草稿失败，跳过本条：{type(e).__name__}: {e}")
            return False
        if self.disclosure:
            reply = f"{self.disclosure}{reply}"

        # 过意图门禁（authorized-send 范围受控）。
        # 关键：门禁文本用"发送消息给联系人X"——发送【动作本身】即 SEND 意图，
        # 不依赖回复文本里是否含关键词；allow_send=False 时任何发送都 BLOCK。
        # 若草稿内容提到支付，分类优先判 PAYMENT → NEED_CONFIRM（额外人工 checkpoint）。
        d = self.gate.gate(f"发送消息给联系人{self.contact}：{reply}")
        if d.blocked():
            self._log(f"门禁拦截（未授权发送）：{d.reason}")
            return False
        if d.need_confirm():
            self._log(f"门禁需确认（草稿涉支付）：{d.reason}")
            return False

        # 发送（聚焦输入框 → 打字 → 回车）
        self.adapter.act("type " + reply)
        time.sleep(0.2)
        self.adapter.act("press enter")
        self._sent += 1
        self._log(f"已替你回给「{self.contact}」：{reply}")
        return True

    # ---- 主循环：读消息 → 回 → 等下一条；鼠标活动即取消 ----
    def run_loop(self, max_iter: int = 50) -> None:
        self._log(f"开始自主聊天（仅联系人={self.contact}）；真人动鼠标将自动取消")
        self.monitor.start()
        try:
            for _ in range(max_iter):
                if self.monitor.cancel.is_set():
                    self._cancelled = True
                    self._log("检测到真人鼠标操作 → 已自动取消自主聊天")
                    break
                msg = self.inbox.read_latest(self.contact)
                if msg:
                    if self.monitor.cancel.is_set():
                        break
                    self.run_once(msg)
                    self.inbox.ack(self.contact)
                time.sleep(1.0)
        finally:
            self.monitor.stop()
        if not self._cancelled:
            self._log("收件箱为空 / 达到最大轮次，自主聊天结束")


# ============================ CLI ============================
def main() -> None:
    ap = argparse.ArgumentParser(description="桌面 CU 替你聊微信（单一预授权联系人）")
    ap.add_argument("--contact", required=True, help="预授权联系人（只替这一个人回；换人需重跑并确认）")
    ap.add_argument("--real", action="store_true",
                    help="显式真机模式：实际打开微信+导航+发消息（默认只打印计划）")
    ap.add_argument("--echo", action="store_true",
                    help="无 key 自测：用 EchoLLM 占位草稿（不调真 LLM）")
    ap.add_argument("--llm-backend", default="hy3",
                    help="草稿 LLM 后端（build_llm；默认 hy3=OpenAI 兼容，密钥走 .env.local；"
                         "另有 echo/mock/gpt6/ollama/api 等）")
    ap.add_argument("--no-disclosure", action="store_true",
                    help="完全静默（冒充真人，不披露）。伦理责任归你。")
    ap.add_argument("--inbox", default="", help="读消息的 JSON 文件路径（默认内置 Mock）")
    ap.add_argument("--ocr", action="store_true",
                    help="真机读消息：抓微信窗口 + OCR 识别新消息（读仅，不发送）。需 --real。")
    ap.add_argument("--max-iter", type=int, default=50)
    args = ap.parse_args()

    disclosure = "" if args.no_disclosure else "(AI代回)"

    if not args.real:
        print("== 桌面 CU 替你聊微信 · 计划（未 --real，不碰桌面）==")
        for s in [
            f"1. 打开/聚焦微信（Win+R → wechat，或聚焦已开窗口）",
            f"2. 搜索框点开与「{args.contact}」的会话",
            f"3. 监听收件箱（{'OCR 抓屏读消息' if args.ocr else 'JSON/Mock'}）；真人鼠标一动立即取消",
            f"4. 每条新消息：LLM({args.llm_backend}) 草稿 + IntentGate(authorized-send) 放行 + 键入回车",
            f"5. 删除/支付永不碰；换联系人即停",
            f"   披露={'关(冒充真人)' if args.no_disclosure else '开(每条带'+disclosure+')'}",
        ]:
            print("  " + s)
        print("\n>>> 要真机执行，加 --real（会真实打开微信并接管键鼠/发送）。")
        return

    # ---- 真机装配 ----
    from perception.llm_bridge import build_llm
    adapter = DesktopAdapter(dry_run=False, focus_title="微信")
    adapter.enable()
    llm = EchoLLM() if args.echo else build_llm(args.llm_backend)
    if args.ocr:
        inbox = OCRInbox(adapter, args.contact)
    else:
        inbox = FileInbox(args.inbox) if args.inbox else MockInbox()
    monitor = HumanOverrideMonitor()

    agent = WeChatChatAgent(args.contact, adapter, llm, inbox,
                            allow_send=True, disclosure=disclosure, monitor=monitor)
    agent.open_and_focus()
    agent.navigate_to_contact(args.contact)
    agent.run_loop(max_iter=args.max_iter)


if __name__ == "__main__":
    main()
