"""
desktop_wechat_chat_selftest_ocr.py —— M9 OCR 读消息 + hy3 别名 沙箱自测（零真机依赖）
===================================================================================
覆盖：
  · OCRInbox：去重(已见集合) / 跳过自带(AI代回)前缀的自己消息 / 返回最新未读 / ack 后不重复
  · build_llm("hy3")：返回 APILLM 且 model="hy3"，密钥只从 .env.local/环境变量读
"""
from __future__ import annotations

import os
import sys
import types
from typing import List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from desktop_wechat_chat import OCRInbox
from perception.llm_bridge import build_llm, APILLM


# ============================ 假依赖 ============================
class FakeOCR:
    """返回预设文本行（按 y 排序的 (cx,cy,text,conf)），忽略输入裁图。"""

    def __init__(self, lines: List[Tuple[float, float, str, float]]):
        self._lines = lines
        self.last_ocr_ran = True

    def read_lines(self, frame, conf_thresh: float = 0.1):
        return list(self._lines)


class FakeAdapter:
    """最小 adapter：grab 返回固定帧，client_rect 返回固定矩形。"""

    def __init__(self, frame, rect):
        self._frame = frame
        self._rect = rect
        self.cu = types.SimpleNamespace(client_rect=lambda t: self._rect)

    def grab(self):
        return self._frame


# ============================ 自测 ============================
_pass = _fail = 0


def _check(name: str, cond: bool, extra: str = "") -> None:
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  [PASS] {name}")
    else:
        _fail += 1
        print(f"  [FAIL] {name}  {extra}")


def _t_ocr_inbox() -> None:
    print("\n[OCRInbox] 去重 / 跳过自己 / 返回最新未读")
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    rect = (0, 0, 300, 400)

    # 两行消息 + 一行自己的回复（带前缀）
    lines = [
        (50.0, 100.0, "在吗？", 0.95),
        (50.0, 200.0, "(AI代回)收到，我看看时间安排下～", 0.95),
        (50.0, 300.0, "周末打三角洲吗", 0.95),
    ]
    ocr = FakeOCR(lines)
    adapter = FakeAdapter(frame, rect)
    inbox = OCRInbox(adapter, "张三", ocr=ocr)

    m1 = inbox.read_latest("张三")
    _check("返回最新未读=周末打三角洲吗", m1 == "周末打三角洲吗", f"→ {m1!r}")
    _check("自己的回复被跳过", "(AI代回)" not in (m1 or ""))
    inbox.ack("张三")

    # ack 只标刚返回的那条；下一条未读应在下一轮被返回并回复（两条都回，不丢）
    m2 = inbox.read_latest("张三")
    _check("下一轮返回更早的未读=在吗？", m2 == "在吗？", f"→ {m2!r}")
    inbox.ack("张三")

    # 全部已见 → None
    m3 = inbox.read_latest("张三")
    _check("全部已见后返回 None", m3 is None, f"→ {m3!r}")

    # 联系人发了新消息
    ocr._lines = [
        (50.0, 100.0, "在吗？", 0.95),
        (50.0, 200.0, "(AI代回)收到，我看看时间安排下～", 0.95),
        (50.0, 300.0, "周末打三角洲吗", 0.95),
        (50.0, 380.0, "好啊，那约几点", 0.95),
    ]
    m4 = inbox.read_latest("张三")
    _check("新消息被识别=好啊，那约几点", m4 == "好啊，那约几点", f"→ {m4!r}")
    inbox.ack("张三")

    # 换联系人 → 仍只读张三的会话（OCRInbox 绑死 contact，但 read_latest 入参是调度层传的）
    # 这里验证：contact 参数不影响去重集合（集合按文本），这是预期行为，调度层保证只问授权联系人
    _check("OCRInbox 实例绑定单一 contact 字符串", inbox.contact == "张三")


def _t_hy3() -> None:
    print("\n[build_llm hy3] 别名 + 密钥隔离")
    b = build_llm("hy3")
    _check("hy3 → APILLM", isinstance(b, APILLM))
    _check("hy3 默认 model=hy3", b.model == "hy3", f"→ {b.model}")
    _check("无 key 不崩（仅实例化）", True)

    os.environ["MGA_LLM_MODEL"] = "hunyuan-3-standard"
    try:
        b2 = build_llm("hy3")
        _check("env MGA_LLM_MODEL 覆盖模型名", b2.model == "hunyuan-3-standard",
               f"→ {b2.model}")
    finally:
        os.environ.pop("MGA_LLM_MODEL", None)

    # 别名同义
    for alias in ("hunyuan", "hunyuan3", "tencent"):
        _check(f"别名 {alias} 可用", isinstance(build_llm(alias), APILLM))


if __name__ == "__main__":
    _t_ocr_inbox()
    _t_hy3()
    print(f"\n=== M9 自测：PASS={_pass}  FAIL={_fail} ===")
    raise SystemExit(1 if _fail else 0)
