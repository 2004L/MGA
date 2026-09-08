"""
desktop_wechat_chat_selftest.py —— 沙箱自测（全 mock，不碰真实桌面/键鼠）
验证：① IntentGate 红线放松（allow_send 开关）② 鼠标取消 ③ 授权范围内才发送
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from safety.intent_gate import IntentGate
from desktop_wechat_chat import (WeChatChatAgent, HumanOverrideMonitor,
                                 MockInbox, EchoLLM)


class MockCU:
    def __init__(self):
        self.calls = []
        self.name = "MockCU"

    def client_rect(self, title):
        return (0, 0, 800, 600)

    def click(self, x, y): self.calls.append(("click", x, y))
    def type(self, t): self.calls.append(("type", t))
    def press(self, k): self.calls.append(("press", k))
    def focus_window(self, t): self.calls.append(("focus", t))
    def hotkey(self, *ks): self.calls.append(("hotkey", ks))


class MockAdapter:
    def __init__(self):
        self.cu = MockCU()
        self.enabled = True

    def enable(self):
        self.enabled = True

    def act(self, action):
        parts = action.split()
        op = parts[0].lower()
        if op == "click" and len(parts) >= 3:
            self.cu.click(float(parts[1]), float(parts[2]))
        elif op == "type" and len(parts) >= 2:
            self.cu.type(" ".join(parts[1:]))
        elif op == "press" and len(parts) >= 2:
            self.cu.press(parts[1])
        elif op == "hotkey" and len(parts) >= 2:
            self.cu.hotkey(*parts[1:])
        elif op == "focus" and len(parts) >= 2:
            self.cu.focus_window(" ".join(parts[1:]))


_n, _ok = 0, 0


def _check(name, cond, extra=""):
    global _n, _ok
    _n += 1
    if cond:
        _ok += 1
        print(f"  [OK] {name} {extra}")
    else:
        print(f"  [FAIL] {name} {extra}")


def test_intent_gate():
    print("\n[1] IntentGate 红线放松（allow_send 开关）")
    g_off = IntentGate(allow_send=False)
    _check("默认 SEND→BLOCK", g_off.gate("给张三发送一条消息").blocked())
    _check("DELETE 永远 BLOCK", g_off.gate("删除聊天记录").blocked())
    _check("PAYMENT 永远 NEED_CONFIRM",
           g_off.gate("微信支付付这笔").need_confirm())

    g_on = IntentGate(allow_send=True)
    d = g_on.gate("给张三发条消息")
    _check("allow_send=True SEND→ALLOW", d.verdict == "ALLOW", f"→ {d.reason}")
    _check("allow_send=True DELETE 仍 BLOCK", g_on.gate("删除聊天记录").blocked())
    _check("allow_send=True PAYMENT 仍 NEED_CONFIRM",
           g_on.gate("转账给张三").need_confirm())


def test_send_scope():
    print("\n[2] 授权范围内才发送（run_once）")
    # 授权：allow_send=True → 应发送
    a = WeChatChatAgent("张三", MockAdapter(), EchoLLM(), MockInbox(),
                        allow_send=True, disclosure="(AI代回)", monitor=HumanOverrideMonitor())
    a.run_once("在吗")
    sent = [c for c in a.adapter.cu.calls if c[0] in ("type", "press")]
    _check("授权后真正发送(type+enter)",
           any(c[0] == "type" for c in sent) and any(c[0] == "press" for c in sent),
           f"calls={sent}")

    # 未授权：allow_send=False → 不应发送
    a2 = WeChatChatAgent("张三", MockAdapter(), EchoLLM(), MockInbox(),
                         allow_send=False, disclosure="", monitor=HumanOverrideMonitor())
    res = a2.run_once("在吗")
    _check("未授权 run_once 返回 False", res is False)
    _check("未授权零发送", len(a2.adapter.cu.calls) == 0,
           f"calls={a2.adapter.cu.calls}")


def test_mouse_cancel():
    print("\n[3] 真人鼠标活动 → 自动取消（run_loop 不发）")
    mon = HumanOverrideMonitor()
    mon.cancel.set()          # 模拟：真人已动鼠标
    a = WeChatChatAgent("张三", MockAdapter(), EchoLLM(), MockInbox(),
                        allow_send=True, disclosure="(AI代回)", monitor=mon)
    a.run_loop(max_iter=3)
    _check("取消态下未发送", len([c for c in a.adapter.cu.calls if c[0] == "type"]) == 0,
           f"calls={a.adapter.cu.calls}")
    _check("标记 _cancelled", a._cancelled is True)


def test_normal_loop():
    print("\n[4] 正常循环：收到消息→发送→收件箱空结束")
    a = WeChatChatAgent("张三", MockAdapter(), EchoLLM(), MockInbox(),
                        allow_send=True, disclosure="(AI代回)", monitor=HumanOverrideMonitor())
    a.run_loop(max_iter=3)
    _check("正常发送了 1 条", a._sent == 1, f"sent={a._sent}")


if __name__ == "__main__":
    test_intent_gate()
    test_send_scope()
    test_mouse_cancel()
    test_normal_loop()
    print(f"\n自测结果：{_ok}/{_n} 通过")
    sys.exit(0 if _ok == _n else 1)
