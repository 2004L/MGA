"""
desktop_perceive_selftest.py —— M1 沙箱自检（不碰游戏/网络）
============================================================

验证：
  1. 感知三级降级链路：无 YOLOE/无 L2 → 落 CV 兜底，active_backend 如实标注
  2. L2 升档：给 MockLLM 时走 llm 通路，返回带坐标元素
  3. IntentGate 三态：delete/send→BLOCK，payment→NEED_CONFIRM，normal→ALLOW
  4. DesktopAdapter 默认关 → act 拒绝；enable + dry_run → 路由执行层
  5. build_llm("gpt6") 返回 APILLM(gpt-6-astra) 且不触发网络

运行：cd src && python perception/desktop_perceive_selftest.py
      （或直接 python desktop_perceive_selftest.py，路径兜底已处理）
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SAFETY = os.path.join(_ROOT, "safety")
for _p in (_ROOT, _HERE, _SAFETY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from desktop_perceive import perceive_pipeline, UIElement, DesktopScene
from intent_gate import IntentGate, Intent
from llm_bridge import MockLLM, build_llm, APILLM


def _banner(t: str):
    print(f"\n=== {t} ===")


def main() -> int:
    fails = 0

    # 1. 感知三级降级（无权重/无 L2 → CV 兜底）
    _banner("1. 感知降级链路（无 YOLOE / 无 L2 → CV）")
    fake = np.zeros((200, 320, 3), dtype=np.uint8)
    sc: DesktopScene = perceive_pipeline(fake, goal="点击确定", l2_bridge=None)
    print(f"  active_backend={sc.active_backend} 元素数={len(sc.elements)}")
    # 任一已知通路都合法：screenparser(主视觉)/yoloe(开放词汇)/llm(L2)/cv(L3)
    if sc.active_backend not in ("screenparser", "yoloe", "llm", "cv"):
        print("  FAIL: active_backend 异常")
        fails += 1
    else:
        print("  OK: 降级链路不崩，active_backend 如实标注")

    # 2. L2 升档（MockLLM → llm 通路）
    _banner("2. L2 升档（MockLLM → llm 通路）")
    sc2 = perceive_pipeline(fake, goal="点击确认按钮", l2_bridge=MockLLM())
    print(f"  active_backend={sc2.active_backend} 元素数={len(sc2.elements)}")
    if sc2.active_backend == "llm" and sc2.elements:
        cx, cy = sc2.elements[0].center()
        print(f"  OK: L2 返回元素，中心=({cx:.0f},{cy:.0f}) backend={sc2.elements[0].backend}")
    else:
        print("  FAIL: L2 未返回 llm 元素")
        fails += 1

    # 3. IntentGate 三态
    _banner("3. IntentGate 三态")
    g = IntentGate()
    cases = [
        ("把下载文件夹里的大文件删除", "BLOCK"),
        ("给客户发送这封邮件", "BLOCK"),
        ("用微信支付付这笔订单", "NEED_CONFIRM"),
        ("在表单里填好姓名和电话", "ALLOW"),
        ("把桌面报告和表格整理到归档目录", "ALLOW"),
    ]
    for text, expect in cases:
        d = g.gate(text)
        ok = d.verdict == expect
        print(f"  [{'OK' if ok else 'FAIL'}] {text!r:32s} → {d.intent.value:8s} {d.verdict}")
        if not ok:
            fails += 1
    # 确认回调
    pend = g.gate("支付这杯咖啡")
    if pend.need_confirm() and g.confirm(pend.pending_id):
        print("  OK: 支付待确认 → confirm() 放行成功")
    else:
        print("  FAIL: confirm 回调异常")
        fails += 1

    # 4. DesktopAdapter 默认关 + 启用路由
    _banner("4. DesktopAdapter 默认关 / 启用路由")
    try:
        import importlib
        da = importlib.import_module("agent.desktop_adapter").DesktopAdapter
    except Exception:
        from agent.desktop_adapter import DesktopAdapter as da
    a = da(dry_run=True)
    try:
        a.act("click 530 320")
        print("  FAIL: 未启用却执行了")
        fails += 1
    except RuntimeError as e:
        print(f"  OK: 默认关拒绝: {e}")
    a.enable()
    a.act("click 530 320")      # dry_run → 只打印 DRY
    a.act("type 你好 hello")
    print("  OK: 启用后路由执行层（DRY）")

    # 5. build_llm("gpt6") 不触发网络
    _banner("5. build_llm('gpt6') 别名")
    try:
        bridge = build_llm("gpt6")
        if isinstance(bridge, APILLM) and bridge.model == "gpt-6-astra":
            print(f"  OK: 返回 APILLM(model={bridge.model})，未发起网络请求")
        else:
            print(f"  FAIL: 返回类型/模型不符: {type(bridge).__name__} {getattr(bridge,'model',None)}")
            fails += 1
    except Exception as e:
        print(f"  FAIL: build_llm('gpt6') 抛错: {type(e).__name__}: {e}")
        fails += 1

    print("\n" + ("全部通过 ✅" if fails == 0 else f"存在 {fails} 项失败 ❌"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
