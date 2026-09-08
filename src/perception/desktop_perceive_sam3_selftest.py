"""生产入口自检：确认 SAM3（经 SemanticLocator）真正接进 perceive_pipeline。

A. 轻量：use_sam3=False + 合成帧 → 不崩，优雅降级（yoloe 权重 404 → 落 cv 兜底）
B. 真机：use_sam3=True + 真实微信截图 → 开放词汇通道经 SemanticLocator 走到 SAM3，
   返回 active_backend="semantic" 且含 SAM3 元素（会真正加载 3.4GB 模型，约 30s）

用法：python desktop_perceive_sam3_selftest.py [--real]
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
from desktop_perceive import perceive_pipeline

IMG = os.path.join(_ROOT, "wechat_full.png")
REAL = "--real" in sys.argv


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return ok


def test_light():
    print("=== A. 轻量：use_sam3=False，合成帧 ===")
    fake = np.zeros((200, 320, 3), dtype=np.uint8)
    sc = perceive_pipeline(fake, goal="点击确定按钮", l2_bridge=None,
                           force_backend="yoloe", use_sam3=False)
    ok = check("force_backend=yoloe 不崩、返回 Scene",
               sc is not None and hasattr(sc, "elements"))
    ok &= check("yoloe 权重缺失时优雅降级（非 semantic 即 cv 兜底）",
                sc.active_backend in ("cv", "semantic"), f"backend={sc.active_backend}")
    return ok


def test_routing():
    """路由逻辑（mock，不加载 3.4GB 模型）：验证 SAM3 升级通道按预期触发/跳过。"""
    print("=== C. 路由逻辑：SAM3 升级通道的触发条件（mock）===")
    import desktop_perceive as dp
    from detector import Element
    orig_screen = dp._try_screenparser
    orig_open = dp._try_yoloe_open
    fake = np.zeros((200, 320, 3), dtype=np.uint8)
    ok = True
    try:
        captured = {}
        def fake_open(frame, names=None, use_sam3=True):
            captured['called'] = True
            captured['names'] = names
            captured['use_sam3'] = use_sam3
            return [Element(id="s1", label="icon", bbox=(1, 1, 2, 2), conf=0.9)]
        dp._try_yoloe_open = fake_open

        # 1) L1 命中 → 直接 return screenparser，不升级
        dp._try_screenparser = lambda f: [Element(id="u1", label="Button",
                                                   bbox=(1, 1, 2, 2), conf=0.9)]
        sc = dp.perceive_pipeline(fake, goal="点击红色按钮", use_sam3=True)
        ok &= check("L1 已命中 → 直接返回 screenparser，不升级 SAM3",
                    sc.active_backend == "screenparser", f"backend={sc.active_backend}")

        # 2) L1 空 + goal → 升级 SAM3（这正是之前漏掉的生产路径）
        captured.clear()
        dp._try_screenparser = lambda f: []
        sc = dp.perceive_pipeline(fake, goal="点击红色按钮", use_sam3=True)
        ok &= check("L1 空+goal → 升级到 SAM3 通道（semantic）",
                    sc.active_backend == "semantic", f"backend={sc.active_backend}")
        ok &= check("开放词汇通道被调用且收到语义名",
                    captured.get('called') and captured.get('names') == ["点击红色按钮"],
                    f"names={captured.get('names')}")
        ok &= check("use_sam3 透传 True", captured.get('use_sam3') is True)

        # 3) L1 空 + goal 但 use_sam3=False → 不升级（落 cv）
        captured.clear()
        sc = dp.perceive_pipeline(fake, goal="点击红色按钮", use_sam3=False)
        ok &= check("L1 空+goal 但 use_sam3=False → 不升级（落 cv 兜底）",
                    sc.active_backend == "cv", f"backend={sc.active_backend}")

        # 4) L1 空 + goal 空 + use_sam3=True → 不无谓触发 SAM3
        captured.clear()
        sc = dp.perceive_pipeline(fake, goal="", use_sam3=True)
        ok &= check("L1 空+goal 为空 → 不无谓跑 SAM3",
                    not captured.get('called'), f"called={captured.get('called')}")
    finally:
        dp._try_screenparser = orig_screen
        dp._try_yoloe_open = orig_open
    return ok


def test_real():
    if not os.path.isfile(IMG):
        print(f"跳过真机：缺 {IMG}")
        return True
    import cv2
    import desktop_perceive as dp
    frame = cv2.imread(IMG)
    ok = True
    orig_screen = dp._try_screenparser
    dp._try_screenparser = lambda f: []   # L1 空：隔离真实 screenparser 行为，确保走升级通道
    try:
        # B1: 生产默认路径（不传 force_backend）+ L1 空 + goal → 升级 SAM3（本轮修复核心点）
        print("=== B. 真机：生产默认路径（无 force_backend）+ L1 空 + goal → 升级 SAM3 ===")
        t0 = time.time()
        sc = perceive_pipeline(frame, goal="icon", l2_bridge=None, use_sam3=True)
        dt = time.time() - t0
        ok &= check("返回 Scene", sc is not None)
        ok &= check("active_backend == semantic（生产默认路径升级 SAM3）",
                    sc.active_backend == "semantic", f"backend={sc.active_backend} 耗时={dt:.0f}s")
        ok &= check("含 SAM3 元素", len(sc.elements) > 0, f"n={len(sc.elements)}")
        if sc.elements:
            e = sc.elements[0]
            print(f"   示例: {e.label} bbox={tuple(e.bbox)} conf={e.conf:.2f} backend={e.backend}")
        # B2: force_backend=yoloe 钩子仍通（对照）
        print("=== B2. 真机：force_backend=yoloe 对照 ===")
        t0 = time.time()
        sc = perceive_pipeline(frame, goal="icon", l2_bridge=None,
                               force_backend="yoloe", use_sam3=True)
        dt = time.time() - t0
        ok &= check("force_backend=yoloe 仍走 semantic",
                    sc.active_backend == "semantic", f"backend={sc.active_backend} 耗时={dt:.0f}s")
    finally:
        dp._try_screenparser = orig_screen
    return ok


if __name__ == "__main__":
    ok = test_light()
    ok &= test_routing()
    if REAL:
        ok &= test_real()
    else:
        print("\n（真机层跳过，加 --real 启用）")
    print("\n结果:", "全部通过" if ok else "存在失败")
    sys.exit(0 if ok else 1)
