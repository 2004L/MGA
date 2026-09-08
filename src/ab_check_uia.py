"""
ab_check_uia.py —— UIA 融合的 A/B 对照验证
==========================================
同一时刻、同一屏幕，对比两条路的**文字质量**：
    A 旧路径：ScreenParser(框) + EasyOCR(文字)      → 噪声多
    B 新路径：ScreenParser(框) + OCR + UIA 真值融合 → 噪声应大幅下降

为什么必须量化对比：「感觉上更准」不算数。要看带文字元素数、噪声数、
中文名数这些硬指标，才能证明重采数据是值得的。

用法：
  PYTHONPATH=src python -m ab_check_uia
"""
import os
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
ROOT = os.path.dirname(SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from exec.computer_use import ComputerUse, SafetyGuard  # noqa: E402
from perception.detector import (OCRBackend, ScreenParserBackend,  # noqa: E402
                                 TriLevelLocator, UIALocator)


def is_noise(t: str) -> bool:
    """判定 OCR 噪声：过短，或纯 ASCII 短串（'on'/'L7'/'STS' 这类误识高发区）。"""
    t = (t or "").strip()
    if len(t) < 2:
        return True
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in t)
    return (not has_cjk) and len(t) < 4


def quality(els):
    named = [e for e in els if (getattr(e, "text", None) or "").strip()]
    noise = [e for e in named if is_noise(e.text)]
    cjk = [e for e in named
           if any("\u4e00" <= ch <= "\u9fff" for ch in e.text)]
    return len(els), len(named), len(noise), len(cjk), named


def report(tag, els, dt):
    n, nm, nz, cj, named = quality(els)
    pct = (nz / nm * 100) if nm else 0
    print(f"\n{tag}")
    print(f"  元素 {n} 个 | 带文字 {nm} 个 | 噪声 {nz} 个 ({pct:.0f}%) | "
          f"中文名 {cj} 个 | 耗时 {dt:.1f}s")
    print(f"  文字样例: {[e.text for e in named[:14]]}")
    return nm, nz


def main():
    cu = ComputerUse(SafetyGuard(dry_run=True, require_confirm=False,
                                 allowed_region=(0, 0, 1920, 1080)))
    res = cu.screenshot()
    shot = os.path.join(ROOT, "blobs", "shots", "_ab_check.png")
    os.makedirs(os.path.dirname(shot), exist_ok=True)
    if isinstance(res, Image.Image):
        res.save(shot)
    else:
        shot = str(res)
    print(f"截图: {shot}")

    # A：旧路径（只有 OCR 补文字，UIA 不参与）
    loc_a = TriLevelLocator(visual=ScreenParserBackend(),
                            ocr=OCRBackend(scale=0.5), uia=None)
    t0 = time.time()
    els_a = loc_a.detect(shot)
    nm_a, nz_a = report("【A】旧路径：ScreenParser + OCR", els_a, time.time() - t0)

    # B：新路径（UIA 真值融合，覆盖 OCR 噪声 + 补充漏检控件）
    uia = UIALocator(budget=8.0)
    if not uia.available():
        print("\n!! uiautomation 不可用（pip install uiautomation comtypes）")
        return 1
    loc_b = TriLevelLocator(visual=ScreenParserBackend(),
                            ocr=OCRBackend(scale=0.5), uia=uia)
    t0 = time.time()
    els_b = loc_b.detect(shot)
    nm_b, nz_b = report("【B】新路径：ScreenParser + OCR + UIA 真值融合",
                        els_b, time.time() - t0)

    print("\n=== 结论 ===")
    print(f"  带文字元素: {nm_a} → {nm_b}  ({nm_b - nm_a:+d})")
    print(f"  噪声元素  : {nz_a} → {nz_b}  ({nz_b - nz_a:+d})")
    if nm_b:
        print(f"  噪声占比  : {nz_a / max(nm_a,1) * 100:.0f}% → "
              f"{nz_b / nm_b * 100:.0f}%")
    print("\n噪声降下来 = 训练数据的 (题目,答案) 终于对齐了，")
    print("模型学的才不是「噪声串 → 坐标」的假映射。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
