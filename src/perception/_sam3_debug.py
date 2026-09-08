"""临时调试：打印 SAM3 set_text_prompt 的原始输出结构，确认字段名与阈值影响。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PIL import Image                                    # noqa: E402
from sam3_locator import SAM3Locator                     # noqa: E402


def summarize(o, depth=0):
    pad = "  " * depth
    if o is None:
        return f"{pad}None"
    if isinstance(o, dict):
        s = f"{pad}dict({len(o)}) keys={list(o.keys())}"
        for k, v in list(o.items())[:8]:
            s += "\n" + summarize(v, depth + 1) + f"   <- {k}"
        return s
    if isinstance(o, (list, tuple)):
        return f"{pad}{type(o).__name__}({len(o)})"
    if hasattr(o, "shape"):
        return f"{pad}{type(o).__name__} shape={tuple(o.shape)} dtype={getattr(o,'dtype','')}"
    if hasattr(o, "__len__"):
        try:
            return f"{pad}{type(o).__name__} len={len(o)}"
        except Exception:
            pass
    return f"{pad}{type(o).__name__} {str(o)[:80]}"


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "wechat_full.png")
    prompts = sys.argv[2:] or ["button", "icon", "text", "person", "window", "screen"]

    loc = SAM3Locator(conf=0.05)          # 故意压低阈值，先看有没有东西
    if not loc.available:
        print("不可用:", loc.load_error)
        return 2

    img = Image.open(img_path).convert("RGB")
    print(f"图片 {img_path}  size={img.size}")
    state = loc._processor.set_image(img)

    for p in prompts:
        res = loc._processor.set_text_prompt(state=state, prompt=p)
        print(f"\n=== prompt={p!r} ===")
        print(summarize(res))
        # 用 locator 自己的解析器看能抽出什么
        boxes, scores, masks = loc._unpack(res)
        print("  解析 → boxes:", None if boxes is None else boxes.shape,
              "| scores:", None if scores is None else scores.shape,
              "| masks:", None if masks is None else "yes")
        if scores is not None:
            import numpy as np
            s = np.asarray(scores).reshape(-1)
            if s.size == 0:
                print("  score: (空，该概念在画面里无命中)")
            else:
                print(f"  score max={s.max():.3f} min={s.min():.3f} n={len(s)} "
                      f"| >0.05:{(s>0.05).sum()} >0.5:{(s>0.5).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
