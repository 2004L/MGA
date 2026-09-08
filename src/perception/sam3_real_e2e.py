"""真机验证：base python 经 worker 子进程(.venv-sam3)调 SAM3。

不 mock、不跳过：验证 base 端到端能否拉起 venv、加载 3.4GB 权重、返回 Element。
用法：python sam3_real_e2e.py <image> [prompt1 prompt2 ...]
"""
import os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import cv2
from sam3_locator import SAM3Locator

IMG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "wechat_full.png")
PROMPTS = sys.argv[2:] or ["icon", "button", "window"]


def main():
    t0 = time.time()
    print(f"[base] 使用镜像: {IMG}")
    frame = cv2.imread(IMG)
    if frame is None:
        print("ERR 读图失败"); sys.exit(1)
    print(f"[base] 图尺寸 {frame.shape[1]}x{frame.shape[0]}, 自动进入 worker 模式(import sam3 会失败)")

    loc = SAM3Locator()  # 默认路径 + 自动选 worker
    print(f"[base] 自动进入模式={'worker' if loc._mode=='worker' else 'in-process'}"
          f"（import sam3 失败→走子进程）")

    try:
        els = loc.detect_named(frame, PROMPTS)
    except Exception as e:
        print(f"detect_named 抛异常: {type(e).__name__}: {e}")
        sys.exit(1)

    dt = time.time() - t0
    print(f"[base] 拿到 {len(els)} 个元素 (耗时 {dt:.1f}s)")
    for i, el in enumerate(els[:8]):
        cx = (el.bbox[0] + el.bbox[2]) / 2
        cy = (el.bbox[1] + el.bbox[3]) / 2
        cp = loc.click_point(i)
        print(f"  #{el.id:<3} {el.label:<8} conf={el.conf:.2f} "
              f"bbox=({el.bbox[0]:.0f},{el.bbox[1]:.0f},{el.bbox[2]:.0f},{el.bbox[3]:.0f}) "
              f"center=({cx:.0f},{cy:.0f}) mask_centroid={cp}")
    print("E2E_OK")


if __name__ == "__main__":
    main()
