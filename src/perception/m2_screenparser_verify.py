"""
m2_screenparser_verify.py —— M2 真通验证：ScreenParser 主视觉通道真能跑
============================================================================
用合成桌面图（标题栏 + 按钮 + 输入框）验证：
  · 权重已就位（blobs/weights/ScreenParser/best.pt）
  · ScreenParser 模型能加载 + 前向推理 + 产出结构化 UI 元素
  · desktop_perceive.perceive_pipeline 的 L1 标 active_backend="screenparser"

环境坑（仅本沙箱）：torch(cu132) + torchvision(CPU) 混装 → CUDA 后端 NMS 缺失。
**历史教训**：本脚本曾 monkey-patch 掉 _ensure 强制 CPU（旧注释「既规避坑」），
导致验收走的是与生产不同的路径 —— 验收通过但生产路径照崩，主通道静默降级成
CV 找方块，长期未被发现。现在 detector.py 的 ScreenParserBackend 已默认
device='cpu'，**本脚本不再 patch，必须与生产走完全相同路径**，否则结论无效。
真实 GPU 机器可显式传 device="cuda"（需 torchvision 配套）。

定量召回（>90%）需真实桌面截图 + ground truth，沙箱无此条件；
此处只证「模型加载 + 推理 + 元素产出」端到端通，量化验收留真机（M2 验收关）。
"""

from __future__ import annotations

import os
import sys

# 强制 CPU：规避本沙箱 CUDA/torchvision 混装导致的 NMS 崩
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from PIL import Image, ImageDraw
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from desktop_perceive import perceive_pipeline


def _make_sample(path: str) -> np.ndarray:
    W, H = 900, 560
    img = Image.new("RGB", (W, H), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, W - 20, H - 20], outline=(120, 120, 120), width=2)
    d.rectangle([22, 22, W - 22, 60], fill=(40, 90, 160))          # 标题栏
    d.rectangle([120, 460, 260, 510], fill=(70, 130, 220))         # 按钮1
    d.rectangle([300, 460, 440, 510], fill=(200, 80, 80))          # 按钮2
    d.rectangle([120, 200, 600, 250], outline=(90, 90, 90), width=2)  # 输入框1
    d.rectangle([120, 300, 600, 350], outline=(90, 90, 90), width=2)  # 输入框2
    img.save(path)
    return np.array(img)


def main() -> int:
    print(f"[env] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
    os.makedirs("blobs/shots", exist_ok=True)
    sample = _make_sample("blobs/shots/m2_sample.png")

    # ---- 真实模型加载 + 推理（强制 CPU）----
    from ultralytics import YOLO
    from perception.weights import resolve_weights
    ckpt = resolve_weights("docling-project/ScreenParser")
    print(f"[load] {ckpt}  exists={os.path.isfile(ckpt)}")
    model = YOLO(ckpt)
    model.to("cpu")
    res = model.predict(sample, imgsz=1280, conf=0.10, iou=0.10, device="cpu")
    raw = []
    for r in res:
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            raw.append((model.names[int(cls_id)], (x1, y1, x2, y2), float(conf)))
    print(f"[ScreenParser] 检出元素数={len(raw)}")
    for lbl, bb, c in raw[:10]:
        print(f"    {lbl:16s} bbox={bb} conf={c:.2f}")

    # ---- 完整管线：perceive_pipeline 的 L1 应标 screenparser ----
    # E 修复（重要）：以前这里 monkey-patch 掉 ScreenParserBackend._ensure 强制 CPU，
    # 注释还写着「既规避坑」—— 结果**验收走的是另一条路径**：验收绿、生产照崩
    # （CUDA NMS），主通道静默降级成 CV 找方块，藏了很久。
    # 现在生产侧已默认 device='cpu'，这里**不再 patch**，必须与生产走完全相同的
    # 调用路径，否则验收结论无效。
    sc = perceive_pipeline(sample, goal="点击确定")
    print(f"\n[perceive_pipeline] active_backend={sc.active_backend} 元素数={len(sc.elements)}")
    ok = sc.active_backend == "screenparser" and len(sc.elements) >= 0
    print("M2 主视觉通道真通:", "✅ 模型加载+推理+产出元素+管线标 screenparser"
          if ok else "❌ 未走 screenparser")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
