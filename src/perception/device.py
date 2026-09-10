"""
device.py —— 推理设备统一收敛
=============================
为什么需要这个模块：

    本机（以及相当多国内环境）torchvision 未编译 CUDA 版 NMS。一旦
    ultralytics 自动把张量放到 CUDA，predict 阶段必抛
        NotImplementedError: Could not run 'torchvision::nms' with arguments
        from the 'CUDA' backend
    该异常发生在感知主通道，还会被上层降级链路吞掉 —— 症状是整条链退化成
    CV 找方块（元素全是 rect、没有任何语义标签），从外部看「功能没坏、只是
    没结果」，排查成本极高（本次就是这么藏了很久）。

    根因不在模型，在于 **predict 没指定 device，交给 ultralytics 隐式自动选**。

所以：所有 ultralytics 推理一律经本模块，强制带上 device，杜绝隐式选择。
新增通路时请直接用 `device.predict(...)`，不要再裸调 `model.predict(...)`。
"""
from __future__ import annotations

from typing import Optional

# 默认钉死 CPU：ScreenParser 等模型本就标注「CPU 可跑」，且本机 CUDA 路径已验证会崩。
# 确实有可用 GPU 且 torchvision 配套时，显式传 device="cuda" 覆盖。
DEFAULT_DEVICE = "cpu"


def resolve_device(device: Optional[str] = None) -> str:
    """解析推理设备：未指定则用默认（CPU）。"""
    return device or DEFAULT_DEVICE


def predict(model, frame, device: Optional[str] = None, **kw):
    """统一推理入口：给 ultralytics 的 predict 强制补 device。

    用法与 model.predict(frame, **kw) 完全一致，只是 device 不再可省略。
    """
    kw["device"] = resolve_device(device)
    return model.predict(frame, **kw)
