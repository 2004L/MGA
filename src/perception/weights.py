"""
weights.py —— 模型权重解析（本地优先 + 镜像兜底）
================================================
为什么不直接把权重名丢给 YOLO：
    本机（以及国内相当多环境）的网络实况是
        github.com       不可达
        huggingface.co   不可达
        pypi.org         可达（所以依赖装得上）
        hf-mirror.com    可达
    而 ultralytics / transformers 默认从 github / HF 拉权重，直接就是连不上超时。
    实测 YOLOv8n 权重走了 3 次重试全失败、耗时 6 分钟才放弃。

    所以统一由本模块解析：先在 blobs/weights/ 找本地文件，找不到再经 HF 镜像下载，
    下载过一次后续全部走本地缓存，不再碰网络。

权重规格 spec 支持三种写法：
    1) 本地文件路径            "blobs/weights/yolov8n.pt"
    2) 裸文件名               "yolov8n.pt"      → 在 weights_dir 下找
    3) HF 仓库 id             "docling-project/ScreenParser"
                             → 在 weights_dir/<仓库名>/ 找，找不到经镜像下载

零依赖：仅在真正需要下载时才 import huggingface_hub。
"""
from __future__ import annotations

import os

WEIGHTS_DIR = os.path.join("blobs", "weights")
DEFAULT_MIRROR = "https://hf-mirror.com"


def hf_endpoint() -> str:
    """HF 端点：优先尊重用户已设的 HF_ENDPOINT，否则用国内可达的镜像。"""
    return os.environ.get("HF_ENDPOINT") or DEFAULT_MIRROR


def resolve_weights(spec: str, weights_dir: str = WEIGHTS_DIR,
                    hf_filename: str = "best.pt") -> str:
    """把权重规格解析成**本地可加载路径**，找不到才联网（经镜像）。"""
    # 1) 已是存在的本地路径
    if os.path.isfile(spec):
        return spec
    # 2) 裸文件名 → 权重目录
    local = os.path.join(weights_dir, spec)
    if os.path.isfile(local):
        return local
    # 3) HF 仓库 id → 本地缓存目录
    if "/" in spec:
        repo_dir = os.path.join(weights_dir, spec.split("/")[-1])
        cached = os.path.join(repo_dir, hf_filename)
        if os.path.isfile(cached):
            return cached
        if os.path.isdir(repo_dir):
            for fn in sorted(os.listdir(repo_dir)):
                if fn.endswith(".pt"):
                    return os.path.join(repo_dir, fn)
        return _download(spec, hf_filename, repo_dir)
    # 4) 解析不了：原样返回，交给上层按默认逻辑处理（其失败会走既有降级链路）
    return spec


def _download(repo: str, filename: str, dest_dir: str) -> str:
    """经 HF 镜像下载权重到本地目录。失败直接抛错，由调用方降级。"""
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_MIRROR)
    from huggingface_hub import hf_hub_download
    os.makedirs(dest_dir, exist_ok=True)
    return hf_hub_download(repo, filename, local_dir=dest_dir)


# 各 visual_backend 的默认权重：screenparser 用 GUI 检测器，yolo 用 COCO 检测器。
# 区别很关键——COCO 只能看见 person/tv/laptop，用它采 GUI 操作轨迹等于
# 教模型「看着电视点确定」；GUI 场景必须用 docling-project/ScreenParser（55 类 UI 组件）。
BACKEND_WEIGHTS = {
    "screenparser": "docling-project/ScreenParser",
    "yolo": "yolov8n.pt",
}


if __name__ == "__main__":
    print("HF 端点:", hf_endpoint())
    print("权重目录:", WEIGHTS_DIR)
    for backend, spec in BACKEND_WEIGHTS.items():
        try:
            p = resolve_weights(spec)
            tag = "本地" if os.path.isfile(p) else "待下载"
            print(f"  {backend:<14} {spec:<32} → {p}  [{tag}]")
        except Exception as e:
            print(f"  {backend:<14} {spec:<32} → 解析失败 {type(e).__name__}")
