"""
dino_detect.py —— 小恐龙障碍检测：深度学习(YOLOE 开放词汇) 主通道 + 颜色/形状 CV 兜底
===================================================================================

用户诉求："别用 cv 了，用 ScreenParser（深度学习）做 GUI 桌面访问能力 / computer use"。

落地到小恐龙障碍检测，需要分清楚两件事（避免重造轮子、也避免用错模型）：

  · ScreenParser（docling，55 类 UI 组件）的词汇是 Button/Text/Icon…，
    它**看不见**游戏里的恐龙/仙人掌/鸟——这些不在它的训练域。
    ScreenParser 在本项目里的正确职责是 **GUI 桌面访问**（定位 Chrome 窗口、
    识别窗口内按钮/输入框），这一层已在 exec/gui_agent.py 的 locate_game_window 接好。
  · 小恐龙游戏自身元素（恐龙 / 仙人掌 / 鸟）最适合用 **YOLOE 开放词汇检测器**：
    按语义名直接检测 "cactus" / "bird" / "dinosaur"，输出带名框，
    不受单色配色影响，无需 OCR。这才是"用深度学习替代纯色 CV"的正确模型。

所以本模块是 YOLOE 主通道；任一级不可用（ultralytics 未装 / 权重缺失 / 单帧推理异常）
→ 由 demo_dino_real.detect 自动降级到颜色+形状 CV 兜底，绝不崩（满血运行）。

接口：
    det = DeepDinoDetector()            # 懒加载，不触发任何下载/import
    if det.load():                      # 返回 True=深度学习可用
        dino_x, raw = det.detect(scene) # raw: [(x_rel, y_bottom, is_bird), ...]
    # dino_x=None 且 raw=[] → 上层改用 CV 兜底
"""
from __future__ import annotations

import os
import glob

import numpy as np

# 权重搜索顺序：本地 blobs/weights 里任意 yoloe*.pt 优先；其次尝试经镜像下载；
# 都失败则返回 None（上层走 CV 兜底）。不强制联网。
_WEIGHTS_DIR = os.path.join("blobs", "weights")
_DOWNLOAD_REPO = "jameslahm/yoloe"      # HF 仓库（含 yoloe-11s.pt / yoloe-11l.pt）
_PROMPTS = {
    "dinosaur": "dinosaur",
    "cactus": "cactus",
    "bird": "bird",
    # 开放词汇兜底：模型没见过 cactus 时，给它一个更泛的提示也能召回大部分障碍
    "plant": "plant",
}


def _find_local_yoloe() -> str:
    """在 blobs/weights 下找任意 yoloe*.pt，找到即返回路径。"""
    pat = os.path.join(_WEIGHTS_DIR, "yoloe*.pt")
    hits = sorted(glob.glob(pat))
    if hits:
        return hits[0]
    # 也找下载目录 blobs/weights/yoloe/*.pt
    pat2 = os.path.join(_WEIGHTS_DIR, "yoloe", "yoloe*.pt")
    hits2 = sorted(glob.glob(pat2))
    return hits2[0] if hits2 else ""


def _maybe_download() -> str:
    """经 HF 镜像拉一次 yoloe-11s.pt；失败返回空串（上层降级 CV）。"""
    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import hf_hub_download
        dest = os.path.join(_WEIGHTS_DIR, "yoloe")
        os.makedirs(dest, exist_ok=True)
        p = hf_hub_download(_DOWNLOAD_REPO, "yoloe-11s.pt", local_dir=dest)
        return p if os.path.isfile(p) else ""
    except Exception:
        return ""


class DeepDinoDetector:
    """YOLOE 开放词汇检测器，专用于小恐龙游戏元素。懒加载、缺失降级。

    用法见模块 docstring。detect() 返回 (dino_x_rel, raw_obstacles)，
    dino_x_rel 为相对游戏区左缘的恐龙中心 x；raw_obstacles 是
    (x_rel 左缘, y_bottom 底部相对 y, is_bird) 元组列表。
    全部不可用/异常时返回 (None, [])，由上层用 CV 兜底。
    """

    def __init__(self, weights: str = "", imgsz: int = 320,
                 conf: float = 0.15, iou: float = 0.25):
        # 实测坑（同 ScreenParser）：torch CPU + imgsz>=1280 + 多线程会段错误；
        # imgsz=320 + 单线程稳定。小恐龙游戏区本就窄，320 足够。
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self._weights_spec = weights
        self._model = None
        self.available = False
        self._tried = False
        self.load_error = ""

    def _resolve_weights(self) -> str:
        if self._weights_spec and os.path.isfile(self._weights_spec):
            return self._weights_spec
        local = _find_local_yoloe()
        if local:
            return local
        if self._weights_spec:
            # 用户显式指定了某权重名 → 让 ultralytics 走默认（可能联网下载，失败即降级）
            return self._weights_spec
        dl = _maybe_download()
        return dl

    def load(self) -> bool:
        """返回 True=深度学习可用；线程安全由调用方保证（单线程 demo）。"""
        if self._tried:
            return self.available
        self._tried = True
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            from ultralytics import YOLO
            p = self._resolve_weights()
            if not p or not os.path.isfile(p):
                self.load_error = "未找到 YOLOE 权重(放 blobs/weights/yoloe*.pt 即自动激活)"
                self.available = False
                return False
            self._model = YOLO(p)
            # 开放词汇：按名检测小恐龙三元素
            try:
                self._model.set_classes(list(_PROMPTS.values()))
            except Exception:
                pass  # 部分权重不支持 set_classes → 退化为免提示(内置词汇)
            self.available = True
        except Exception as e:
            self._model = None
            self.available = False
            self.load_error = f"{type(e).__name__}: {e}"
        return self.available

    def detect(self, scene: np.ndarray):
        """YOLOE 检测小恐龙元素。异常/不可用 → (None, [])。"""
        if not self.load():
            return (None, [])
        try:
            H, W = scene.shape[:2]
            from perception.device import predict   # 统一收敛 device，杜绝隐式选 CUDA
            results = predict(
                self._model, scene, imgsz=self.imgsz, conf=self.conf,
                iou=self.iou, verbose=False)
            dino_x = None
            obs: list = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for b, c, _cf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    x1, y1, x2, y2 = map(int, b.tolist())
                    try:
                        label = str(self._model.names[int(c)]).lower()
                    except Exception:
                        label = "object"
                    cx = (x1 + x2) / 2.0
                    if "dino" in label or "trex" in label or "trex" in label:
                        dino_x = cx
                    elif "bird" in label:
                        obs.append((float(x1), float(y2), True))
                    elif "cactus" in label or "plant" in label or "succulent" in label:
                        obs.append((float(x1), float(y2), False))
                    # 其他标签（含开放词汇误召回）忽略
            return (dino_x, obs)
        except Exception as e:
            # 单帧推理异常 → 降级 CV，不崩
            self.load_error = f"detect 异常: {type(e).__name__}: {e}"
            return (None, [])
