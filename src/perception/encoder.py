"""
encoder.py —— MGA ①→② 真实视觉编码器（产出投影层要的特征网格）
============================================================
把一张截图变成 (n_patches, enc_dim) 的特征网格，喂给 ② 投影层
(MLP/Q-Former) 做语义对齐 + 长度控制。

设计（与全系统一致，零依赖自检 + 满血运行）：
- backend 是字符串："screenparser" / "yolo" / "mage-vl" / None。
  真实模型**懒加载**（首次 encode 才 import+load），缺失即降级，绝不崩，
  不破坏 python src/main.py 的零依赖自检。
- ScreenParser/YOLO 产出的是"检测框"不是"特征张量"，最诚实的真实特征 =
  把检测框编成 (n_patches, enc_dim) 空间网格（每格占位数/中心/尺寸/置信），
  真反映当前 UI 布局，喂给 ② 投影层的是真实信号而非随机张量。
- mage-vl 这类主干直接吐特征张量：尽力取 hidden states 切格，失败降级。
- 没接/加载失败/无检出 → 降级「基于真实像素的确定性特征」（截图真参与计算，可复现）；
  最极端兜底仍是哈希，但正常走不到。

对应架构：
    ① 感知层(冻结编码器) ──encode()──▶ (n_patches, enc_dim) ──▶ ② 投影层
"""
from __future__ import annotations

import numpy as np
from projection.projector import VisionEncoder


class RealScreenEncoder(VisionEncoder):
    """真实截图 → 特征网格。

    backend 为字符串时懒加载真实模型（screenparser/yolo/mage-vl），把检测框/
    主干特征编成 (n_patches, enc_dim) 真实网格；加载失败或无检出则降级像素特征。
    backend 为可调用（旧接口）时直接当其为真骨干；backend=None 走像素降级。
    """

    def __init__(self, enc_dim: int = 64, n_patches: int = 16,
                 backend="screenparser", conf: float = 0.10, seed: int = 0):
        self.enc_dim, self.n_patches = enc_dim, n_patches
        self.backend = backend          # str / callable / None
        self.conf = conf
        # 懒加载状态：未加载前不碰任何重依赖
        self._model = None
        self._model_kind = None         # "yolo" / "mage-vl"
        self._loaded = False
        self._use_pixel = (backend is None)   # None 直接走像素
        self._load_err = None
        rng = np.random.default_rng(seed)
        # 像素降级用的固定基：把每个 patch 的标量强度投到 enc_dim 维方向
        self._basis = rng.standard_normal((n_patches, enc_dim)) * 0.1
        # 检测框→网格的固定线性映射基（6 维原始信号 → enc_dim，确定性，不训练）
        self._basis6 = rng.standard_normal((6, enc_dim)) * 0.1

    # -- 主入口 ------------------------------------------------------------
    def encode(self, image) -> np.ndarray:
        # 0) 旧接口兼容：backend 是可调用真骨干
        if callable(self.backend):
            try:
                feat = np.atleast_2d(self.backend(image))
                return self._fit_dim(feat)
            except Exception:
                return self._pixel_features(image)
        # 1) 字符串 backend：懒加载 + 真实模型编码（失败降级像素）
        if not self._use_pixel:
            if not self._loaded:
                self._lazy_load()
            if not self._use_pixel and self._model is not None:
                try:
                    return self._extract_from_model(image)
                except Exception as e:   # 单帧失败也降级，不中断闭环
                    self._load_err = e
                    self._use_pixel = True
        # 2) 像素降级（截图真参与计算，可复现）
        return self._pixel_features(image)

    # -- 懒加载真实模型（首次 encode 才执行）------------------------------
    def _lazy_load(self):
        self._loaded = True
        try:
            if self.backend in ("screenparser", "yolo"):
                from ultralytics import YOLO
                # 权重经 weights.resolve_weights 解析：本地优先，缺失走 HF 镜像
                # （github.com/HF 在本机不可达，直接给 YOLO 权重名会超时失败）
                from perception.weights import resolve_weights, BACKEND_WEIGHTS
                spec = BACKEND_WEIGHTS.get(self.backend, "yolov8n.pt")
                self._model = YOLO(resolve_weights(spec))
                self._model_kind = "yolo"
            elif self.backend == "mage-vl":
                from transformers import AutoModel
                # 占位权重：真实部署替换为 mage-vl 官方权重
                self._model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
                self._model_kind = "mage-vl"
            else:
                self._use_pixel = True
        except Exception as e:
            # 缺依赖（ultralytics/transformers 未装）或加载失败 → 降级像素，不崩
            self._use_pixel = True
            self._load_err = e

    # -- 真实模型 → 特征网格 ----------------------------------------------
    def _extract_from_model(self, image) -> np.ndarray:
        if self._model_kind == "yolo":
            # YOLO 接受路径或 numpy 数组；直接出检测框
            from perception.device import predict   # 统一收敛 device，杜绝隐式选 CUDA
            res = predict(self._model, image, conf=self.conf, verbose=False)[0]
            boxes = []
            if res.boxes is not None:
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0]) if b.conf is not None else 1.0
                    boxes.append((x1, y1, x2, y2, conf))
            arr = self._load_array(image)
            return self._encode_boxes_to_grid(boxes, arr.shape[:2])
        # mage-vl：主干吐特征张量，尽力取 hidden states 切格，失败由上层降级
        from PIL import Image
        import torch
        arr = self._load_array(image)
        pil = Image.fromarray(arr).convert("RGB")
        # 简化预处理：Resize 到 224，归一化到 [0,1]（真实部署用官方 processor）
        small = np.array(pil.resize((224, 224)), dtype=np.float32) / 255.0
        small = small.transpose(2, 0, 1)[None]   # (1,3,224,224)
        pixel_values = torch.from_numpy(small)
        out = self._model(pixel_values=pixel_values)
        hid = out.last_hidden_state[0].detach().cpu().numpy()  # (seq, hid)
        return self._fit_dim(hid[: self.n_patches])

    # -- 检测框 → 空间网格特征 --------------------------------------------
    def _encode_boxes_to_grid(self, boxes, image_shape) -> np.ndarray:
        """把检测框编成 (n_patches, enc_dim)：每格 = [占位数, Σ中心x, Σ中心y,
        Σ宽, Σ高, Σ置信]（占位的取均值），再固定线性映射到 enc_dim 维。
        boxes: list of (x1,y1,x2,y2,conf)；image_shape=(H,W) 用于归一化。"""
        H, W = image_shape[:2]
        gh = gw = int(round(self.n_patches ** 0.5))   # 16 → 4x4
        acc = np.zeros((gh, gw, 6), dtype=np.float32)
        for (x1, y1, x2, y2, conf) in boxes:
            cx, cy = (x1 + x2) / 2.0 / max(W, 1), (y1 + y2) / 2.0 / max(H, 1)
            w, h = (x2 - x1) / max(W, 1), (y2 - y1) / max(H, 1)
            ci = min(gh - 1, int(cy * gh))
            cj = min(gw - 1, int(cx * gw))
            acc[ci, cj, 0] += 1.0
            acc[ci, cj, 1] += cx
            acc[ci, cj, 2] += cy
            acc[ci, cj, 3] += w
            acc[ci, cj, 4] += h
            acc[ci, cj, 5] += conf
        occ = acc[..., 0:1]
        has = (occ > 0).astype(np.float32)
        mean = acc[..., 1:] / np.maximum(occ, 1.0)     # 占位的取均值，否则 0
        raw = np.concatenate([has, mean], axis=-1).reshape(-1, 6)  # (16,6)
        feat = raw @ self._basis6
        n = np.linalg.norm(feat)
        if n > 1e-12:
            feat = feat / n                       # 整网格单位范数，量纲稳
        return feat

    # -- 像素降级 ----------------------------------------------------------
    def _pixel_features(self, image) -> np.ndarray:
        try:
            arr = self._load_array(image)
            gray = arr[..., :1] if arr.ndim == 3 else arr
            if arr.ndim == 3:
                gray = gray.astype(np.float32).mean(axis=2)
            else:
                gray = gray.astype(np.float32)
            gh = gw = int(round(self.n_patches ** 0.5))   # 默认 16→4x4
            H, W = gray.shape
            bh, bw = max(1, H // gh), max(1, W // gw)
            ph, pw = bh * gh, bw * gw
            g = gray[:ph, :pw].reshape(gh, bh, gw, bw).mean(axis=(1, 3))  # (gh,gw)
            grid = g.reshape(-1)                            # (n_patches,)
            grid = grid / (np.linalg.norm(grid) + 1e-9)
            feat = grid[:, None] * self._basis              # (n_patches, enc_dim)
            return feat
        except Exception:
            return self._fallback_hash(image)

    # -- 工具 ---------------------------------------------------------------
    def _load_array(self, image):
        """numpy 数组直接用；路径尝试 PIL；其余抛错走兜底。"""
        if isinstance(image, np.ndarray):
            return image
        from PIL import Image  # 缺失即抛错 → 落像素兜底（但此处已是像素路径，故最终走 hash）
        return np.array(Image.open(image).convert("L"))

    def _fit_dim(self, feat: np.ndarray) -> np.ndarray:
        """维度对齐 + 形状保证：主干维度 ≠ 投影层 enc_dim 时过固定线性映射；
        末维强制对齐 enc_dim，行数不足补零、超出截断到 n_patches。"""
        feat = np.atleast_2d(feat)
        if feat.shape[1] != self.enc_dim:
            feat = self._align_dim(feat)
        if feat.shape[0] != self.n_patches:
            out = np.zeros((self.n_patches, self.enc_dim), dtype=feat.dtype)
            n = min(feat.shape[0], self.n_patches)
            out[:n] = feat[:n]
            return out
        return feat

    def _fallback_hash(self, image) -> np.ndarray:
        """最终兜底：基于图像标识的确定性特征（缺依赖/路径异常时也不崩）。"""
        rng = np.random.default_rng(abs(hash(str(image))) % (2 ** 32))
        return rng.standard_normal((self.n_patches, self.enc_dim))

    def _align_dim(self, feat: np.ndarray) -> np.ndarray:
        """主干特征维度 ≠ 投影层 enc_dim 时，固定线性映射对齐（确定性，不训练）。"""
        d = feat.shape[1]
        rng = np.random.default_rng(abs(hash(str((d, self.enc_dim)))) % (2 ** 32))
        A = rng.standard_normal((d, self.enc_dim))
        return feat @ A


def build_vision_encoder(real: bool = False, enc_dim: int = 64,
                         n_patches: int = 16, backend=None) -> VisionEncoder:
    """工厂：real=True → RealScreenEncoder（按 backend 串接真模型/像素降级）；
    real=False → 投影层自带的 MockVisionEncoder（零依赖确定性占位）。
    backend 默认随 real=True 走 "screenparser"。"""
    if real:
        b = backend if backend is not None else "screenparser"
        return RealScreenEncoder(enc_dim=enc_dim, n_patches=n_patches, backend=b)
    from projection.projector import MockVisionEncoder
    return MockVisionEncoder(enc_dim=enc_dim, n_patches=n_patches)


if __name__ == "__main__":
    # 自检：像素降级路径产出正确形状；与投影层对接
    from projection.projector import build_projector
    enc = build_vision_encoder(real=True, enc_dim=64, n_patches=16)
    proj = build_projector("mlp", enc_dim=64, out_dim=64)
    # 模拟一张真实截图（灰度 200x200），验证像素特征
    img = (np.arange(200 * 200) % 255).reshape(200, 200).astype("uint8")
    vf = enc.encode(img)
    lf = proj.forward(vf)
    print("真实像素截图 → vision_feats", vf.shape, "→ llm_feats", lf.shape)
    assert vf.shape == (16, 64) and lf.shape == (16, 64)
    # Mock 路径形状一致
    enc2 = build_vision_encoder(real=False, enc_dim=64, n_patches=16)
    vf2 = enc2.encode({"t": 1.0, "pos": [100, 100]})
    print("Mock 截图 → vision_feats", vf2.shape)
    assert vf2.shape == (16, 64)
    print("✅ encoder 自检通过：真实像素降级 / Mock 都产出 (n_patches, enc_dim) 特征网格")
