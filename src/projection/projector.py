"""
src/projection/projector.py —— MGA ② 投影层（MLP / Q-Former，自研）

在 MGA 架构里的位置：
    感知层(冻结编码器 SigLIP/DINOv2/SenseVoice)  ──▶  [本层]  ──▶  MiniMind 大脑层
    投影层只干两件事（报告 1.2 / 用户方案）：
        1) 语义对齐：把编码器特征映射到 LLM 能懂的语义空间，消跨模态鸿沟
        2) 长度控制：把不定长视觉/音频特征压成固定长度表示，控 LLM 输入 token 数
                      （这点和 token 经济直接相关——token 少=省钱=System1 倾向）

设计原则（与系统一致，零污染）：
    - 默认 numpy 自研，零重依赖即可训练跑通（先跑通后优化）。
    - 真编码器/真 LLM 用懒加载（torch/transformers），缺失即报错由上层降级回 Mock。
    - 可插拔：MLPProjector 与 QFormerProjector 同接口，阶段一用 MLP，阶段二换 Q-Former。

⚠️ 两个 embedding 必须分清（用户方案第五节）：
    - M5 记忆系统的 Embedder：文本→向量，用于「检索」，轻量模型(sentence-transformers 等)
    - 本层 LLMSpaceEmbedder：文本/视觉特征→「LLM 输入空间」，用于「对齐训练」
    两者各司其职、互不影响。本文件里的 MockLLMSpaceEmbedder 只是 LLM 输入空间的占位，
    与 knowledge/graphrag.py 的 MockEmbedder 没有任何耦合。

训练策略（用户方案第四节，冻结模式）：
    train_projector() 只更新投影层参数，视觉编码器与 LLM 文本空间都冻结。
    对齐损失支持两种：'infonce'(InfoNCE/ITC，BLIP-2/LLaVA 同款，生产级需大 batch)
    与 'mse'(归一化 MSE 回归，极小规模/小模型下可稳定收敛，自检默认用它做最小演示)。
    跑通冻结模式后再考虑联合微调。
"""

from __future__ import annotations

import os
import re
import json
import pickle
from typing import List, Tuple, Optional, Callable

import numpy as np


# ---------------------------------------------------------------------------
# 0. 可插拔原语：视觉编码器 & LLM 输入空间嵌入（都独立于 M5 记忆 Embedder）
# ---------------------------------------------------------------------------
class VisionEncoder:
    def encode(self, image) -> np.ndarray:
        """返回特征序列 (n_patches, enc_dim)。"""
        raise NotImplementedError


class MockVisionEncoder(VisionEncoder):
    """零依赖视觉编码器占位：把 image(任意可哈希对象) 哈希成确定性特征网格。
    真部署换成 SigLIP/DINOv2（懒加载 torch+transformers），接口不变。"""

    def __init__(self, enc_dim: int = 64, n_patches: int = 16, seed: int = 0):
        self.enc_dim = enc_dim
        self.n_patches = n_patches

    def encode(self, image) -> np.ndarray:
        key = str(image)
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 32))
        return rng.standard_normal((self.n_patches, self.enc_dim))


class LLMSpaceEmbedder:
    def encode(self, text: str) -> np.ndarray:
        """文本 → LLM 输入空间向量 (out_dim,)。这是对齐训练的目标空间。"""
        raise NotImplementedError


class MockLLMSpaceEmbedder(LLMSpaceEmbedder):
    """零依赖 LLM 输入空间占位：字符+词哈希成向量，L2 归一化（与 M5 的 Embedder 无关）。
    真部署换成 LLM 的 text embedding（如 text-embedding-3-small / sentence-transformers）。"""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        for ch in text:
            vec[hash(ch) % self.dim] += 1.0
        for tok in re.findall(r"\w+", text):
            vec[hash(tok) % self.dim] += 2.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# 1. 投影器基类 + 对比损失（ITC / InfoNCE）
# ---------------------------------------------------------------------------
def _infonce_loss(Y: np.ndarray, T: np.ndarray, tau: float = 0.07):
    """对比损失：每行 Y[i] 应与 T[i] 最相似（对角线为正样本），其余为负样本。
    返回 (loss, dL/dY)。Y,T 形状均为 (B, out)，已建议 L2 归一化。"""
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    S = Yn @ Tn.T / tau                                   # (B,B)
    m = S.max(axis=1, keepdims=True)
    eS = np.exp(S - m)
    row_sum = eS.sum(axis=1, keepdims=True)
    logsum = np.log(row_sum) + m
    pos = S[np.arange(S.shape[0]), np.arange(S.shape[0])]
    loss = float(np.mean(logsum.squeeze() - pos))
    P = eS / row_sum                                     # softmax 概率
    dS = P.copy()
    dS[np.arange(S.shape[0]), np.arange(S.shape[0])] -= 1.0
    dS /= S.shape[0]
    dYn = (1.0 / tau) * (dS @ Tn)                        # (B,out)
    norms = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9
    dY = np.zeros_like(Yn)
    for i in range(Y.shape[0]):
        yn_i = Yn[i]
        dY[i] = (dYn[i] - yn_i * (yn_i @ dYn[i])) / norms[i]   # d yn / d Y 链式
    return loss, dY


def _mse_loss(Y: np.ndarray, T: np.ndarray):
    """归一化 MSE 对齐损失：比 InfoNCE 在「极小规模 + 小模型」下更易收敛，同样是对齐训练。
    Y,T: (B,out)。先各自 L2 归一化，再求 ||Yn-Tn||^2 均值；返回 (loss, dL/dY)。
    生产级用 InfoNCE(ITC)，本函数仅用于可在笔记本上跑通的最小演示。"""
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    d = Yn - Tn
    loss = float(np.mean(np.sum(d * d, axis=1)))
    dYn = 2.0 * d / Y.shape[0]                        # dL/dYn
    norms = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9
    dY = np.zeros_like(Yn)
    for i in range(Y.shape[0]):
        yn_i = Yn[i]
        dY[i] = (dYn[i] - yn_i * (yn_i @ dYn[i])) / norms[i]   # 链式法则到原始 Y
    return loss, dY


class Projector:
    def forward(self, features: np.ndarray) -> np.ndarray:
        """features: (n_patches, enc_dim) → 投影后的语义向量。
        MLP 返回 (n_patches, out_dim)（不压缩长度）；
        Q-Former 返回 (n_queries, out_dim)（固定长度，做长度控制）。"""
        raise NotImplementedError

    def parameters(self) -> List[np.ndarray]:
        raise NotImplementedError

    def train_step(self, dY: np.ndarray):
        """用对比损失对投影器参数的梯度 dY(B,out) 做一步更新（冻结模式只动这里）。"""
        raise NotImplementedError

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self._state(), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self._load_state(pickle.load(f))

    def _state(self) -> dict:
        raise NotImplementedError

    def _load_state(self, state: dict):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. MLP 投影器（阶段一，LLaVA 同款 2 层：Linear-ReLU-Linear，numpy 自研可训练）
# ---------------------------------------------------------------------------
class MLPProjector(Projector):
    def __init__(self, enc_dim: int, out_dim: int, hidden_dim: int = 128,
                 lr: float = 1e-2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.enc_dim, self.out_dim, self.hidden_dim, self.lr = enc_dim, out_dim, hidden_dim, lr
        # LLaVA 风格：x→W1→ReLU→W2→y；缩放初始化稳定训练
        self.W1 = rng.standard_normal((enc_dim, hidden_dim)) * (2.0 / enc_dim) ** 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, out_dim)) * (2.0 / hidden_dim) ** 0.5
        self.b2 = np.zeros(out_dim)
        self._cache: List[Tuple[np.ndarray, np.ndarray]] = []

    def forward(self, features: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(features)
        h = np.maximum(0.0, x @ self.W1 + self.b1)        # ReLU
        y = h @ self.W2 + self.b2
        self._cache.append((x, h))                         # 存激活供反向
        return y

    def parameters(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def train_step(self, dY: np.ndarray):
        if not self._cache:
            return
        x, h = self._cache.pop()                            # 与最近一次 forward 配对
        # 训练信号是 pooled (1,out)，均匀回传到每个 patch（mean 的反向）
        n = x.shape[0]
        dY = np.repeat(np.atleast_2d(dY) / n, n, axis=0)    # (n_patches, out)
        dW2 = h.T @ dY
        db2 = dY.sum(0)
        dH = (dY @ self.W2.T) * (h > 0)                     # ReLU 梯度
        dW1 = x.T @ dH
        db1 = dH.sum(0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def _state(self) -> dict:
        return {"kind": "mlp", "W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2,
                "enc_dim": self.enc_dim, "out_dim": self.out_dim,
                "hidden_dim": self.hidden_dim, "lr": self.lr}

    def _load_state(self, s: dict):
        for k in ("W1", "b1", "W2", "b2", "enc_dim", "out_dim", "hidden_dim", "lr"):
            setattr(self, k, s[k])


# ---------------------------------------------------------------------------
# 3. Q-Former 投影器（阶段二，长度控制：固定 n_queries 输出，跨模态对齐强）
#    这里是「最小可跑」numpy 版：可学习 query 令牌 + 单层交叉注意力。
#    生产级应直接用 HuggingFace BLIP-2 的 QFormerConfig 复用（含 2 阶段 ITC/ITM/ITG）。
# ---------------------------------------------------------------------------
class QFormerProjector(Projector):
    def __init__(self, enc_dim: int, out_dim: int, n_queries: int = 8,
                 lr: float = 1e-2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.enc_dim, self.out_dim, self.n_queries, self.lr = enc_dim, out_dim, n_queries, lr
        self.queries = rng.standard_normal((n_queries, out_dim)) * 0.1
        self.Wq = rng.standard_normal((out_dim, out_dim)) * 0.1
        self.Wk = rng.standard_normal((enc_dim, out_dim)) * 0.1
        self.Wv = rng.standard_normal((enc_dim, out_dim)) * 0.1
        self.Wout = rng.standard_normal((out_dim, out_dim)) * 0.1
        self.bout = np.zeros(out_dim)
        self._cache: list = []

    def forward(self, features: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(features)                        # (P, enc_dim)
        Q = self.queries @ self.Wq                         # (N, out)
        K = x @ self.Wk                                    # (P, out)
        V = x @ self.Wv                                    # (P, out)
        A = Q @ K.T / (self.out_dim ** 0.5)                # (N, P) 注意力
        A = np.exp(A - A.max(axis=1, keepdims=True))
        A = A / A.sum(axis=1, keepdims=True)
        ctx = A @ V                                        # (N, out)
        out = ctx @ self.Wout + self.bout                  # (N, out) ★ 固定长度 N
        self._cache.append((x, Q, K, V, A, ctx))
        return out

    def parameters(self) -> List[np.ndarray]:
        return [self.queries, self.Wq, self.Wk, self.Wv, self.Wout, self.bout]

    def train_step(self, dY: np.ndarray):
        """简化训练：学 Wout/bout/queries/Wv（交叉注意力当作冻结特征提取器）。
        生产级 Q-Former 还会反向传播过注意力与 key 投影，并跑 2 阶段对齐。"""
        if not self._cache:
            return
        x, Q, K, V, A, ctx = self._cache.pop()
        N = self.n_queries
        dY = np.atleast_2d(dY)
        # out 在 trainer 里被 mean 成 (1,out)，反传回每个 query 要除以 N
        d_out = np.repeat(dY / N, N, axis=0)               # (N, out)
        dWout = ctx.T @ d_out
        dbout = d_out.sum(0)
        d_ctx = d_out @ self.Wout.T                        # (N, out)
        dA = d_ctx @ V.T                                   # (N, P)
        dQ = dA @ K                                        # (N, out)
        d_queries = dQ @ self.Wq.T                         # (N, out)
        dV = A.T @ d_ctx                                   # (P, out)
        dWv = x.T @ dV                                     # (enc, out)
        self.Wout -= self.lr * dWout
        self.bout -= self.lr * dbout
        self.queries -= self.lr * d_queries
        self.Wv -= self.lr * dWv

    def _state(self) -> dict:
        return {"kind": "qformer", "queries": self.queries, "Wq": self.Wq,
                "Wk": self.Wk, "Wv": self.Wv, "Wout": self.Wout, "bout": self.bout,
                "enc_dim": self.enc_dim, "out_dim": self.out_dim,
                "n_queries": self.n_queries, "lr": self.lr}

    def _load_state(self, s: dict):
        for k in ("queries", "Wq", "Wk", "Wv", "Wout", "bout",
                  "enc_dim", "out_dim", "n_queries", "lr"):
            setattr(self, k, s[k])


# ---------------------------------------------------------------------------
# 3.5 配置：config.json 的 training.loss_mode 映射到 loss_kind（demo→mse / production→infonce）
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config.json")


def load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def resolve_loss_kind(explicit: Optional[str] = None) -> str:
    """explicit 优先；否则读 config.json 的 training.loss_mode：
    'demo'（默认）→ 归一化 MSE 回归；'production' → InfoNCE(ITC)。"""
    if explicit in ("mse", "infonce"):
        return explicit
    mode = load_config().get("projection", {}).get("training", {}).get("loss_mode", "demo")
    return "infonce" if mode == "production" else "mse"


# ---------------------------------------------------------------------------
# 4. 冻结模式训练循环：只更新投影层，编码器与文本空间都冻结
# ---------------------------------------------------------------------------
def train_projector(projector: Projector, vision_encoder: VisionEncoder,
                    text_embedder: LLMSpaceEmbedder,
                    pairs: List[Tuple[object, str]],
                    epochs: int = 100, tau: float = 0.07,
                    loss_kind: Optional[str] = None, verbose: bool = True) -> List[float]:
    """冻结模式对齐训练：只更新投影层，编码器与文本空间都冻结。
    loss_kind: 默认 None → 读 config.json 的 training.loss_mode（demo→mse / production→infonce）；
        也可显式传 'mse'（归一化回归，极小规模可收敛）或 'infonce'（InfoNCE/ITC，生产级需大 batch）。
    每 epoch 整批作一次对齐（mse 全批回归；infonce 含 batch 内负样本）。"""
    loss_kind = resolve_loss_kind(loss_kind)
    if loss_kind == "mse":
        loss_fn = _mse_loss
    elif loss_kind == "infonce":
        loss_fn = lambda Y, T: _infonce_loss(Y, T, tau)
    else:
        raise ValueError(f"未知损失: {loss_kind}")
    losses = []
    for ep in range(epochs):
        # 前向：所有图 → 投影 → 沿 token 轴 mean 成 (1,out) 对齐向量
        Ys, Ts = [], []
        for img, cap in pairs:
            Y = projector.forward(vision_encoder.encode(img))
            Ys.append(np.mean(Y, axis=0))                  # (out,)
            Ts.append(np.atleast_2d(text_embedder.encode(cap))[0])   # (out,)
        Yb = np.stack(Ys)                                 # (B, out)
        Tb = np.stack(Ts)                                 # (B, out)
        loss, dYb = loss_fn(Yb, Tb)
        # 反向：每样本配对其 forward 缓存，做一步 SGD
        for i in range(len(pairs)):
            projector.train_step(dYb[i:i + 1])
        losses.append(loss)
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"  epoch {ep:>3}  loss={loss:.4f}")
    return losses


# ---------------------------------------------------------------------------
# 5. 工厂：默认 MLP（阶段一），阶段二切 qformer，零改动切换
# ---------------------------------------------------------------------------
def build_projector(kind: str = "mlp", enc_dim: int = 64, out_dim: int = 64,
                    **kw) -> Projector:
    if kind == "mlp":
        return MLPProjector(enc_dim, out_dim, hidden_dim=kw.get("hidden_dim", 128),
                            lr=kw.get("lr", 1e-2))
    if kind == "qformer":
        return QFormerProjector(enc_dim, out_dim, n_queries=kw.get("n_queries", 8),
                                lr=kw.get("lr", 1e-2))
    raise ValueError(f"未知投影器类型: {kind}")


# ---------------------------------------------------------------------------
# 6. 自检：MLP 对齐训练 / Q-Former 长度控制 / 可插拔 / 持久化
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    os.makedirs("blobs", exist_ok=True)
    print("=== ② 投影层自检（numpy 自研，零重依赖，冻结模式对齐训练）===")

    enc_dim = out_dim = 64
    labels = [f"concept_{i}" for i in range(24)]      # 合成图文概念
    pairs = [(l, l) for l in labels]
    rng = np.random.default_rng(7)
    G = rng.standard_normal((enc_dim, out_dim)) * 0.5  # 隐藏真映射：语义→视觉（仅造数据用）

    class SyntheticPairedEncoder(VisionEncoder):
        """合成配对编码器：图像特征 = 文本语义向量经固定变换 G + 微噪声。
        这样『存在』真实对齐，投影器应能学到把它还原回 LLM 文本空间。"""
        def __init__(self, n_patches: int = 16, noise: float = 0.02):
            self.n_patches, self.noise = n_patches, noise
        def encode(self, image):
            t = MockLLMSpaceEmbedder(out_dim).encode(str(image))   # 语义向量
            base = t @ G.T                                         # (enc_dim,)
            return np.tile(base, (self.n_patches, 1)) + self.noise * rng.standard_normal(
                (self.n_patches, enc_dim))

    vis = SyntheticPairedEncoder(n_patches=16, noise=0.02)
    txt = MockLLMSpaceEmbedder(out_dim)

    def align_diag(projector) -> float:
        s = []
        for img, cap in pairs:
            y = np.mean(projector.forward(vis.encode(img)), axis=0)
            t = txt.encode(cap)
            s.append(float(y @ t / (np.linalg.norm(y) + 1e-9) / (np.linalg.norm(t) + 1e-9)))
        return float(np.mean(s))

    # --- 阶段一：MLP 投影器，冻结模式对齐训练（归一化 MSE，可收敛）---
    print("\n--- 阶段一：MLP 投影器（LLaVA 2层，冻结模式对齐训练）---")
    mlp = build_projector("mlp", enc_dim, out_dim)
    before = align_diag(mlp)
    losses = train_projector(mlp, vis, txt, pairs, epochs=200, loss_kind="mse", verbose=True)
    after = align_diag(mlp)
    print(f"图文对齐余弦：训练前={before:.3f} → 训练后={after:.3f}")
    assert losses[-1] < losses[0], "❌ MLP 对齐损失未下降"
    assert after > before + 0.1, "❌ MLP 对齐质量未明显提升"
    print("✅ MLP 投影器：冻结模式对齐训练通过（损失下降、图文对齐提升）")

    # --- 阶段二预览：Q-Former 长度控制（固定 n_queries）+ 可训练 ---
    print("\n--- 阶段二预览：Q-Former 长度控制（固定输出长度）---")
    qf = build_projector("qformer", enc_dim, out_dim, n_queries=8)
    for np_ in (4, 16, 64):
        feat = MockVisionEncoder(enc_dim=enc_dim, n_patches=np_).encode("cat")
        out = qf.forward(feat)
        print(f"  输入 patches={np_:>2} → 输出形状={out.shape}（固定 {out.shape[0]}）")
        assert out.shape[0] == 8, "❌ Q-Former 输出长度应固定为 n_queries"
    qloss = train_projector(qf, vis, txt, pairs, epochs=120, loss_kind="mse", verbose=False)
    print(f"  Q-Former 训练损失：{qloss[0]:.4f} → {qloss[-1]:.4f}")
    assert qloss[-1] < qloss[0], "❌ Q-Former 损失未下降"
    print("✅ Q-Former 投影器：长度控制（固定 N）+ 冻结模式可训练通过")

    # --- 可插拔：同一段训练代码，换 kind 即用 ---
    print("\n--- 可插拔验证：build_projector 一行切换 MLP/Q-Former ---")
    for kind in ("mlp", "qformer"):
        p = build_projector(kind, enc_dim, out_dim)
        _ = p.forward(vis.encode("concept_0"))
        assert hasattr(p, "train_step") and hasattr(p, "parameters")
    print("✅ 可插拔：MLP 与 Q-Former 同接口，零改动切换")

    # --- 与 M5 记忆 Embedder 区分（仅注释，不耦合）---
    print("\n[注] 本层 LLMSpaceEmbedder = LLM 输入空间对齐；与 M5 memory_system/"
          "knowledge/graphrag 的检索 Embedder 相互独立，互不调用。")

    # --- 生产级损失说明 + InfoNCE(ITC) 可用性烟测（不要求极小规模收敛）---
    print("\n[注] 生产级对齐用 InfoNCE(ITC/ITM/ITG, BLIP-2/Q-Former 同款)；"
          "本自检默认归一化 MSE 仅为『极小规模可收敛』最小演示。InfoNCE 已内置：")
    _ = train_projector(build_projector("mlp", enc_dim, out_dim), vis, txt, pairs,
                        epochs=2, loss_kind="infonce", verbose=False)
    print("  ✅ InfoNCE(ITC) 损失可计算、梯度可回传（大 batch/大模型下即生产路径）")

    # --- 持久化 ---
    mlp.save("blobs/projector_mlp.pkl")
    mlp2 = build_projector("mlp", enc_dim, out_dim)
    mlp2.load("blobs/projector_mlp.pkl")
    assert np.allclose(mlp2.W1, mlp.W1), "❌ 投影器参数未正确恢复"
    print("✅ 持久化往返 OK（投影器参数可存可载）")

    print("\n投影层自检全部通过：语义对齐(MLP训练) + 长度控制(Q-Former固定N) + 可插拔 + 持久化。")
