"""MGA ② 投影层：连接感知编码器与 LLM 大脑的桥梁（MLP / Q-Former，自研）。

语义对齐：把编码器特征映射到 LLM 能懂的语义空间；
长度控制：把不定长视觉/音频特征压成固定长度表示（控 LLM 输入 token 数）。
"""

from .projector import (
    VisionEncoder, MockVisionEncoder, LLMSpaceEmbedder, MockLLMSpaceEmbedder,
    Projector, MLPProjector, QFormerProjector,
    _infonce_loss, _mse_loss, train_projector, build_projector,
)

__all__ = [
    "VisionEncoder", "MockVisionEncoder", "LLMSpaceEmbedder", "MockLLMSpaceEmbedder",
    "Projector", "MLPProjector", "QFormerProjector",
    "_infonce_loss", "_mse_loss", "train_projector", "build_projector",
]
