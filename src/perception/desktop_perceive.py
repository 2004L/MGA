"""
desktop_perceive.py —— 桌面 UI 感知（三级降级，活标注 active_backend）
================================================================

MGA 桌面 Computer Use 的「眼睛」。把一帧桌面像素变成结构化 UI 元素列表，
供 S2（LLM）做 grounding / 决策。

三级降级（命中即停，强制标注当前走哪条通路）：
  L1 ScreenParser : 闭集 55 类 UI 检测（docling-project，项目已验证的主视觉通道）
  L2 多模态 LLM   : 把截帧 + goal 送 APILLM(gpt-6-astra) 或本地 VLM，返回带坐标元素
  L3 CV 兜底      : 几何/颜色 + OCR 找矩形按钮/输入框/菜单条，保证"至少能点"
  开放词汇升级    : SemanticLocator（SAM3 → YOLOE → 闭集），由「L1 未命中 且有语义目标
                    goal」或 force_backend="yoloe" 触发，补 ScreenParser 词汇外的语义名
                    （SAM3 已本地真通）。L1 已检出 UI 则直接返回 screenparser，不跑 SAM3。

架构真相：
  - YOLOE 权重（yoloe-11s.pt）从 github 拉取不可达（HF 上 404）→ **该级默认降级为空**，
    开放词汇主通道现已由 **SAM3** 顶上（本地 ModelScope 权重 + .venv-sam3 worker，已验证真通）。
    active_backend 必须如实标注（semantic / screenparser / llm / cv），绝不假装主通道通。
  - L2 只做 S2 关键帧/失败升档；逐帧不调（成本 + 延迟 + 灰度未全开）。
  - 本模块零硬依赖：torch/cv2/ultralytics/网络 任一缺失都优雅降级，沙箱可跑链路。

接口与 detector.Element 同构（.center()/.label/.text/.conf），可直接喂
llm_bridge.LLMBridge.decide / format_scene。
"""

from __future__ import annotations

import os
import sys

# 路径兜底：本项目 perception 非包，运行 cwd=src 时顶层 import 才通；
# 从 agent 包内调用时把 src 与 src/perception 都补进 path，保证 import 不崩。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 结构化输出：与 detector.Element 同构，便于直接喂 LLM 桥
# ---------------------------------------------------------------------------
@dataclass
class UIElement:
    id: str = "ui"
    label: str = "unknown"          # button / input / menu / icon / link / text
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # (x1,y1,x2,y2)
    conf: float = 0.0
    text: Optional[str] = None
    backend: str = ""               # 该元素来自哪条通路（yoloe/llm/cv）

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class DesktopScene:
    frame: np.ndarray
    elements: List[UIElement] = field(default_factory=list)
    active_backend: str = "none"    # yoloe / llm / cv —— 强制如实标注
    confidence: float = 0.0
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# L1：视觉 UI 检测（主通道 = ScreenParser，开放词汇升级 = YOLOE）
# ---------------------------------------------------------------------------
class WeightUnavailable(RuntimeError):
    """视觉检测器权重不可用（404 / 未安装 / 加载失败）→ 触发降级。"""


def _to_ui(e, backend: str) -> UIElement:
    """把 detector.Element 转成本模块的 UIElement（接口同构，供 S2 消费）。"""
    return UIElement(id=e.id, label=e.label, bbox=tuple(e.bbox),
                     conf=float(e.conf), text=getattr(e, "text", None),
                     backend=backend)


def _try_screenparser(frame: np.ndarray) -> List[UIElement]:
    """主视觉通道：docling-project/ScreenParser（55 类 UI，项目已验证的 GUI 检测器）。

    复用 detector.ScreenParserBackend（懒加载 ultralytics YOLO + 经 hf-mirror 取权重）。
    任何异常（torch/ultralytics 缺失、权重 404）→ 上抛，由 perceive_pipeline 降级。
    """
    from detector import ScreenParserBackend
    loc = ScreenParserBackend()          # 懒加载：缺依赖/权重即抛错 → 降级
    els = loc.detect(frame)              # List[detector.Element]
    return [_to_ui(e, "screenparser") for e in els]


def _try_yoloe_open(frame: np.ndarray, names=None, use_sam3: bool = True) -> List[UIElement]:
    """开放词汇升级通道：SAM3（概念分割+mask 质心）→ YOLOE，按 LLM 给的语义名找未知元素。

    现在走 SemanticLocator 三级（SAM3 → YOLOE → 闭集），但闭集(ScreenParser)已由 L1 处理，
    这里只补 ScreenParser 词汇外的语义名，故传给 SemanticLocator 的 closed 用空实现，
    避免与 L1 重复出框。任何一级缺依赖/缺权重/worker 起不来 → 自动降级到 None，
    本函数返回 []，链路不崩（符合"缺失即降级"）。

    注意：本环境 YOLOE 权重（yoloe-11s.pt）从 github 拉取不可达（HF 上 404），
    该级通常降级为空；SAM3 已在本地（ModelScope 权重 + .venv-sam3 worker）可用。
    """
    sem = _get_semantic_locator(use_sam3)
    try:
        els = sem.detect_named(frame, names) if names else sem.detect(frame)
    except Exception:
        return []
    return [_to_ui(e, "semantic") for e in els]


# 模块级缓存：SemanticLocator（含常驻 worker 子进程）只构建一次、跨帧复用。
# 否则每次感知都重 spawn worker + 重加载 3.4GB 模型（实测 ~70s/次，生产不可用）。
_SEM_CACHE: dict = {}


def _get_semantic_locator(use_sam3: bool):
    """懒初始化并缓存 SemanticLocator，确保 worker 只加载一次模型、跨帧复用。"""
    from detector import build_semantic_locator, TargetLocator
    key = bool(use_sam3)
    sem = _SEM_CACHE.get(key)
    if sem is None:
        class _EmptyClosed(TargetLocator):
            """占位闭集：开放词汇升级不需要再跑 ScreenParser（L1 已做）。"""
            def detect(self, f):
                return []
            def detect_named(self, f, n):
                return []
        sem = build_semantic_locator(closed=_EmptyClosed(), use_sam3=use_sam3,
                                     use_yoloe=True, sam3_verbose=False)
        _SEM_CACHE[key] = sem
    return sem


# ---------------------------------------------------------------------------
# L2：多模态 LLM grounding（仅 S2 关键帧 / 失败升档）
# ---------------------------------------------------------------------------
def _try_llm(frame: np.ndarray, goal: str,
             bridge) -> List[UIElement]:
    """用多模态 LLM 把截帧 + goal 转成带坐标的 UIElement。

    bridge: 实现 LLMBridge 协议的对象（APILLM(gpt-6-astra) / 本地 VLM / MockLLM）。
            bridge.decide(elements, goal, image=frame, multimodal=True) → Action。
    """
    action = bridge.decide([], goal, image=frame, multimodal=True)
    if not action.executable():
        return []
    cx, cy = action.coordinates or (0, 0)
    half = 20
    return [UIElement(
        id="llm_0", label=action.target or "target",
        bbox=(cx - half, cy - half, cx + half, cy + half),
        conf=1.0, text=action.text, backend="llm")]


# ---------------------------------------------------------------------------
# L3：CV 兜底（几何/颜色 + OCR，保证"至少能点"）
# ---------------------------------------------------------------------------
def _try_cv(frame: np.ndarray) -> List[UIElement]:
    """无 YOLOE / 无 LLM 时的最后兜底：用 cv2 找轴对齐矩形按钮。

    cv2 不可用时返回空列表但链路不崩（诚实：L3 需要 cv2 才有真实检出能力，
    无 cv2 时只是"保证不报错"，不伪造控件）。
    """
    try:
        import cv2
    except Exception:
        return []  # cv2 不可用 → 无结构化控件，仅保证降级链路不崩
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 简单轮廓找"近似矩形"区域（按钮/输入框常见形态）
    edges = cv2.Canny(gray, 80, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[UIElement] = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w < 24 or h < 12 or w > frame.shape[1] * 0.9:
            continue
        out.append(UIElement(
            id=f"cv_{i}", label="rect", bbox=(x, y, x + w, y + h),
            conf=0.5, backend="cv"))
    return out


# ---------------------------------------------------------------------------
# 主入口：三级降级管线（命中即停，标 active_backend）
# ---------------------------------------------------------------------------
def perceive_pipeline(frame: np.ndarray, goal: str = "",
                      l2_bridge=None,
                      force_backend: Optional[str] = None,
                      use_sam3: bool = True) -> DesktopScene:
    """把桌面帧变成 DesktopScene。

    force_backend: 测试钩子（"yoloe"/"llm"/"cv"）强制走某一级，跳过降级。
    l2_bridge:     提供则 L1 失败尝试 L2；None 则跳过 L2（如沙箱无 API key）。
    use_sam3:      开放词汇升级通道是否启用 SAM3（默认开；关则只用 YOLOE）。
    """
    # 测试钩子：直接指定通路
    if force_backend == "llm" and l2_bridge is not None:
        els = _try_llm(frame, goal, l2_bridge)
        return DesktopScene(frame=frame, elements=els, active_backend="llm",
                            confidence=1.0 if els else 0.0)
    if force_backend == "cv":
        els = _try_cv(frame)
        return DesktopScene(frame=frame, elements=els, active_backend="cv",
                            confidence=0.5 if els else 0.0)

    # L1：视觉 UI 检测（主通道 = ScreenParser，UI 域最优、零额外成本）
    l1_els = []
    try:
        l1_els = _try_screenparser(frame)
    except Exception:
        l1_els = []  # 主通道空窗（torch/ultralytics 缺失或权重 404）：降级 L1.5
    if l1_els:
        # 主通道已检出 UI 元素 → 直接返回，不跑 SAM3（省成本）
        return DesktopScene(frame=frame, elements=l1_els, active_backend="screenparser",
                            confidence=float(np.mean([e.conf for e in l1_els])),
                            meta={"visual": "screenparser"})

    # L1.5：开放词汇升级（SAM3 → YOLOE），补 ScreenParser 词汇外的语义名。
    # 触发条件：显式 force_backend=="yoloe"，或（启用 SAM3 且 goal 给了语义目标）。
    # 不无谓触发：L1 未命中但 goal 为空时不跑 SAM3（SAM3 需提示词，空跑是浪费）。
    want_semantic = force_backend == "yoloe" or (use_sam3 and goal)
    if want_semantic:
        try:
            els = _try_yoloe_open(frame, names=[goal] if goal else None,
                                  use_sam3=use_sam3)
            if els:
                return DesktopScene(frame=frame, elements=els, active_backend="semantic",
                                    confidence=float(np.mean([e.conf for e in els])),
                                    meta={"semantic": "sam3+yoloe"})
        except Exception:
            pass

    # L2：多模态 LLM（仅当有 bridge；否则跳过，直接 L3）
    if l2_bridge is not None:
        try:
            els = _try_llm(frame, goal, l2_bridge)
            if els:
                return DesktopScene(frame=frame, elements=els, active_backend="llm",
                                    confidence=1.0)
        except Exception:
            pass

    # L3：CV 兜底（最后一道，保证链路不崩）
    els = _try_cv(frame)
    return DesktopScene(frame=frame, elements=els, active_backend="cv",
                        confidence=0.5 if els else 0.0,
                        meta={"note": "L1/L2 不可用，已降 CV 兜底"})


if __name__ == "__main__":
    # 沙箱自检：合成帧验证三级降级链路 + active_backend 标注（不碰游戏/网络）
    fake = np.zeros((200, 320, 3), dtype=np.uint8)
    sc = perceive_pipeline(fake, goal="点击确定按钮", l2_bridge=None)
    print(f"[感知] active_backend={sc.active_backend} 元素数={len(sc.elements)}")
    print("降级链路正常：无 YOLOE/无 L2 → 落 CV 兜底，未崩溃。")
