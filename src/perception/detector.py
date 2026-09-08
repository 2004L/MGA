"""
detector.py —— MGA M3「精准目标定位器」（开箱集成 + 满血运行版）

设计原则（学泽定，2026-09-02 修正）：
- 能力成熟且重造无意义处直接集成：YOLO/ScreenParser 检测器、OCR、UIA。
- 仅核心差异点自研：与预判帧 / System1 / token 经济 / 编排逻辑的咬合。
- 集成必须带降级兜底 + 阈值调优，确保满血运行（任何一级不可用就跳过，不崩）。

满血运行要点（写进代码，不是口头）：
1. 阈值调优：UI 场景 conf=0.10，宁可多框不要漏框（漏框才致命）。
2. 三级降级：YOLO 检测 → OCR 补文字 → UIA 语义定位（任一级缺失自动跳过）。
3. 双通道：DOM 快通道优先（原生 App），缺关键元素自动降级视觉通道。
4. 触发式调用：定位器只在「预判帧偏差超阈值 → 关键帧」时才被调用，
   平日 System1 纯数学运行，定位器零开销。
5. 结构化上下文：检测结果以 bbox 列表注入 LLM，让 3B 模型在 GUI 任务超 72B。

对应架构：
    ① 感知层（视觉）   ← ScreenParser 几何定位
    ② 投影层           ← Element.to_context() 把几何变 LLM 可读文本
    ⑥ 执行层 CU        ← 用检测到的坐标点击（computer_use 模块）
   M3 定位器横跨 ①②⑥，是「像素 → 结构化元素」的工业级实现。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np  # read_lines/_readtext 在模块级用到 np（截图数组），集中导入避免局部缺失


# ---------------------------------------------------------------------------
# 1. 结构化元素：像素 → LLM 可读语言的核心载体
# ---------------------------------------------------------------------------
@dataclass
class Element:
    id: str                 # 持久追踪 ID（接 Chat-Scene 思路，供 System1 按 ID 跟踪）
    label: str              # 类别：Button / Text Input / Slider ...
    bbox: tuple             # (x1, y1, x2, y2) 像素坐标
    conf: float = 1.0
    text: Optional[str] = None   # OCR 填充：按钮上写的字（"确定"/"取消"）
    text_src: str = "ocr"        # 文字来源：ocr / uia（UIA 零误差，优先用于出题）

    def center(self) -> tuple:
        return ((self.bbox[0] + self.bbox[2]) // 2,
                (self.bbox[1] + self.bbox[3]) // 2)

    def to_context(self, idx: int) -> str:
        txt = f"「{self.text}」" if self.text else ""
        return f"{idx}. {self.label}{txt} 位于 {self.center()}"


# ---------------------------------------------------------------------------
# 2. 定位器抽象（角色，不绑任何框架）
# ---------------------------------------------------------------------------
class TargetLocator(ABC):
    @abstractmethod
    def detect(self, frame) -> List[Element]:
        """frame: 截图路径(str) 或 numpy 数组。返回结构化元素列表。"""
        ...


# ---------------------------------------------------------------------------
# 3. 第一级：ScreenParser / YOLO 视觉检测（开箱集成，懒加载）
# ---------------------------------------------------------------------------
class ScreenParserBackend(TargetLocator):
    """基于 docling-project/ScreenParser（YOLO11-Large, 25.4M 参数）。
    2540 万参数、77.1 万截图微调、55 类 UI 组件。毫秒级，CPU 可跑。
    集成姿势：只做推理前向，不引训练栈、不反向、不碰 ultralytics 训练 API。"""

    def __init__(self, model_name: str = "docling-project/ScreenParser",
                 imgsz: int = 1280, conf: float = 0.10, iou: float = 0.10):
        self.model_name = model_name
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self._model = None

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO   # 仅推理时加载，缺失即抛错由上层降级
            # 本地权重优先，缺失经 HF 镜像下载（github/HF 直连在本机不可达）
            from perception.weights import resolve_weights
            self._model = YOLO(resolve_weights(self.model_name))
        return self._model

    def detect(self, frame) -> List[Element]:
        model = self._ensure()
        results = model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou)
        els: List[Element] = []
        for r in results:
            for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                els.append(Element(
                    id=str(uuid.uuid4())[:8],
                    label=model.names[int(cls_id)],
                    bbox=(x1, y1, x2, y2),
                    conf=float(conf),
                ))
        return els


# ---------------------------------------------------------------------------
# 4. OCR 补文字（第二级兜底，让 LLM 知道按钮写啥）
# ---------------------------------------------------------------------------
class OCRBackend:
    """EasyOCR 识别每个 bbox 区域文字。ScreenParser 不识文字，这一级补上。
    懒加载：未装 easyocr 时 recognize() 原样返回，不崩。"""

    def __init__(self, langs: List[str] = None, scale: float = 1.0):
        """scale: OCR 输入缩放系数。1080p 全图 easyocr 约 18s（CPU 吃满），
        缩到 0.5 面积变 1/4 → 约 4~5s。长时程采集时必开，否则既采不到几条
        又会把用户电脑拖卡。识别率略降，但用于「按目标名找按钮」足够。"""
        self.langs = langs or ["ch_sim", "en"]
        self.scale = float(scale) if scale and 0 < scale <= 1.0 else 1.0
        self._reader = None
        self._cache_ts = 0.0      # 上次真实 OCR 的时刻，供调用方判断文字新鲜度
        self._cache_fp = None
        self._cache_ocr = None
        self.last_ocr_ran = False  # 本次 recognize 是否真的跑了 OCR（False=命中缓存）

    def _ensure(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.langs)
        return self._reader

    def _load_img(self, frame):
        """统一把 frame（np.ndarray / PIL / 路径）转成 np.ndarray；失败返回 None。"""
        try:
            import numpy as np
        except Exception:
            return None
        try:
            if isinstance(frame, np.ndarray):
                return frame
            if isinstance(frame, (str, bytes, os.PathLike)):
                from PIL import Image
                return np.array(Image.open(frame))
            return np.asarray(frame)   # PIL Image / 任意可转数组对象
        except Exception:
            return None

    def _readtext(self, img):
        """整图 OCR 一次，返回原始文本块 (poly, text, conf)（原图尺度）。

        按指纹缓存：静态屏/相邻帧指纹相同 → 复用上次结果，省掉每帧 ~18s 推理。
        指纹用大幅降采样的字节哈希，足够区分「屏是否变了」，开销可忽略。"""
        import time as _time
        from PIL import Image as _Image
        try:
            fp = hash(img[::40, ::40].tobytes())
        except Exception:
            fp = None
        self.last_ocr_ran = False
        if self._cache_fp == fp and self._cache_ocr:
            return self._cache_ocr            # 屏没变 → 复用
        reader = self._ensure()
        h, w = img.shape[:2]
        small = img
        if self.scale < 1.0:
            small = np.asarray(_Image.fromarray(img).resize(
                (max(1, int(w * self.scale)), max(1, int(h * self.scale)))))
        try:
            ocr_res = reader.readtext(small)  # 整图一次：(poly, text, conf)
        except Exception:
            return []
        # 缩放图的坐标映射回原图尺度，后续空间分配逻辑无需感知 scale
        if self.scale < 1.0:
            inv = 1.0 / self.scale
            ocr_res = [([(p[0] * inv, p[1] * inv) for p in poly], txt, conf)
                       for poly, txt, conf in ocr_res]
        self._cache_fp, self._cache_ocr = fp, ocr_res
        self._cache_ts = _time.time()
        self.last_ocr_ran = True
        return ocr_res

    def read_lines(self, frame, conf_thresh: float = 0.1) -> List:
        """整图 OCR 返回文本行列表，按垂直位置从上到下排序。

        每行：(cx, cy, text, conf)。供读屏场景（如微信消息抓取）直接用，
        不依赖 ScreenParser 元素。读不到图/缺依赖时返回空列表（不崩）。"""
        img = self._load_img(frame)
        if img is None:
            return []
        out = []
        for poly, txt, conf in self._readtext(img):
            if not txt or conf < conf_thresh:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            out.append((sum(xs) / len(xs), sum(ys) / len(ys), txt, float(conf)))
        out.sort(key=lambda t: t[1])
        return out

    def recognize(self, frame, elements: List[Element]) -> List[Element]:
        """整图一次 OCR，再把文本块按空间重叠挂到 ScreenParser 元素上。

        为什么不是逐框裁图 OCR（旧实现）：UI 元素框常只有 20~30px，
        easyocr 在这么小的裁图上几乎读不出字 → 实测 210 个元素 0 个带文字。
        整图 OCR 一次拿到全部文本块（含坐标），再按 IoU/中心包含分配给元素，
        既快（1 次推理 vs N 次）又能让小控件正确带字。"""
        if not elements:
            return elements
        img = self._load_img(frame)
        if img is None:
            return elements   # 无法读图（依赖缺失/路径不存在）→ 跳过 OCR，不崩
        ocr_res = self._readtext(img)
        H, W = img.shape[:2]
        for e in elements:
            x1, y1, x2, y2 = e.bbox
            ew, eh = max(1, x2 - x1), max(1, y2 - y1)
            texts = []
            for poly, txt, _conf in ocr_res:
                bx1 = min(p[0] for p in poly); by1 = min(p[1] for p in poly)
                bx2 = max(p[0] for p in poly); by2 = max(p[1] for p in poly)
                ix1, iy1 = max(x1, bx1), max(y1, by1)
                ix2, iy2 = min(x2, bx2), min(y2, by2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                if inter <= 0:
                    continue
                union = ew * eh + (bx2 - bx1) * (by2 - by1) - inter
                iou = inter / union if union > 0 else 0.0
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                inside = x1 <= cx <= x2 and y1 <= cy <= y2
                if inside or iou >= 0.3:
                    texts.append(txt)
            e.text = " ".join(texts) if texts else None
        return elements


# ---------------------------------------------------------------------------
# 5. 第三级：UIA 语义定位（控件树，兜底 YOLO 漏检）
# ---------------------------------------------------------------------------
class UIALocator(TargetLocator):
    """Windows UI Automation 控件树：**系统直接告诉你控件叫什么、在哪**。

    这是整个感知链里唯一零误差的文字来源——控件的 Name / ControlType 是
    系统原生属性，不经过任何视觉推理，不像 OCR 会把「关闭」读成「STS」。

    实测（2026-09-03，src/uia_probe.py）：递归遍历 505 节点 → 190 个有效带名
    控件，耗时 3.6s，其中中文名 136 个、噪声仅 8 个（4%）。同期 OCR 输出
    几乎全是误识（'on'/'L7'/'遇活 Windows'）。

    修掉的三个致命问题（原来这一级等于没接）：
    1. **只遍历 root.GetChildren()** → 那只拿到顶层窗口（十几个），根本拿不到
       按钮和文本框。改为递归遍历整棵控件树。
    2. **不过滤离屏控件** → 最小化窗口的坐标是 (-31981,-31981) 这种，混进训练
       数据就是纯噪声。改为按屏幕范围过滤。
    3. **uiautomation 从未安装** → import 抛异常被 except 吞掉，静默返回 []，
       上层以为「屏幕上没有元素」。现在 available() 显式探测。
    """

    def __init__(self, budget: float = 8.0, max_nodes: int = 4000,
                 max_depth: int = 14):
        self.budget = budget          # 遍历时间预算(秒)，防止某些 App 的控件树极深
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self._avail = None

    def available(self) -> bool:
        """显式探测依赖，别让 ImportError 被吞成『屏幕上没有元素』。"""
        if self._avail is None:
            try:
                import uiautomation  # noqa: F401
                self._avail = True
            except Exception:
                self._avail = False
        return self._avail

    def _screen_size(self):
        try:
            import ctypes
            return (ctypes.windll.user32.GetSystemMetrics(0),
                    ctypes.windll.user32.GetSystemMetrics(1))
        except Exception:
            return (1920, 1080)

    def detect(self, frame) -> List[Element]:
        if not self.available():
            return []
        import time as _time
        import uiautomation as auto

        deadline = _time.time() + self.budget
        sw, sh = self._screen_size()
        els: List[Element] = []

        def on_screen(r) -> bool:
            # 离屏/最小化窗口的坐标是 -31981 这种，必须挡在门外
            return (r.left >= -8 and r.top >= -8
                    and r.right <= sw + 8 and r.bottom <= sh + 8)

        def walk(ctrl, depth: int):
            if (depth > self.max_depth or len(els) >= self.max_nodes
                    or _time.time() > deadline):
                return
            try:
                children = ctrl.GetChildren()
            except Exception:
                return
            for c in children:
                if _time.time() > deadline or len(els) >= self.max_nodes:
                    return
                try:
                    name = (c.Name or "").strip()
                    rect = c.BoundingRectangle
                    ctype = c.ControlTypeName or "Control"
                except Exception:
                    continue
                if name and rect is not None and on_screen(rect):
                    w, h = rect.right - rect.left, rect.bottom - rect.top
                    if 4 <= w <= sw and 4 <= h <= sh:   # 挡掉零面积与整屏容器
                        els.append(Element(
                            id=str(uuid.uuid4())[:8],
                            label=ctype,
                            bbox=(rect.left, rect.top, rect.right, rect.bottom),
                            conf=1.0,      # 系统原生属性，不存在置信度问题
                            text=name,
                            text_src="uia",   # 系统原生控件名，零误差
                        ))
                walk(c, depth + 1)

        try:
            for top in auto.GetRootControl().GetChildren():
                if _time.time() > deadline or len(els) >= self.max_nodes:
                    break
                walk(top, 0)
        except Exception as ex:
            print(f"[UIA] 遍历失败: {type(ex).__name__}: {ex}")
        return els


def _iou(a, b) -> float:
    """两个 bbox 的交并比，用于把 UIA 控件匹配到视觉检测框。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    ub = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / float(ua + ub - inter)


def fuse_uia_text(els: List[Element], uia_els: List[Element],
                  iou_thresh: float = 0.25) -> List[Element]:
    """用 UIA 的**真值文字**纠正检测结果的文字。

    这是整条数据链路里性价比最高的一步：
      · 位置仍由 YOLO/ScreenParser 提供——它对自绘 UI、游戏、Canvas 也能框出来，
        而这些恰恰是 UIA 访问不到的，两边互补
      · 文字优先取 UIA 的系统原生 Name——零误差
      · 只有 UIA 覆盖不到的元素，才退回 OCR 的结果

    OCR 的问题不是慢，是**会把「关闭」读成「STS」**。这种噪声进了训练数据，
    模型学到的就是噪声映射，换一屏立刻崩（实测 goal='on' 时模型直接生成崩坏）。

    匹配不到的 UIA 控件会作为新元素补进来，提高召回——系统里有、但视觉漏检的
    控件，正是最该被学会点的那些。
    """
    if not uia_els:
        return els
    used = set()
    for e in els:
        best_i, best_iou = -1, 0.0
        for i, u in enumerate(uia_els):
            if i in used:
                continue
            ov = _iou(e.bbox, u.bbox)
            if ov > best_iou:
                best_iou, best_i = ov, i
        if best_i >= 0 and best_iou >= iou_thresh:
            e.text = uia_els[best_i].text      # 真值覆盖 OCR 噪声
            e.text_src = "uia"
            used.add(best_i)
    for i, u in enumerate(uia_els):
        if i not in used:
            els.append(u)                      # 视觉漏检但系统里真实存在的控件
    return els


# ---------------------------------------------------------------------------
# 6. 三级降级定位器：YOLO 检测 → OCR 补字 → UIA 语义
# ---------------------------------------------------------------------------
class TriLevelLocator(TargetLocator):
    """任意一级缺失/失败都自动跳过，保证满血运行不中断。"""

    def __init__(self, visual: TargetLocator, ocr: Optional[OCRBackend] = None,
                 uia: Optional[TargetLocator] = None):
        self.visual, self.ocr, self.uia = visual, ocr, uia

    def detect(self, frame) -> List[Element]:
        els: List[Element] = []
        try:
            els = self.visual.detect(frame)
        except Exception as e:
            print(f"[TriLevel] 视觉通道失败: {e}")
        if self.ocr:
            els = self.ocr.recognize(frame, els)

        # UIA 不再只是「视觉全漏时的兜底」——它最大的价值是提供**零误差文字**，
        # 直接纠正 OCR 的误识。视觉出框位置 + UIA 出真值文字 = 训练数据该有的样子。
        #
        # 原来这一级写在 `if not els:` 里，而 YOLO 永远能检出上百个元素，
        # 所以它这辈子都不会被调用——等于白写，数据照样被 OCR 噪声污染。
        #
        # 注：els 为空时 fuse_uia_text 会把 UIA 元素全部补进来，
        # 所以「视觉全漏 → 降级 UIA」的兜底语义天然满足，无需额外分支。
        if self.uia:
            try:
                uia_els = self.uia.detect(frame)
            except Exception as ex:
                print(f"[TriLevel] UIA 通道失败: {type(ex).__name__}: {ex}")
                uia_els = []
            if uia_els:
                els = fuse_uia_text(els, uia_els)
        return els


# ---------------------------------------------------------------------------
# 7. 双通道：DOM 快通道优先 + 视觉通道降级
# ---------------------------------------------------------------------------
class DOMChannel(TargetLocator):
    """uiautomation 控件树（原生 App / 控件树完整时 0.3-0.8s）。
    桌面端用 uiautomation；Android 换 uiautomator2 同接口即可。"""

    def detect(self, frame) -> List[Element]:
        try:
            import uiautomation as auto
        except Exception:
            return []
        els: List[Element] = []
        try:
            for c in auto.GetRootControl().GetChildren():
                try:
                    rect = c.BoundingRectangle
                    if rect:
                        els.append(Element(
                            id=str(uuid.uuid4())[:8], label=c.ControlTypeName,
                            bbox=(rect.left, rect.top, rect.right, rect.bottom),
                            conf=1.0, text=c.Name or None))
                except Exception:
                    continue
        except Exception:
            pass
        return els


class DualChannelLocator:
    """DOM 快通道优先；关键元素缺失（如 WebView 内嵌）自动降级视觉通道。"""

    def __init__(self, dom: TargetLocator, visual: TargetLocator,
                 require_labels=("Button", "TextInput", "Text Input")):
        self.dom, self.visual, self.require_labels = dom, visual, require_labels

    def detect(self, frame) -> List[Element]:
        els = self.dom.detect(frame)
        have_key = any(e.label in self.require_labels for e in els)
        if not have_key:                       # DOM 缺关键元素 → 降级视觉
            els = self.visual.detect(frame)
        return els


# ---------------------------------------------------------------------------
# 8. 与预判帧 / System1 / 记忆系统的完整闭环
# ---------------------------------------------------------------------------
def correction_pipeline(frame, sys1_predict, locator: TargetLocator,
                        llm_decide, computer_click, memory=None) -> dict:
    """
    预判帧监控发现偏差 → 关键帧 → 定位器锁坐标 → LLM 选目标 → CU 点击 → 记忆。

    sys1_predict(id) -> (x, y)    : System1 对该物体的物理预测位置
    llm_decide(elements, goal)    : 返回应选中的 Element
    computer_click(x, y)          : 执行点击
    memory.challenge_memory(...)  : 成功/失败回写经验（可选）
    """
    elements = locator.detect(frame)
    if not elements:
        return {"status": "NO_ELEMENT", "elements": []}

    # 1) 与 System1 预测比对：挑出偏差最大的元素作为候选
    candidates = []
    for e in elements:
        pred = sys1_predict(e.id)
        if pred:
            dev = abs(pred[0] - e.center()[0]) + abs(pred[1] - e.center()[1])
            candidates.append((dev, e))
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 2) 结构化上下文注入 LLM，让它只推理「点哪个」
    ctx = "当前屏幕UI元素：\n" + "\n".join(e.to_context(i + 1) for i, e in enumerate(elements))
    target = llm_decide(ctx, goal="点击确认按钮")

    # 3) Computer Use 执行
    cx, cy = target.center()
    ok = computer_click(cx, cy)

    # 4) 记忆回写（成功/失败 → 经验系统）
    if memory and target.id:
        memory.challenge_memory(target.id, success=ok)

    return {"status": "CLICKED" if ok else "FAILED",
            "target": target.label, "center": (cx, cy),
            "elements": [e.to_context(i + 1) for i, e in enumerate(elements)]}


# ---------------------------------------------------------------------------
# 8.5 开放词汇定位器（YOLOE / YOLO-World）：按语义名直接检测元素
# ---------------------------------------------------------------------------
class OpenVocabLocator(TargetLocator):
    """YOLOE（清华，ICCV 2025）或 YOLO-World（腾讯，CVPR 2024）。
    相比 ScreenParser 的质变：
      - 文本提示：model.set_classes(["retry button","confirm button"]) → 直接出带名框
        LLM 上下文从「Button 在 (520,340)」升级为「retry button 在 (520,340)」，省掉 OCR。
      - 免提示：内置 1200+ 类，自动识别画面所有可见物。
      - 跑闭集时速度与 YOLO11 完全一致（开放模块重参数化进检测头），需要时才切开放模式。
      - 带分割掩码 results[0].masks（YOLOE-seg），可做像素级点击区域。
    集成姿势：只做推理前向；ultralytics 懒加载；缺失即由上层降级到 ScreenParser/OCR/UIA。

    用法：
        loc = OpenVocabLocator("yoloe-11s.pt")      # 闭集/免提示
        loc.detect_named(frame, ["确认","重试"])     # 文本提示：按语义名找
    """

    PROMPT_FREE_MODELS = ("yoloe", "yolo-world", "worldv2")

    def __init__(self, model_name: str = "yoloe-11s.pt",
                 imgsz: int = 1280, conf: float = 0.10, iou: float = 0.10):
        self.model_name, self.imgsz, self.conf, self.iou = model_name, imgsz, conf, iou
        self._model = None

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        return self._model

    def detect(self, frame) -> List[Element]:
        """免提示模式：依赖模型内置词汇自动识别（如 YOLOE 的 1200+ 类）。"""
        model = self._ensure()
        results = model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou)
        return self._to_elements(model, results)

    def detect_named(self, frame, names: List[str]) -> List[Element]:
        """文本提示模式：按 LLM 给的语义名找元素，框直接带语义标签，无需 OCR。"""
        model = self._ensure()
        try:
            model.set_classes(names)
        except Exception:
            pass  # 部分权重不支持 set_classes → 退化为免提示 detect()
        results = model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou)
        return self._to_elements(model, results)

    @staticmethod
    def _to_elements(model, results) -> List[Element]:
        els: List[Element] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                # YOLOE 开放模式下 names 是提示名；闭集下是类别名
                try:
                    label = model.names[int(cls_id)]
                except Exception:
                    label = "object"
                els.append(Element(
                    id=str(uuid.uuid4())[:8],
                    label=str(label),
                    bbox=(x1, y1, x2, y2),
                    conf=float(conf),
                ))
        return els


class SemanticLocator(TargetLocator):
    """语义增强定位器：ScreenParser（闭集 55 类，UI 域最优）为主，
    需要按 LLM 语义名找未知元素时，逐级降级：

        SAM3（开放词汇 + mask 质心） → YOLOE（开放词汇） → 闭集让 LLM 自己挑

    闭集够用就不动开放通道（零额外成本），只在目标名超出 55 类时才调用。
    注意：**空结果也算失败**，会继续往下一级探——否则"开放通道返回空"会被
    当成成功结果直接返回，那不叫降级（原实现就踩了这个坑）。
    """

    def __init__(self, closed: TargetLocator,
                 open_vocab: Optional[TargetLocator] = None,
                 sam3: Optional[TargetLocator] = None):
        self.closed, self.open_vocab, self.sam3 = closed, open_vocab, sam3

    def detect(self, frame) -> List[Element]:
        try:
            return self.closed.detect(frame)
        except Exception:
            return []

    def detect_named(self, frame, names: List[str]) -> List[Element]:
        """LLM 指定语义名找元素：SAM3 → YOLOE → 闭集，任一级真出结果才返回。"""
        for loc in (self.sam3, self.open_vocab):
            if not loc:
                continue
            try:
                got = loc.detect_named(frame, names)
            except Exception:
                got = []
            if got:
                return got
        return self.detect(frame)  # 退回闭集，靠 LLM 从 55 类里挑


def build_semantic_locator(closed: Optional[TargetLocator] = None,
                           use_sam3: bool = True,
                           use_yoloe: bool = True,
                           sam3_verbose: bool = True) -> SemanticLocator:
    """组装带 SAM3 的三级语义定位器（SAM3 → YOLOE → 闭集）。

    为什么要延迟 import：sam3_locator 反过来 import 本模块的 Element/TargetLocator，
    在模块顶层 import 会形成循环导入 → 必须放函数里。

    任何一级装不起来都不抛错：SAM3 缺依赖/缺权重/worker 起不来 → 该级为 None，
    链路自动缩成两级或一级，符合"缺失即降级"。
    """
    if closed is None:
        closed = ScreenParserBackend()

    sam3 = None
    if use_sam3:
        try:
            from sam3_locator import SAM3Locator
            sam3 = SAM3Locator(verbose=sam3_verbose)
        except Exception:
            sam3 = None

    yolo = None
    if use_yoloe:
        try:
            yolo = OpenVocabLocator()
        except Exception:
            yolo = None

    return SemanticLocator(closed=closed, open_vocab=yolo, sam3=sam3)


# ---------------------------------------------------------------------------
# 9. 自检（python detector.py 直接运行，不依赖 ultralytics）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class MockVisual(TargetLocator):
        """演示用假定位器，验证降级/上下文/闭环逻辑，不下载模型。"""
        def detect(self, frame):
            return [
                Element(id="e1", label="Button", bbox=(500, 300, 560, 340), conf=0.92),
                Element(id="e2", label="Text Input", bbox=(100, 100, 300, 130), conf=0.80),
            ]

    # 三级降级定位器（视觉 + OCR + UIA）
    loc = TriLevelLocator(visual=MockVisual(), ocr=OCRBackend(), uia=UIALocator())

    # System1 对该物体的物理预测（模拟预判帧输出）
    sys1 = {"e1": (520, 320)}   # 预测 e1 在 (520,320)，实际检测到 (530,320)

    def fake_llm(ctx, goal):
        # 真实环境此处调用 LLM，根据 ctx 选目标；演示直接选第一个 Button
        return Element(id="e1", label="Button", bbox=(500, 300, 560, 340), text="确定")

    def fake_click(x, y):
        print(f"  [Computer Use] 点击 ({x}, {y})")
        return True

    print("=== M3 定位器 + 预判帧 + 记忆 闭环演示 ===")
    out = correction_pipeline("fake.png", sys1.get, loc, fake_llm, fake_click)
    print("上下文：")
    for line in out["elements"]:
        print(" ", line)
    print("结果：", out["status"], "→", out["target"], out["center"])
    print("注：真实部署把 MockVisual 换成 ScreenParserBackend() 即满血运行。")

    # ---- 开放词汇升级演示：按语义名找元素，标签直接带名，无需 OCR ----
    print("\n=== 开放词汇定位器（YOLOE）升级演示 ===")

    class MockOpenVocab(TargetLocator):
        """假开放词汇定位器：LLM 说找「重试」，它就直接返回带名框。"""
        def detect_named(self, frame, names):
            # 演示：LLM 要找 "重试"，模型直接吐带语义名的框
            return [Element(id="r1", label="重试", bbox=(620, 360, 700, 400), conf=0.90)]
        def detect(self, frame):
            return [Element(id="x1", label="object", bbox=(620, 360, 700, 400), conf=0.90)]

    sem = SemanticLocator(closed=MockVisual(), open_vocab=MockOpenVocab())
    # LLM 解析指令「点击重试按钮」→ 提取语义名 ["重试"] → 开放词汇定位
    named = sem.detect_named("fake.png", ["重试"])
    print("开放词汇 detect_named(['重试']) 结果：")
    for e in named:
        print(" ", e.to_context(1))
    print("→ 标签已是『重试』，LLM 无需再靠 OCR 猜文字；三级降级链其余两级可跳过。")
