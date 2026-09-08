"""
events.py —— 语义事件触发（① 感知层的「内容变化」唤醒器）
====================================================
为什么需要它（和上轮采不到轨迹直接相关）：

    System1 的触发是**几何**的：被跟踪元素的位置偏差超阈值才唤醒 System2。
    在静止桌面上，被跟踪元素（如某个 Button）中心逐帧不变 → 几何偏差恒为 0
    → 跟踪器自适应收敛 → 永不触发关键帧 → 采不到任何轨迹。

    但屏幕的**内容**是活的。实测真实桌面每帧：Button 数 38→41→43→48、
    Text 数 83→87→94→95。几何不动，语义一直在变。

    本模块补上「语义触发」：比较前后帧的检测结果**类别计数签名**，
    出现显著变化（某类数量突变、或关注类别出现/消失）即判定为事件，
    与 System1 几何触发做「或」合并，唤醒 System2。

    代价：仅对上一帧 detector 输出的类别计数做 diff，零 NN、零额外推理，
    复用已经算出来的检测框，不增加截屏/推理开销。

    等级（与全系统一致）：这是 ① 感知层之上的轻量事件判定，不是新模型。
"""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

try:
    from perception.detector import Element
except Exception:  # 独立 import 时也能当纯函数用
    Element = "Element"


class SemanticEventDetector:
    """基于元素类别计数签名的语义事件检测。"""

    def __init__(self, count_delta: int = 3, watch_classes: List[str] = None,
                 min_elements: int = 1):
        """
        count_delta   : 某类别前后帧数量差的绝对值 >= 该值 → 事件。
        watch_classes : 额外关注的类别名；这些类别「出现或完全消失」即事件
                        （即使数量差没到 count_delta，比如 0→1）。
        min_elements  : 当帧检测元素数 < 该值时不判事件（避免空屏/首帧抖动误触发）。
        """
        self.count_delta = max(1, count_delta)
        self.watch = set(watch_classes or [])
        self.min_elements = min_elements
        self.prev: Counter = None

    @staticmethod
    def _signature(elements: List) -> Counter:
        c = Counter()
        for e in elements:
            c[getattr(e, "label", "?")] += 1
        return c

    def check(self, elements: List) -> Tuple[bool, str]:
        """返回 (是否事件, 原因描述)。首次调用只建基线、不触发。"""
        cur = self._signature(elements)
        if self.prev is None:
            self.prev = cur
            return False, ""
        if sum(cur.values()) < self.min_elements:
            self.prev = cur
            return False, ""

        reasons = []
        for k in set(cur) | set(self.prev):
            cur_n, prev_n = cur.get(k, 0), self.prev.get(k, 0)
            d = cur_n - prev_n
            if abs(d) >= self.count_delta:
                reasons.append(f"{k}: {prev_n}→{cur_n}")
            elif k in self.watch and (cur_n == 0) != (prev_n == 0):
                reasons.append(f"{k} {'出现' if cur_n else '消失'}")

        self.prev = cur
        return bool(reasons), "; ".join(reasons)

    def reset(self) -> None:
        self.prev = None


if __name__ == "__main__":
    # 自检：类别数量突变 / 关注类出现 / 0 元素不触发 / 平稳不变不触发
    def E(label):
        return type("E", (), {"label": label})()

    det = SemanticEventDetector(count_delta=3, watch_classes=["Button", "Person"])
    a = [E("Button") for _ in range(38)] + [E("Text") for _ in range(83)]
    b = [E("Button") for _ in range(41)] + [E("Text") for _ in range(87)]   # +3 Button/+4 Text → 触发
    c = [E("Button") for _ in range(41)] + [E("Text") for _ in range(87)]   # 平稳 → 不触发
    d = []                                                                     # 空屏 → 不触发
    e = [E("Button") for _ in range(41)] + [E("Person") for _ in range(1)]    # Person 出现 → 触发

    r1, w1 = det.check(a); print("首帧(基线):", r1, repr(w1))
    r2, w2 = det.check(b); print("突变:", r2, repr(w2)); assert r2 and "Button" in w2
    r3, w3 = det.check(c); print("平稳:", r3, repr(w3)); assert not r3
    r4, w4 = det.check(d); print("空屏:", r4, repr(w4)); assert not r4
    r5, w5 = det.check(e); print("关注类出现:", r5, repr(w5)); assert r5 and "Person" in w5
    print("✅ SemanticEventDetector 自检通过")
