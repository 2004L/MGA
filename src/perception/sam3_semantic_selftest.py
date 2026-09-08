"""
sam3_semantic_selftest.py —— SAM3 接入 SemanticLocator 的自检

分两层，避免"真模型起不来就什么都测不了"：
    mock 层（默认跑，秒级）：用假定位器验证**降级顺序与空结果处理**是否正确。
    real 层（--real，慢）：真的起 SAM3 worker 跑一张图，验证跨环境调用与端到端结果。

用法：
    python src/perception/sam3_semantic_selftest.py            # 只跑 mock
    python src/perception/sam3_semantic_selftest.py --real     # 加跑真 SAM3
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detector import Element, SemanticLocator, TargetLocator   # noqa: E402

PASS: list = []
FAIL: list = []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


class FakeLoc(TargetLocator):
    """可控的假定位器：能返回结果 / 返回空 / 抛异常，并记录被调用情况。"""

    def __init__(self, tag: str, result=None, raise_exc: bool = False):
        self.tag = tag
        self._result = result or []
        self.raise_exc = raise_exc
        self.calls = 0

    def detect(self, frame):
        return list(self._result)

    def detect_named(self, frame, names):
        self.calls += 1
        if self.raise_exc:
            raise RuntimeError(f"{self.tag} 故意抛错")
        return list(self._result)


def _el(tag: str) -> Element:
    return Element(id=f"{tag}_1", label=tag, bbox=(0, 0, 10, 10), conf=0.9)


def run_mock_tests():
    print("\n=== mock 层：降级顺序与空结果处理 ===")

    # 闭集兜底（永远有结果，作为最后一级）
    closed = FakeLoc("closed", [_el("closed")])

    # 1) SAM3 有结果 → 应优先返回 SAM3，且不该去动 YOLOE
    sam3 = FakeLoc("sam3", [_el("sam3")])
    yolo = FakeLoc("yoloe", [_el("yoloe")])
    loc = SemanticLocator(closed=closed, open_vocab=yolo, sam3=sam3)
    got = loc.detect_named("x.png", ["icon"])
    check("SAM3 命中时优先于 YOLOE",
          len(got) == 1 and got[0].label == "sam3" and yolo.calls == 0,
          f"得到 {[e.label for e in got]}, yolo 调用 {yolo.calls} 次")

    # 2) SAM3 返回空 → 必须继续降级到 YOLOE（空结果不算成功）
    sam3 = FakeLoc("sam3", [])
    yolo = FakeLoc("yoloe", [_el("yoloe")])
    loc = SemanticLocator(closed=closed, open_vocab=yolo, sam3=sam3)
    got = loc.detect_named("x.png", ["icon"])
    check("SAM3 空结果会降级到 YOLOE",
          len(got) == 1 and got[0].label == "yoloe",
          f"得到 {[e.label for e in got]}")

    # 3) SAM3 抛异常 → 不能抛穿，要降级到 YOLOE
    sam3 = FakeLoc("sam3", raise_exc=True)
    yolo = FakeLoc("yoloe", [_el("yoloe")])
    loc = SemanticLocator(closed=closed, open_vocab=yolo, sam3=sam3)
    try:
        got = loc.detect_named("x.png", ["icon"])
        check("SAM3 抛异常被吞掉并降级", len(got) == 1 and got[0].label == "yoloe")
    except Exception as e:
        check("SAM3 抛异常被吞掉并降级", False, f"异常穿透: {type(e).__name__}: {e}")

    # 4) SAM3 空 + YOLOE 空 → 落到闭集
    sam3 = FakeLoc("sam3", [])
    yolo = FakeLoc("yoloe", [])
    loc = SemanticLocator(closed=closed, open_vocab=yolo, sam3=sam3)
    got = loc.detect_named("x.png", ["icon"])
    check("两级都空则落闭集", len(got) == 1 and got[0].label == "closed",
          f"得到 {[e.label for e in got]}")

    # 5) 向后兼容：不传 sam3 时行为与旧版一致（走 YOLOE）
    yolo = FakeLoc("yoloe", [_el("yoloe")])
    loc = SemanticLocator(closed=closed, open_vocab=yolo)
    got = loc.detect_named("x.png", ["icon"])
    check("不传 sam3 时向后兼容", len(got) == 1 and got[0].label == "yoloe")

    # 6) 一级都没有 → 只剩闭集
    loc = SemanticLocator(closed=closed)
    got = loc.detect_named("x.png", ["icon"])
    check("无开放通道时纯闭集也能返回", len(got) == 1 and got[0].label == "closed")


def run_real_tests():
    print("\n=== real 层：真 SAM3（会起常驻 worker，首次约 1~2 分钟）===")
    from sam3_locator import SAM3Locator

    img = os.path.join(_ROOT, "wechat_full.png")
    if not os.path.isfile(img):
        check("真机图片存在", False, f"缺 {img}")
        return

    loc = SAM3Locator()
    ok = loc.available
    check("SAM3 可用", ok, loc.load_error or "")
    if not ok:
        return

    print(f"  运行模式 = {loc.mode}")
    check("base python 下应走 worker 模式",
          loc.mode in ("direct", "worker"), f"实际 {loc.mode}")

    els = loc.detect_named(img, ["icon"])
    check("真 SAM3 检出元素", len(els) > 0, f"命中 {len(els)} 个")
    if els:
        cp = loc.click_point(0)
        print(f"  示例: {els[0].to_context(1)} conf={els[0].conf:.2f} 质心={cp}")

    # 接进 SemanticLocator 端到端
    sem = SemanticLocator(closed=FakeLoc("closed", [_el("closed")]),
                          open_vocab=None, sam3=loc)
    got = sem.detect_named(img, ["icon"])
    check("SemanticLocator 端到端拿到 SAM3 结果",
          len(got) > 0 and all(e.label == "icon" for e in got),
          f"得到 {len(got)} 个, label={[e.label for e in got][:3]}")

    loc.close()


def main() -> int:
    run_mock_tests()
    if "--real" in sys.argv:
        run_real_tests()
    else:
        print("\n（跳过真机层，加 --real 启用）")

    print(f"\n结果: {len(PASS)} 通过 / {len(FAIL)} 失败")
    if FAIL:
        print("失败项:", ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
