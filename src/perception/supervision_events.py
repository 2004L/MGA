"""
supervision_events.py —— 基于 Roboflow supervision 的「结构化视频事件层」
=========================================================================

MGA 视频感知层（让模型能看见 + 数据分析 + 运输数据给大模型）的**第一块**。

把"一帧检测结果"升级为"带持久 ID 的跟踪目标 + 越线计数 + 区域停留时长"的
结构化事件，喂给 System2（LLM）。这正是 supervision 的强项，也是原 events.py
（只有类别计数签名 diff）补不上的硬缺口。

设计要点（对齐 MGA 全局纪律）：
  · 可选依赖：supervision 未安装时**优雅降级**到 events.SemanticEventDetector
    （类别计数唤醒），绝不静默崩溃，也绝不假装主通道通（active_backend 如实标注）。
  · 模型无关：输入是 xyxy + class_id + class_name + conf，任何检测器
    （YOLO / SAM3 / 桌面 ScreenParser）的输出都能喂进来。
  · 真实跟踪：用 sv.ByteTrack 做跨帧持久 ID 关联；sv.LineZone 做越线计数；
    sv.PolygonZone 做区域计数 + 自管 entry/exit 时间戳算停留时长(dwell)。
  · 事件可审计：每次越线/停留结束都产出一条结构化事件 dict，
    含 tracker_id / class / 方向 / 时长，直接 transport 给 S2。

依赖：supervision (pip install supervision)，以及它拉进的 numpy / opencv-python。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---- 可选依赖：supervision 不在则降级 ----
try:
    import supervision as sv
    _HAS_SV = True
except Exception:  # pragma: no cover - 依赖缺失时的诚实降级
    sv = None
    _HAS_SV = False


# ---------------------------------------------------------------------------
# 输入适配：与 detector.Element / perception.UIElement 同构，便于直接喂
# ---------------------------------------------------------------------------
@dataclass
class TrackBox:
    """单帧检测框，模型无关。xyxy = (x1,y1,x2,y2)。"""
    xyxy: Tuple[float, float, float, float]
    class_id: int = 0
    class_name: str = "object"
    conf: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def boxes_from_mga(elements) -> List[TrackBox]:
    """把 MGA UIElement / detector.Element 列表转成 TrackBox 列表。

    两种对象都有 .bbox(xyxy) / .label(类别名) / .conf；class_id 没有就按
    class_name 哈希成稳定 int（同名的跨帧 id 一致，ByteTrack 才能稳定关联）。
    """
    out: List[TrackBox] = []
    for e in elements or []:
        bbox = getattr(e, "bbox", None) or getattr(e, "xyxy", None)
        if not bbox:
            continue
        label = getattr(e, "label", None) or getattr(e, "text", "") or "object"
        conf = float(getattr(e, "conf", 0.0) or 0.0)
        out.append(TrackBox(xyxy=tuple(bbox), class_id=_stable_id(label),
                            class_name=str(label), conf=conf))
    return out


def _stable_id(name: str) -> int:
    """类别名 → 稳定 int（同名跨帧一致）。仅用于在 supervision 里区分 class。"""
    return int(abs(hash(name))) % 1000


# ---------------------------------------------------------------------------
# 几何：判断点在有向线段哪一侧（用于逐目标越线方向判定）
# ---------------------------------------------------------------------------
def _side(p1: Tuple[float, float], p2: Tuple[float, float],
          pt: Tuple[float, float]) -> int:
    """返回 +1 / -1 / 0：pt 相对有向线段 p1->p2 的哪一侧。"""
    x1, y1 = p1
    x2, y2 = p2
    x, y = pt
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# 线 / 区域配置
# ---------------------------------------------------------------------------
@dataclass
class LineCfg:
    id: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]


@dataclass
class ZoneCfg:
    id: str
    polygon: List[Tuple[float, float]]  # [(x,y), ...]


# ---------------------------------------------------------------------------
# 事件结构
# ---------------------------------------------------------------------------
@dataclass
class CrossEvent:
    line_id: str
    tracker_id: int
    class_name: str
    direction: str          # "in" / "out"（相对线段 p1->p2 朝向）
    frame: int
    t: float

    def to_dict(self) -> dict:
        return {"type": "line_cross", "line_id": self.line_id,
                "tracker_id": self.tracker_id, "class_name": self.class_name,
                "direction": self.direction, "frame": self.frame, "t": round(self.t, 3)}


@dataclass
class DwellEvent:
    zone_id: str
    tracker_id: int
    class_name: str
    dwell_sec: float
    frame_in: int
    frame_out: int

    def to_dict(self) -> dict:
        return {"type": "zone_dwell", "zone_id": self.zone_id,
                "tracker_id": self.tracker_id, "class_name": self.class_name,
                "dwell_sec": round(self.dwell_sec, 2),
                "frame_in": self.frame_in, "frame_out": self.frame_out}


# ---------------------------------------------------------------------------
# 主引擎
# ---------------------------------------------------------------------------
class VideoEventAnalyzer:
    """逐帧喂检测框，产出结构化跟踪/越线/停留事件。

    仅当 supervision 可用时构造；否则请用 build_event_analyzer() 拿降级实现。
    """

    def __init__(self, lines: Sequence[LineCfg] = (), zones: Sequence[ZoneCfg] = (),
                 frame_wh: Tuple[int, int] = (1920, 1080)):
        if not _HAS_SV:
            raise RuntimeError("supervision 未安装：请用 build_event_analyzer() 拿降级实现")
        self.frame_wh = frame_wh
        self.byte_track = sv.ByteTrack()
        self.frame_idx = 0
        self.last_side: Dict[str, Dict[int, int]] = {}   # line_id -> {tracker_id: side}

        # 线
        self.lines: Dict[str, LineCfg] = {}
        self.line_zones: Dict[str, object] = {}
        for ln in lines:
            self.lines[ln.id] = ln
            self.line_zones[ln.id] = sv.LineZone(
                start=sv.Point(*ln.p1), end=sv.Point(*ln.p2))
            self.last_side[ln.id] = {}

        # 区域 + 停留跟踪
        self.zones: Dict[str, ZoneCfg] = {}
        self.poly_zones: Dict[str, object] = {}
        self.zone_enter: Dict[str, Dict[int, Tuple[int, float, str]]] = {}  # zone-> {tid:(frame,t,class_name)}
        for z in zones:
            self.zones[z.id] = z
            self.poly_zones[z.id] = sv.PolygonZone(
                polygon=np.array(z.polygon, dtype=np.int64))
            self.zone_enter[z.id] = {}

        self.cross_events: List[CrossEvent] = []
        self.dwell_events: List[DwellEvent] = []
        self.last_detections = None

    # -------------------- 单帧处理 --------------------
    def update(self, boxes: Sequence[TrackBox], t: Optional[float] = None) -> dict:
        """喂一帧，返回当前结构化快照 + 累积新事件。

        返回 dict 含：tracked(当前帧带 ID 目标) / line_counts / zone_counts /
        new_cross / new_dwell。新事件同时累积进 self.cross_events / self.dwell_events。
        """
        if t is None:
            t = time.time()
        self.frame_idx += 1
        fidx = self.frame_idx

        if not boxes:
            dets = sv.Detections.empty()
        else:
            xyxy = np.array([b.xyxy for b in boxes], dtype=float)
            cid = np.array([int(b.class_id) for b in boxes], dtype=int)
            conf = np.array([float(b.conf) for b in boxes], dtype=float)
            dets = sv.Detections(xyxy=xyxy, class_id=cid, confidence=conf)

        # 1) 跟踪：得到持久 tracker_id（supervision 0.28+ 用 update_with_detections；
        #    ByteTrack 自 0.28 deprecated、0.31 移除，届时需切到 sv.Tracker 统一接口）
        dets = self.byte_track.update_with_detections(dets)
        self.last_detections = dets

        # class_id → class_name 映射（不依赖检测框顺序，避免 ByteTrack 重排错位）
        id2name = {}
        for b in boxes:
            id2name[int(b.class_id)] = b.class_name

        def name_of(cid: int) -> str:
            return id2name.get(int(cid), "object")

        tracked = []
        if dets.tracker_id is not None:
            for i in range(len(dets)):
                tid = int(dets.tracker_id[i])
                name = name_of(dets.class_id[i])
                tracked.append({"tracker_id": tid, "class_name": name,
                                "xyxy": [float(v) for v in dets.xyxy[i].tolist()],
                                "center": list(boxes[i].center)})

        # 2) 越线：sv.LineZone 聚合计数 + 逐目标方向事件
        new_cross: List[CrossEvent] = []
        for lid, lz in self.line_zones.items():
            lz.trigger(dets)
            ln = self.lines[lid]
            prev = self.last_side[lid]
            cur: Dict[int, int] = {}
            if dets.tracker_id is not None:
                for i in range(len(dets)):
                    tid = int(dets.tracker_id[i])
                    c = boxes[i].center
                    s = _side(ln.p1, ln.p2, c)
                    cur[tid] = s
                    if tid in prev and prev[tid] != 0 and s != 0 and prev[tid] != s:
                        direction = "in" if s == 1 else "out"
                        ev = CrossEvent(lid, tid, boxes[i].class_name, direction, fidx, t)
                        new_cross.append(ev)
                        self.cross_events.append(ev)
            self.last_side[lid] = cur

        # 3) 区域计数 + 停留时长
        new_dwell: List[DwellEvent] = []
        for zid, pz in self.poly_zones.items():
            mask = pz.trigger(dets)  # bool 数组：哪些框在区域内
            enter = self.zone_enter[zid]
            if dets.tracker_id is not None:
                present = set()
                for i in range(len(dets)):
                    if not bool(mask[i]):
                        continue
                    tid = int(dets.tracker_id[i])
                    present.add(tid)
                    cname = name_of(dets.class_id[i])
                    if tid not in enter:
                        # 进入：记录进入帧/时间/类别名（沿用至离开时结算，避免离开帧已出区域丢类名）
                        enter[tid] = (fidx, t, cname)
                # 离开 → 结算 dwell
                for tid in list(enter.keys()):
                    if tid not in present:
                        fin, tin, cname = enter.pop(tid)
                        dwell = max(0.0, t - tin)
                        ev = DwellEvent(zid, tid, cname, dwell, fin, fidx)
                        new_dwell.append(ev)
                        self.dwell_events.append(ev)

        return {
            "tracked": tracked,
            "line_counts": {lid: {"in": lz.in_count, "out": lz.out_count}
                            for lid, lz in self.line_zones.items()},
            "zone_counts": {zid: int(pz.current_count) for zid, pz in self.poly_zones.items()},
            "new_cross": [e.to_dict() for e in new_cross],
            "new_dwell": [e.to_dict() for e in new_dwell],
            "active_backend": "supervision",
        }

    def summary(self) -> dict:
        return {
            "active_backend": "supervision" if _HAS_SV else "none",
            "n_cross_events": len(self.cross_events),
            "n_dwell_events": len(self.dwell_events),
            "line_counts": {lid: {"in": lz.in_count, "out": lz.out_count}
                            for lid, lz in self.line_zones.items()},
            "zone_counts": {zid: int(pz.current_count) for zid, pz in self.poly_zones.items()},
        }


# ---------------------------------------------------------------------------
# 降级工厂：supervision 可用→真跟踪；不可用→类别计数唤醒（events.py）
# ---------------------------------------------------------------------------
def build_event_analyzer(lines=(), zones=(), frame_wh=(1920, 1080),
                         count_delta: int = 3, watch_classes=None):
    """返回结构化事件分析器。

    supervision 可用 → VideoEventAnalyzer（真跟踪/越线/停留）。
    不可用 → _FallbackAnalyzer（仅类别计数签名 diff，沿用 events.SemanticEventDetector）。
    """
    if _HAS_SV:
        return VideoEventAnalyzer(lines=lines, zones=zones, frame_wh=frame_wh)
    try:
        from perception.events import SemanticEventDetector
    except Exception:
        from events import SemanticEventDetector  # 脚本直跑时（src/perception 在 path）

    class _FallbackAnalyzer:
        active_backend = "classcount-fallback"

        def __init__(self):
            self.det = SemanticEventDetector(count_delta=count_delta,
                                            watch_classes=watch_classes)

        def update(self, boxes, t=None):
            elems = [type("E", (), {"label": b.class_name})() for b in (boxes or [])]
            fired, why = self.det.check(elems)
            return {"tracked": [], "line_counts": {}, "zone_counts": {},
                    "new_cross": [], "new_dwell": [],
                    "class_event": {"fired": fired, "why": why},
                    "active_backend": "classcount-fallback"}

        def summary(self):
            return {"active_backend": "classcount-fallback"}

    return _FallbackAnalyzer()


if __name__ == "__main__":
    # 沙箱自检：模拟一个人从线上方走到下方、并在区域内停留若干帧
    if not _HAS_SV:
        print("[supervision_events] supervision 未安装 → 走降级；请 pip install supervision 跑真探针")
    else:
        line = LineCfg(id="A", p1=(100, 200), p2=(400, 200))  # 水平线 y=200
        zone = ZoneCfg(id="entrance", polygon=[(300, 300), (500, 300),
                                               (500, 500), (300, 500)])
        an = VideoEventAnalyzer(lines=[line], zones=[zone])
        # 帧序列：人从上往下穿过线、进入区域停留、再离开
        seq = [
            [TrackBox((150, 100, 180, 140), 0, "person", 0.9)],   # 线上方
            [TrackBox((150, 180, 180, 220), 0, "person", 0.9)],   # 跨线
            [TrackBox((150, 260, 180, 300), 0, "person", 0.9)],   # 线下方
            [TrackBox((350, 350, 380, 390), 0, "person", 0.9)],   # 进入区域
            [TrackBox((350, 350, 380, 390), 0, "person", 0.9)],   # 停留
            [TrackBox((150, 260, 180, 300), 0, "person", 0.9)],   # 离开区域
        ]
        for k, boxes in enumerate(seq):
            snap = an.update(boxes, t=float(k))
            if snap["new_cross"]:
                print("越线:", snap["new_cross"])
            if snap["new_dwell"]:
                print("停留:", snap["new_dwell"])
        print("summary:", an.summary())
        assert an.summary()["n_cross_events"] >= 1, "应至少检测到 1 次越线"
        print("✅ VideoEventAnalyzer 自检通过")
