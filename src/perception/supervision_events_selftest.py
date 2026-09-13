"""
supervision_events_selftest.py —— VideoEventAnalyzer 真探针
===========================================================
用仿真帧验证：① 跨帧持久 ID  ② 越线计数  ③ 区域停留时长  ④ 降级工厂。
运行：PYTHONPATH=src <venv>/python.exe -m perception.supervision_events_selftest

设计要点（架构真相纪律）：测试帧序列必须保证相邻帧 IoU 足够高
（真实视频相邻帧目标位移小，ByteTrack 才能关联同一 ID）。合成数据若
相邻帧框无重叠（IoU≈0）ByteTrack 会失配——这是数据问题，不是代码 bug。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.supervision_events import (
    VideoEventAnalyzer, TrackBox, LineCfg, ZoneCfg, build_event_analyzer, _HAS_SV,
)

ok = True
def chk(name, cond, extra=""):
    global ok
    ok = ok and cond
    print(f"[{'OK' if cond else 'FAIL'}] {name} {extra}")

if not _HAS_SV:
    print("supervision 未安装，无法跑真探针（降级路径见 events.py）。请 pip install supervision。")
    sys.exit(2)

if __name__ == "__main__":
    # ---- 测试①：跟踪持久 ID + 越线（固定 x，纯 y 下移，IoU 高）----
    line = LineCfg(id="A", p1=(100, 200), p2=(400, 200))
    an = VideoEventAnalyzer(lines=[line])
    cross_seq = [
        [TrackBox((150, 100, 180, 140), 0, "person", 0.9)],   # 线上方
        [TrackBox((150, 125, 180, 165), 0, "person", 0.9)],
        [TrackBox((150, 150, 180, 190), 0, "person", 0.9)],   # 接近线
        [TrackBox((150, 175, 180, 215), 0, "person", 0.9)],   # 跨过 y=200
        [TrackBox((150, 200, 180, 240), 0, "person", 0.9)],   # 线下方
        [TrackBox((150, 225, 180, 265), 0, "person", 0.9)],
    ]
    ids = []
    for k, boxes in enumerate(cross_seq):
        snap = an.update(boxes, t=float(k))
        ids += [t["tracker_id"] for t in snap["tracked"]]
    chk("① 跟踪持续同 ID", len(set(ids)) == 1 and len(ids) >= 4, f"ids={ids}")
    lc = an.summary()["line_counts"]["A"]
    chk("① 越线聚合计数>0", lc["in"] + lc["out"] >= 1, f"in={lc['in']} out={lc['out']}")
    chk("① 越线事件产出", an.summary()["n_cross_events"] >= 1,
        f"n={an.summary()['n_cross_events']}")

    # ---- 测试②：区域停留时长（框固定在区域内，停留后离开，IoU 高）----
    zone = ZoneCfg(id="entrance", polygon=[(300, 300), (500, 300), (500, 500), (300, 500)])
    an2 = VideoEventAnalyzer(zones=[zone])
    dwell_seq = [
        [TrackBox((350, 320, 410, 380), 0, "person", 0.9)],   # 区域内
        [TrackBox((350, 330, 410, 390), 0, "person", 0.9)],   # 停留
        [TrackBox((150, 180, 180, 220), 0, "person", 0.9)],   # 离开区域
    ]
    for k, boxes in enumerate(dwell_seq):
        an2.update(boxes, t=float(k))
    dw = an2.summary()["n_dwell_events"]
    chk("② 区域停留事件产出", dw >= 1, f"n={dw}")
    if dw >= 1:
        last = an2.dwell_events[-1].to_dict()
        chk("② 停留时长>0", last["dwell_sec"] > 0, f"dwell={last['dwell_sec']}s")

    # ---- 测试③：降级工厂（强制 sv=None，应回退 classcount，不崩）----
    import perception.supervision_events as m
    _sv, m.sv, m._HAS_SV = m.sv, None, False
    fb = build_event_analyzer()
    snap = fb.update([TrackBox((10, 10, 50, 50), 0, "person", 0.9)])
    chk("③ 降级工厂不崩", snap["active_backend"] == "classcount-fallback", snap["active_backend"])
    m.sv, m._HAS_SV = _sv, True  # 还原

    print("\n%s" % ("✅ ALL PASS" if ok else "❌ FAIL"))
    sys.exit(0 if ok else 1)
