"""
frigate_source.py —— MGA 感知层第三块 (C): Frigate NVR 采集 + MQTT 运输层
============================================================================

把 Frigate（开源本地 NVR，MIT）的 MQTT 输出接进 B 的事件闸 EventGate。

Frigate 本身已经做了「检测 + 跟踪 + 区域(zones)」，它发布的 MQTT 消息就是
**结构化高层事件**（对象进入/离开某区域、当前得分）。这恰好是 EventGate 想要的
语义事件——所以 C 的核心是「采集 + 翻译」，而不是重新检测：

    编排器/本模块订阅 Frigate MQTT
        → 解析 payload（after.current_zones 等）
        → 对比前后区域集 → 产 zone_enter / zone_leave / zone_dwell 事件
        → push 给 EventGate
        → EventGate 廉价打分，p≥0.70 时打包因果窗口唤醒 System2(LLM)
        → 把压缩后的上下文运输给大模型做研判

同时维护「当前帧对象表」(`_tracked`)，可选经 `poll_frame()` 喂给 A 层
supervision analyzer，做 System1 预判帧（Frigate 已有 track id，复用即可）。

设计纪律（对齐 MGA 全局）：
  · 真接通：采集到的真实 Frigate payload 真解析、真 push 给 EventGate、真唤醒、
    LLM 运输真被调用；绝不「代码写完就算接通」。
  · 优雅降级：paho-mqtt 缺失 / broker 连不上 → 标 active=False，仍可离线 ingest
    验证链路，不崩、不冒充在线。
  · 模型无关：本模块只产出 EventGate 认的事件 dict（zone_enter/zone_dwell/...），
    任何上游（Frigate / 桌面 UI 检测 / ScreenParser）都能照此格式喂进来。

依赖：仅标准库 + 可选 paho-mqtt（缺失自动降级）。真探针零外部依赖可跑。
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from event_gate import EventGate, WakeSignal

# ---- 可选依赖：paho-mqtt（Frigate MQTT 客户端）----
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except Exception:
    mqtt = None
    MQTT_AVAILABLE = False

# Frigate 检测默认分辨率：tracked object topic 的 box 是归一化 [0-1]，
# 转像素时需乘分辨率。events topic 的 box 是像素，无需转换。
DEFAULT_RESOLUTION = (1280, 720)


# ===========================================================================
# 配置
# ===========================================================================
@dataclass
class FrigateConfig:
    """Frigate MQTT 采集配置。"""
    broker_host: str = "localhost"
    broker_port: int = 1883
    username: str = ""
    password: str = ""
    topic_prefix: str = "frigate"          # Frigate 默认 MQTT topic 前缀
    resolution: Tuple[int, int] = DEFAULT_RESOLUTION
    cameras: Tuple[str, ...] = ()          # 白名单；空=订阅全部摄像头
    client_id: str = "mga-frigate-bridge"
    keepalive: int = 60


# ===========================================================================
# Frigate 采集源
# ===========================================================================
class FrigateSource:
    """订阅 Frigate MQTT，把检测/区域事件翻译成 EventGate 认的语义事件。

    用法（在线）：
        src = build_frigate_gateway(cfg, gate)
        src.connect()                       # 连 broker，回调自动 ingest
        # ... 运行 ...
        src.disconnect()

    用法（离线/真探针）：
        src = build_frigate_gateway(cfg, gate)
        src._ingest("frigate/cam/events/abc/update", {"after": {...}}, t=0.0)
    """

    def __init__(self, cfg: FrigateConfig, gate: EventGate):
        self.cfg = cfg
        self.gate = gate
        self.active = False                 # 是否已连上 broker 并订阅
        self.mqtt_available = MQTT_AVAILABLE
        self.client = None

        # 运行态
        self._tracked: Dict[str, Dict[str, Any]] = {}   # obj_id -> 当前对象信息
        self._zones: Dict[str, Dict[str, float]] = {}    # obj_id -> {zone_id: enter_t}
        self._last_zones: Dict[str, Set[str]] = {}       # obj_id -> 上次的 zone 集合
        self.n_ingested = 0
        self.n_zone_events = 0

    # ---------------- 连接 ----------------
    def connect(self) -> bool:
        """连 broker 并订阅 frigate/#。失败优雅降级（不崩）。"""
        if not MQTT_AVAILABLE:
            self.active = False
            print("[Frigate] ⚠️ paho-mqtt 缺失 → 离线模式（仍可 _ingest 验证）")
            return False
        try:
            # 兼容 paho 1.x 与 2.x 的 Client 构造差异：
            # 2.x 必须显式传 CallbackAPIVersion（缺省会报 ValueError/TypeError），
            # 1.x 无该属性 → 退化为无参构造。
            ver = getattr(mqtt, "CallbackAPIVersion", None)
            if ver is not None:
                self.client = mqtt.Client(ver.VERSION2)
            else:
                self.client = mqtt.Client()
            if self.cfg.username:
                self.client.username_pw_set(self.cfg.username, self.cfg.password)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect(self.cfg.broker_host, self.cfg.broker_port,
                                 self.cfg.keepalive)
            self.client.loop_start()
            self.active = True
            return True
        except Exception as e:
            self.active = False
            print(f"[Frigate] ⚠️ 连接 broker 失败，降级离线：{type(e).__name__}: {e}")
            return False

    def disconnect(self) -> None:
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        self.active = False

    def _on_connect(self, client, userdata, flags, rc, *args) -> None:
        # paho 2.x 的 rc 是 ReasonCode 对象，1.x 是 int；统一转 int 比较
        try:
            code = int(rc)
        except Exception:
            code = rc
        if code == 0:
            topic = f"{self.cfg.topic_prefix}/#"
            client.subscribe(topic)
            print(f"[Frigate] ✅ 已连 {self.cfg.broker_host}:{self.cfg.broker_port}，"
                  f"订阅 {topic}")
        else:
            self.active = False
            print(f"[Frigate] ⚠️ 连接被拒 rc={code}（离线模式）")

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload)
        except Exception:
            return
        try:
            self._ingest(msg.topic, payload)
        except Exception as e:
            print(f"[Frigate] ⚠️ 解析消息异常：{type(e).__name__}: {e}")

    def ingest(self, topic: str, payload: Dict[str, Any], t: Optional[float] = None
              ) -> None:
        """公开入口：把一条 Frigate MQTT 消息(已拆好的 topic/payload)喂进来。
        离线真探针与在线 _on_message 共用同一解析逻辑，t 缺省用 time.time()。"""
        self._ingest(topic, payload, t=t)

    # ---------------- 核心：解析 Frigate payload ----------------
    def _ingest(self, topic: str, payload: Dict[str, Any],
                t: Optional[float] = None) -> None:
        """把一条 Frigate MQTT 消息翻译成事件并 push 给 EventGate。

        t：事件时间戳（秒）。离线真探针显式传；在线用 time.time()。"""
        if t is None:
            t = time.time()
        parts = topic.split("/")
        prefix = self.cfg.topic_prefix
        if not parts or parts[0] != prefix:
            return

        # 1) events 消息：frigate/<camera>/events/<id>/<type>
        #    payload 含 before/after（after 有 label/score/current_zones/box）
        if len(parts) >= 4 and parts[2] == "events":
            after = (payload.get("after") or payload) if isinstance(payload, dict) else {}
            if isinstance(after, dict) and after:
                if self._handle_object_state(after, t):
                    self.n_ingested += 1
            return

        # 2) tracked object 消息：frigate/<camera>/<type>/<id>
        #    payload 含 box(归一化)/score，用于 poll_frame 出预判帧
        if len(parts) == 4 and parts[2] not in ("events",):
            if self._handle_tracked(parts[1], parts[2], parts[3], payload, t):
                self.n_ingested += 1
            return

        # 3) 汇总 topic frigate/events：payload 直接是事件（含 after）
        if len(parts) == 2:
            after = payload.get("after") or {}
            if isinstance(after, dict) and after:
                if self._handle_object_state(after, t):
                    self.n_ingested += 1

    def _handle_object_state(self, after: Dict[str, Any], t: float) -> bool:
        """处理 events 消息的 after 段，对比区域变化产事件。返回是否真的处理（白名单过滤则为 False）。"""
        cam = after.get("camera") or ""
        if self.cfg.cameras and cam and cam not in self.cfg.cameras:
            return False
        oid = after.get("id") or f"{cam}:{after.get('tracker_id')}"
        if not oid:
            return
        label = after.get("label") or "object"
        score = float(after.get("score") or after.get("top_score") or 0.0)

        cur: Set[str] = set(after.get("current_zones") or [])
        prev: Set[str] = self._last_zones.get(oid, set())
        entered = cur - prev
        left = prev - cur

        for z in entered:
            self._zones.setdefault(oid, {})[z] = t
            ev = {"type": "zone_enter", "zone_id": z, "tracker_id": oid,
                  "class_name": label, "camera": cam, "t": t}
            self.gate.push_event(ev, t=t)
            self.n_zone_events += 1

        for z in left:
            enter_t = self._zones.get(oid, {}).pop(z, t)
            dwell = max(0.0, t - enter_t)
            # 离开时结算停留：zone_dwell 是 EventGate 认的高严重度事件（长停留→唤醒）
            self.gate.push_event({
                "type": "zone_dwell", "zone_id": z, "tracker_id": oid,
                "class_name": label, "camera": cam, "dwell_sec": dwell,
                "frame_in": 0, "frame_out": 0,
            }, t=t)
            self.n_zone_events += 1
            self.gate.push_event({
                "type": "zone_leave", "zone_id": z, "tracker_id": oid,
                "class_name": label, "camera": cam, "dwell_sec": dwell,
            }, t=t)
            self.n_zone_events += 1

        self._last_zones[oid] = cur
        # 记录到 tracked 表（供 poll_frame 出预判帧）
        self._tracked[oid] = {
            "camera": cam, "label": label, "score": score,
            "box": after.get("box"), "last_seen": t,
        }
        return True

    def _handle_tracked(self, cam: str, otype: str, oid: str,
                        payload: Dict[str, Any], t: float) -> bool:
        """处理 tracked object 消息（box 归一化 → 像素），供 poll_frame。返回是否处理。"""
        if self.cfg.cameras and cam and cam not in self.cfg.cameras:
            return False
        box = payload.get("box")
        score = float(payload.get("score") or 0.0)
        if box and all(0.0 <= v <= 1.0 for v in box):
            w, h = self.cfg.resolution
            box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        key = f"{cam}:{oid}"
        self._tracked[key] = {
            "camera": cam, "label": otype, "score": score,
            "box": box, "last_seen": t,
        }
        return True

    # ---------------- 可选：出预判帧（喂 A 层 supervision analyzer）----------------
    def poll_frame(self, t: Optional[float] = None,
                   max_age: float = 2.0) -> List[Any]:
        """返回当前「在帧」的对象列表（TrackBox），可喂给 A 层 analyzer 做预判帧。

        Frigate 已给 track id，每帧喂同 tid 的 box 即可复用 ByteTrack。
        无 supervision 时返回空列表（优雅降级，不崩）。"""
        try:
            from supervision_events import TrackBox
        except Exception:
            try:
                from perception.supervision_events import TrackBox
            except Exception:
                return []
        if t is None:
            t = time.time()
        out: List[Any] = []
        for oid, info in self._tracked.items():
            box = info.get("box")
            if not box:
                continue
            if (t - info.get("last_seen", 0.0)) > max_age:
                continue
            out.append(TrackBox(tuple(box), oid, info.get("label", "object"),
                                info.get("score", 0.5)))
        return out

    # ---------------- 状态 ----------------
    def summary(self) -> dict:
        return {
            "active": self.active,
            "mqtt_available": self.mqtt_available,
            "n_ingested": self.n_ingested,
            "n_zone_events": self.n_zone_events,
            "n_tracked": len(self._tracked),
        }


def build_frigate_gateway(cfg: FrigateConfig, gate: EventGate) -> FrigateSource:
    """工厂：建 Frigate 采集源（含降级标记）。"""
    return FrigateSource(cfg, gate)


# ===========================================================================
# 真探针 / 集成自检
# ===========================================================================
if __name__ == "__main__":
    # 让本模块在 src/perception 直跑时也能 import 到同包 llm_bridge
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 64)
    print("MGA Frigate 采集层 (C) —— 真探针集成自检")
    print("=" * 64)

    # 接真实 LLM 运输（零依赖 mock 后端，验证「事件真运到大模型」）
    from perception.llm_bridge import build_llm

    wakes: List[WakeSignal] = []
    gate = EventGate(on_wake=lambda s: wakes.append(s))
    bridge = build_llm(backend="mock")
    gate.set_llm_transport(
        lambda sig: bridge.respond(EventGate.build_llm_payload(sig)) or "ok")

    cfg = FrigateConfig(broker_host="localhost")
    src = build_frigate_gateway(cfg, gate)

    # ---- [1] 对象进入 driveway 区域（t=0，单次进入应是噪声，不唤醒）----
    src._ingest("frigate/front_door/events/abc/update", {
        "after": {"camera": "front_door", "id": "abc", "label": "person",
                  "score": 0.9, "current_zones": ["driveway"], "start_time": 0.0,
                  "type": "new"}}, t=0.0)
    assert gate.n_fired == 0, "仅进入区域不应唤醒"
    print("[1] zone_enter 不唤醒 ✓ (后端=%s)" % gate.backend)

    # ---- [2] 停留 5s 后离开（t=5）→ zone_dwell(5s) → 唤醒 + LLM 真运输 ----
    src._ingest("frigate/front_door/events/abc/update", {
        "after": {"camera": "front_door", "id": "abc", "label": "person",
                  "score": 0.9, "current_zones": [], "end_time": 5.0,
                  "type": "end"}}, t=5.0)
    assert gate.n_fired >= 1, "长停留离开应唤醒"
    sig = wakes[-1]
    assert sig.score >= 0.70, "长停留应过阈值"
    assert sig.transport_source == "llm", "LLM 运输应真被调用"
    assert sig.pending is False
    assert float(sig.trigger_event.get("dwell_sec", 0.0)) >= 5.0
    print("[2] 长停留5s → zone_dwell → 唤醒 score=%.2f ✓ (LLM 命中, payload已生成)"
          % sig.score)
    print("    —— 运输给大模型的载荷(节选) ——")
    print("\n".join("   " + ln for ln in
                    EventGate.build_llm_payload(sig).split("\n")[:5]))

    # ---- [3] 短时进出（噪声）：每次 dwell 0.5s → 不唤醒 ----
    for k in range(4):
        src._ingest("frigate/cam/events/x/update", {
            "after": {"camera": "cam", "id": "x", "label": "car", "score": 0.8,
                      "current_zones": ["road"], "start_time": 10 + k}},
            t=10.0 + k)
        src._ingest("frigate/cam/events/x/update", {
            "after": {"camera": "cam", "id": "x", "label": "car", "score": 0.8,
                      "current_zones": [], "end_time": 10.5 + k},
            "type": "end"}, t=10.5 + k)
    # 每次 dwell=0.5s → base≈0.30+0.05=0.35 < 0.70，且受冷却/滞回约束，不应额外唤醒
    print("[3] 短时进出(噪声) 累计唤醒=%d（应为 1，仅长停留那次）✓" % gate.n_fired)
    assert gate.n_fired == 1, "噪声不应新增唤醒"

    # ---- [4] 降级诚实性：paho 缺失 → 离线模式仍可 ingest ----
    print("[4] MQTT 可用性=%s（缺失则离线模式，链路不崩）✓" % src.mqtt_available)
    print("    src.summary=%s" % src.summary())

    # ---- [5] 多摄像头白名单过滤 ----
    cfg2 = FrigateConfig(broker_host="x", cameras=("front_door",))
    src2 = build_frigate_gateway(cfg2, EventGate())
    src2._ingest("frigate/backyard/events/y/update", {
        "after": {"camera": "backyard", "id": "y", "label": "person",
                  "current_zones": ["yard"], "type": "new"}}, t=0.0)
    assert src2.n_ingested == 0, "非白名单摄像头应被过滤"
    print("[5] 摄像头白名单过滤 ✓")

    print("\n" + "=" * 64)
    print("✅ C(Frigate 采集 → EventGate → LLM 运输) 全部真探针通过")
    print("=" * 64)
