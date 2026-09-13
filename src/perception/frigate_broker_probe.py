"""
frigate_broker_probe.py —— 真 Frigate MQTT broker 联调探针
====================================================================

目标：不走"代码写完就算接通"，而是真起一个 MQTT broker（本地 TCP，真协议握手），
让 FrigateSource.connect() 真连上去，再从 publisher 发出**真 Frigate 格式**的
MQTT 消息，断言：

    真 publisher → 真 broker → FrigateSource 订阅回调 → 解析 zone 变化
                → push 给 EventGate → 长停留 ≥4s → 闸门开 → 唤醒 + LLM 运输被调用

这证明"采集层"在真 broker 下是真接通的，不是离线 ingest 模拟。

broker 用 gmqtt（纯 Python，仅测试用；生产里 Frigate 自带 MQTT / Mosquitto）。
Frigate 一侧由本脚本里的 paho publisher 模拟（消息格式严格按 Frigate MQTT 规范）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time

# 让本脚本在 src/perception 直跑也能 import 同包
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
sys.path.insert(0, _SRC)

from frigate_source import FrigateConfig, build_frigate_gateway   # noqa: E402
from event_gate import EventGate, WakeSignal                      # noqa: E402
from perception.llm_bridge import build_llm                       # noqa: E402


def _start_broker() -> tuple:
    """后台线程起一个真 amqtt broker（本地 TCP，真 MQTT 协议握手）。

    返回 (stop_fn, ready_event)。Frigate 生产环境用 Mosquitto/Frigate 自带 MQTT，
    这里 amqtt 仅作真 broker 被测端，证明采集层在真 broker 下确为"接通"。
    """
    from amqtt.broker import Broker

    config = {
        "listeners": {
            "default": {
                "type": "tcp",
                "bind": "0.0.0.0:1883",
            }
        },
        "auth": {
            "allow-anonymous": True,
        },
    }
    loop = asyncio.new_event_loop()
    broker_ref: dict = {}
    stop_ev = asyncio.Event()
    ready = threading.Event()

    async def _main():
        # amqtt Broker 构造需在运行中的 loop 内（它会取 get_running_loop）
        broker = Broker(config, loop=loop)
        broker_ref["b"] = broker
        await broker.start()
        ready.set()
        await stop_ev.wait()

    loop.create_task(_main())

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    ready.wait(timeout=15)
    assert ready.is_set(), "amqtt broker 未在 15s 内起来"

    def stop():
        b = broker_ref.get("b")
        if b is not None:
            try:
                asyncio.run_coroutine_threadsafe(b.shutdown(), loop)
            except Exception:
                pass
        # 先让 _main 协程优雅退出，再停 loop（避免 "Task destroyed" 噪声）
        loop.call_soon_threadsafe(stop_ev.set)
        loop.call_soon_threadsafe(loop.stop)

    return stop, ready


def _wait(cond, timeout: float = 10.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


def main() -> int:
    print("=" * 64)
    print("MGA Frigate 采集层 (C) —— 真 MQTT broker 联调探针")
    print("=" * 64)

    stop_broker, _ = _start_broker()
    print("[broker] amqtt 真 broker 已起 localhost:1883 ✅")

    # ---- 接 LLM 运输（mock 后端即可：本探针只验证 broker 链路，
    #      LLM 真回研判已在 prior step 用真 hy3 验证过）----
    wakes: list = []
    gate = EventGate(on_wake=lambda s: wakes.append(s))
    bridge = build_llm(backend="mock")
    gate.set_llm_transport(
        lambda sig: bridge.respond(EventGate.build_llm_payload(sig)) or "ok")

    cfg = FrigateConfig(broker_host="localhost", broker_port=1883)
    src = build_frigate_gateway(cfg, gate)

    # ---- 1) 真连 broker ----
    ok = src.connect()
    assert ok is True, "connect() 应返回 True（真连成功）"
    assert _wait(lambda: src.active, timeout=8), "应在 8s 内 active=True"
    print("[1] FrigateSource.connect() 真连 broker 成功，active=%s ✅" % src.active)
    print("    topic=%s/# 已订阅" % cfg.topic_prefix)

    # ---- 2) 模拟 Frigate 发真格式消息：人进 driveway 区域 ----
    import paho.mqtt.client as mqtt
    pub = mqtt.Client()
    pub.connect("localhost", 1883, 60)
    pub.loop_start()

    enter_payload = {
        "before": {"camera": "front_door", "id": "abc", "label": "person",
                    "current_zones": []},
        "after": {"camera": "front_door", "id": "abc", "label": "person",
                   "score": 0.92, "current_zones": ["driveway"], "type": "new"},
    }
    topic = "frigate/front_door/events/abc/update"
    pub.publish(topic, json.dumps(enter_payload), qos=1).wait_for_publish()
    print("[2] 已发 Frigate enter 消息（person 进入 driveway）")

    # ---- 3) 停留 5s（真实墙钟，事件闸用接收时刻算 dwell）----
    time.sleep(5.0)

    leave_payload = {
        "before": {"camera": "front_door", "id": "abc", "label": "person",
                    "current_zones": ["driveway"]},
        "after": {"camera": "front_door", "id": "abc", "label": "person",
                   "score": 0.92, "current_zones": [], "type": "end"},
    }
    pub.publish(topic, json.dumps(leave_payload), qos=1).wait_for_publish()
    print("[3] 已发 Frigate leave 消息（人离开 driveway，dwell≈5s）")

    pub.loop_stop()
    pub.disconnect()

    # ---- 4) 断言：真消息驱动了 zone_dwell 唤醒 ----
    assert _wait(lambda: gate.n_fired >= 1, timeout=8), \
        "真 MQTT 消息应驱动 zone_dwell 唤醒"
    sig = wakes[-1]
    assert sig.trigger_event.get("type") == "zone_dwell", \
        "触发事件应为 zone_dwell，实际=%s" % sig.trigger_event.get("type")
    assert float(sig.trigger_event.get("dwell_sec", 0.0)) >= 4.0, \
        "dwell 应 ≥4s（真 broker 真实接收时刻），实际=%.2f" % \
        float(sig.trigger_event.get("dwell_sec", 0.0))
    assert sig.score >= 0.70, "score 应过阈值，实际=%.3f" % sig.score
    assert sig.transport_source == "llm", \
        "LLM 运输应被真调用，实际=%s" % sig.transport_source
    print("[4] ✅ 真 MQTT 链路驱动唤醒：trigger=%s, dwell=%.2fs, score=%.3f, "
          "transport=%s" % (sig.trigger_event.get("type"),
                            float(sig.trigger_event.get("dwell_sec", 0.0)),
                            sig.score, sig.transport_source))

    # ---- 5) 断言采集态：确实从 broker 收了消息并解析出 zone 事件 ----
    assert src.n_ingested >= 2, "应至少 ingest 2 条真消息，实际=%d" % src.n_ingested
    assert src.n_zone_events >= 2, \
        "应至少产 2 个 zone 事件(enter+leave)，实际=%d" % src.n_zone_events
    print("[5] 采集态: n_ingested=%d, n_zone_events=%d ✅" %
          (src.n_ingested, src.n_zone_events))

    # ---- 6) 额外：tracked object 消息也能经 broker 进 poll_frame（预判帧喂 A 层）----
    pub2 = mqtt.Client()
    pub2.connect("localhost", 1883, 60)
    pub2.loop_start()
    tracked_topic = "frigate/cam1/person/T1"
    pub2.publish(tracked_topic, json.dumps(
        {"box": [0.4, 0.4, 0.1, 0.1], "score": 0.88}), qos=1).wait_for_publish()
    pub2.loop_stop()
    pub2.disconnect()
    # 直接断言 poll_frame 真返回带框对象（要求 tracked 消息真实送达且仍"在帧"）
    assert _wait(lambda: len(src.poll_frame()) >= 1, timeout=5), \
        "tracked 消息应经 broker 真实送达并出现在 poll_frame 中"
    boxes = src.poll_frame()
    print("[6] tracked object 消息经 broker 进 poll_frame：n_tracked=%d, "
          "poll_frame 返回 %d 个框 ✅" % (src.summary()["n_tracked"], len(boxes)))

    # 清理
    src.disconnect()
    stop_broker()

    print("\n" + "=" * 64)
    print("✅ 真 Frigate broker 联调全通过："
          "connect()→订阅→真消息→zone 解析→事件闸唤醒→LLM 运输 全部真跑通")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
