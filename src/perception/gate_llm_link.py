"""
gate_llm_link.py —— 把事件闸(EventGate)的唤醒信令真·运输给大模型，闭合
「看见 → 唤醒 → 研判」全链路（MGA 感知层 B 与真实 System2 的接线）。

A 层(supervision_events) / C 层(frigate_source) 产结构化事件 → EventGate 打分
p≥0.70 开闸 → 本模块把 WakeSignal 压缩成 prompt → 经 llm_bridge 送真 LLM →
拿回研判文本。整条链路不再"代码写完就算接通"：transport 真发起 HTTP 调用，
返回非 None 才标记 transport_source="llm"，否则 pending。

设计纪律（对齐 MGA 全局）：
  · 真接通：transport 回调真调用 llm_bridge.build_llm() 产出的真后端（hy3/
    openai 兼容），返回大模型原文。绝不用本地规则冒充大模型。
  · 诚实标记：无密钥 / 调用失败 → 返回 None，gate 标记 pending，不崩不冒充。
  · is_real 透传：build_llm 的 is_real 标志如实反映是真大模型还是 Mock 兜底。
  · 密钥纪律：只从环境变量( MGA_LLM_API_KEY )或 .env.local 读，绝不写进代码。

用法：
    from gate_llm_link import make_llm_transport, run_closed_loop
    gate.set_llm_transport(make_llm_transport(backend="openai"))   # 真 LLM
    run_closed_loop()                                               # 跑完整闭环 + 打印
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional

# 让本模块在 src/perception 直跑时也能 import 到同包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_gate import EventGate, WakeSignal  # noqa: E402


# ===========================================================================
# 运输桥：把 WakeSignal 送给真 LLM，返回研判文本
# ===========================================================================
def make_llm_transport(backend: str = "openai",
                       model: str = None,
                       base_url: str = None,
                       api_key: str = None,
                       mock_fallback: bool = True) -> Callable[[WakeSignal], Optional[str]]:
    """返回 EventGate.set_llm_transport 所需的回调。

    backend: 传给 llm_bridge.build_llm 的后端名（openai/hy3/ollama/...）
    密钥：优先入参 api_key，否则读环境变量 MGA_LLM_API_KEY / OPENAI_API_KEY，
          或 .env.local（build_llm 内部 _load_dotenv 已处理）。绝不写进代码。
    mock_fallback: 真后端不可用（无密钥等）时退 MockLLM，闭环仍跑但 is_real=False。

    回调返回：大模型研判文本(str, 非 None=已处理) / None(异常→gate 标记 pending)。
    """
    from llm_bridge import build_llm, MockLLM

    bridge = None
    try:
        bridge = build_llm(backend=backend, model=model, base_url=base_url,
                           api_key=api_key)
    except Exception as e:
        print(f"[gate_llm_link] ⚠️ build_llm 失败：{type(e).__name__}: {e}")

    if bridge is None:
        if mock_fallback:
            bridge = MockLLM()
            print("[gate_llm_link] 无真后端→MockLLM 兜底(is_real=False，闭环仍跑)")
        else:
            return lambda sig: None

    backend_label = type(bridge).__name__
    is_real = getattr(bridge, "is_real", False)
    print(f"[gate_llm_link] 运输桥就绪：backend={backend_label}, is_real={is_real}")

    def transport(sig: WakeSignal) -> Optional[str]:
        prompt = EventGate.build_llm_payload(sig)
        try:
            out = bridge.respond(prompt)
            if not out:
                return None
            return out
        except Exception as e:
            print(f"[gate_llm_link] ⚠️ LLM 调用异常，标记 pending："
                  f"{type(e).__name__}: {e}")
            return None

    transport._bridge = bridge
    return transport


# ===========================================================================
# 闭环 demo / 真探针：A(supervision 真事件流) → B(事件闸) → 真 LLM
# ===========================================================================
def run_closed_loop(backend: str = "openai",
                    key_file: str = None,
                    use_frigate: bool = False,
                    n_frames: int = 30) -> dict:
    """跑完整「看见→唤醒→研判」闭环并打日志。返回 {wake, analysis, backend}。

    key_file: 可选，含 HY3_API_KEY / MGA_LLM_API_KEY 的本地文件（仅读不打印），
              用于把密钥注入环境变量（MGA_LLM_API_KEY）后调真 LLM。绝不回显值。
    """
    # 1) 若有 key 文件且环境变量未设 → 安全读取注入（不打印）
    if key_file and not os.getenv("MGA_LLM_API_KEY"):
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k in ("HY3_API_KEY", "MGA_LLM_API_KEY") and v and k not in os.environ:
                        os.environ["MGA_LLM_API_KEY"] = v
                        break
        except FileNotFoundError:
            print(f"[run_closed_loop] ⚠️ key 文件不存在：{key_file}")

    # 2) 接真 LLM 运输
    transport = make_llm_transport(backend=backend)
    bridge = getattr(transport, "_bridge", None)
    is_real = getattr(bridge, "is_real", False)

    # 3) 选上游：A(supervision) 或 C(frigate 合成)
    wakes: List[WakeSignal] = []
    gate = EventGate(on_wake=lambda s: wakes.append(s))
    gate.set_llm_transport(transport)

    if use_frigate:
        from frigate_source import FrigateSource, FrigateConfig
        src = FrigateSource(FrigateConfig(cameras=["cam1"]), gate)
        # 合成 Frigate 真 payload：某人进入区域并停留 → 应触发 zone_dwell
        t0 = 0.0
        src.ingest("frigate/cam1/events/abc/updates",
                   {"after": {"camera": "cam1", "id": "T1", "label": "person",
                              "current_zones": [], "box": {"x": 0.4, "y": 0.4,
                                                           "w": 0.1, "h": 0.1}}},
                   t=t0)
        for k in range(1, n_frames):
            t = float(k)
            zones = ["entrance"] if k < n_frames - 2 else []
            src.ingest("frigate/cam1/events/abc/updates",
                       {"after": {"camera": "cam1", "id": "T1", "label": "person",
                                  "current_zones": zones,
                                  "box": {"x": 0.4, "y": 0.4, "w": 0.1, "h": 0.1}}},
                       t=t)
    else:
        from supervision_events import (build_event_analyzer, LineCfg, ZoneCfg, TrackBox)
        line = LineCfg(id="A", p1=(100, 200), p2=(400, 200))
        zone = ZoneCfg(id="entrance",
                       polygon=[(300, 300), (500, 300), (500, 500), (300, 500)])
        an = build_event_analyzer(lines=[line], zones=[zone])
        # 一段轨迹：上方→跨线→进入区域→连续停留(累计长停留)→离开
        seq = [
            [TrackBox((150, 100, 180, 140), 0, "person", 0.9)],
            [TrackBox((150, 180, 180, 220), 0, "person", 0.9)],
            [TrackBox((150, 260, 180, 300), 0, "person", 0.9)],
        ]
        for _ in range(n_frames - 6):  # 在区域内持续停留，累计 >4s → 触发
            seq.append([TrackBox((350, 350, 380, 390), 0, "person", 0.9)])
        seq.append([TrackBox((150, 260, 180, 300), 0, "person", 0.9)])  # 离开
        for k, boxes in enumerate(seq):
            from event_gate import feed_analyzer_frame
            feed_analyzer_frame(an, gate, boxes, t=float(k))

    # 4) 取最近一次唤醒 + 对应研判（analysis 已随 transport 写入 WakeSignal）
    wake = wakes[-1] if wakes else (gate.fires[-1] if gate.fires else None)
    analysis = wake.analysis if (wake is not None and wake.analysis) else None
    return {"wake": wake, "analysis": analysis, "is_real": is_real,
            "backend": type(bridge).__name__ if bridge else None,
            "n_fired": gate.n_fired, "n_pending": gate.summary()["n_pending"]}


if __name__ == "__main__":
    import argparse

    print("=" * 64)
    print("MGA 闭环：看见(A/C) → 事件闸(B) → 真 LLM 研判")
    print("=" * 64)

    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="openai")
    p.add_argument("--key-file", default=None,
                   help="含 HY3_API_KEY 的本地文件（仅读取注入环境变量，不打印）")
    p.add_argument("--frigate", action="store_true", help="用 C 层 Frigate 合成流")
    p.add_argument("--frames", type=int, default=30)
    args = p.parse_args()

    res = run_closed_loop(backend=args.backend, key_file=args.key_file,
                          use_frigate=args.frigate, n_frames=args.frames)

    print(f"\n后端={res['backend']}  is_real={res['is_real']}  "
          f"唤醒={res['n_fired']}  待处理={res['n_pending']}")
    if res["wake"] is None:
        print("⚠️ 未产生任何唤醒（检查上游事件流/阈值）")
    else:
        w = res["wake"]
        print(f"触发事件：{w.trigger_event}")
        print(f"打分={w.score:.2f}  来源={w.transport_source}  pending={w.pending}")
        if w.transport_source == "llm" and res["analysis"]:
            print("\n--- 大模型研判原文 ---")
            print(res["analysis"])
            print("--- 结束 ---")
        elif w.pending:
            print("（LLM 不可用 → pending：闭环逻辑通，但本次未真调大模型）")
    print("\n结论：", "真·闭环已跑（大模型实回研判）" if (res["is_real"] and res["analysis"])
          else "闭环接线完成（未接真 LLM，请带 --key-file 重试）")
