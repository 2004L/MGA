"""
sam3_worker.py —— SAM3 常驻推理 worker（跨 Python 环境调用）
============================================================================
为什么要有它：
    sam3 + triton-windows + torchvision(cu132) 只装在 **.venv-sam3**（D 盘），
    而 detector.py / 主程序跑在 **base python**（C 盘只剩几百 MB，装不下也不敢装）。
    两边是不同的解释器，没法直接 import。

    → 起一个常驻子进程，用 stdin/stdout 一行一个 JSON 通信：
      · 模型**只加载一次**（否则每帧重启进程 = 白等 1~2 分钟）
      · base python 侧零依赖，worker 挂了自动降级，不拖垮主链路

协议（一行 JSON in → 一行 JSON out）：
    {"cmd":"ping"}
        → {"ok":true}
    {"cmd":"detect_named","image":"<路径>","names":["icon","button"],"conf":0.5}
        → {"ok":true,"elements":[{"label":..,"bbox":[x1,y1,x2,y2],"conf":..}],
           "centroids":[[x,y], ...]}
    {"cmd":"quit"}
        → 退出

用法（通常由 SAM3Locator 自动拉起，不必手跑）：
    .venv-sam3/Scripts/python.exe src/perception/sam3_worker.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _els_to_json(els, centroids):
    out = []
    for e, c in zip(els, centroids):
        out.append({
            "id": e.id,
            "label": e.label,
            "bbox": list(e.bbox),
            "conf": e.conf,
            "centroid": list(c) if c else None,
        })
    return out


def main() -> int:
    from sam3_locator import SAM3Locator

    # 关键：sam3_locator 的 verbose 打印 + 第三方库(torch/sam3)的加载日志
    # 都会写到 stdout，而 stdout 是 JSON 协议管道。加载期间把 stdout 临时
    # 重定向到 stderr，避免污染协议（父进程读第一行就 JSON 解析失败）。
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    loc = SAM3Locator(verbose=False)
    ready = loc.available
    load_error = loc.load_error
    sys.stdout = real_stdout  # 恢复，之后只有协议 JSON 走 stdout

    if not ready:
        # 起不来也别装死：立刻回报，让调用方降级
        print(json.dumps({"ok": False, "fatal": True,
                          "error": f"SAM3 不可用: {load_error}"}), flush=True)
        return 1

    print(json.dumps({"ok": True, "ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"bad json: {e}"}), flush=True)
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "quit":
                print(json.dumps({"ok": True, "bye": True}), flush=True)
                return 0
            if cmd == "ping":
                print(json.dumps({"ok": True, "pong": True}), flush=True)
                continue
            if cmd == "detect_named":
                names = req.get("names") or []
                image = req.get("image_b64") or req.get("image")  # P0-1：优先内存 b64
                if not image or not names:
                    print(json.dumps({"ok": False,
                                      "error": "需要 image 与 names"}), flush=True)
                    continue
                if req.get("conf") is not None:
                    loc.conf = float(req["conf"])
                els = loc.detect_named(image, names)
                print(json.dumps({
                    "ok": True,
                    "elements": _els_to_json(els, loc.centroids),
                }), flush=True)
                continue
            print(json.dumps({"ok": False, "error": f"未知 cmd: {cmd}"}), flush=True)
        except Exception as e:
            # 单条失败不杀进程，保持常驻
            print(json.dumps({"ok": False,
                              "error": f"{type(e).__name__}: {e}"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
