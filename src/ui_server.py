"""
ui_server.py —— MGA 控制台后端（零第三方依赖，纯标准库）
=========================================================
给你一个能「一键操作 + 看数据 + 看报错」的本地面板：
  · 启动/停止 采集 / 训练 / Demo 三个后台任务
  · 实时日志流（子进程 stdout+stderr 合并，增量拉取）
  · 训练数据概览：条数、响应多样性、goal 分布、最近样本
  · /blobs/ 静态文件服务（可直接在页面里看截图和 Demo 结果图）

为什么不用 Flask/FastAPI：环境里没装。用 http.server 标准库，
零安装、零依赖风险，一条命令就能起。

启动：
  PYTHONPATH=src python -m ui_server            # 默认 http://127.0.0.1:7860 并自动开浏览器
  PYTHONPATH=src python -m ui_server --port 8000 --no-browser

重要：这个服务由**你自己的终端**持有，不受沙箱后台任务回收影响。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
BLOBS = os.path.join(ROOT, "blobs")
DATA = os.path.join(BLOBS, "trajectories.jsonl")
HTML = os.path.join(SRC, "ui", "index.html")

KIND_LABEL = {"collect": "采集", "train": "训练", "demo": "Demo"}

STATE = {"proc": None, "kind": None, "pid": None, "t0": 0.0, "params": {}}
LINES: list = []
LOCK = threading.Lock()
MAX_LINES = 6000

ERR_PAT = ("traceback", "error", "错误", "失败", "exception", "!!")


# ---------------------------------------------------------------------------
# 日志缓冲
# ---------------------------------------------------------------------------
def push(text: str):
    with LOCK:
        LINES.append(text)
        if len(LINES) > MAX_LINES:
            del LINES[:len(LINES) - MAX_LINES]


def err_count() -> int:
    with LOCK:
        return sum(1 for l in LINES if any(p in l.lower() for p in ERR_PAT))


# ---------------------------------------------------------------------------
# 子进程管理
# ---------------------------------------------------------------------------
def _build_cmd(kind: str, p: dict) -> list:
    """构造命令。**用 list 且 shell=False**——参数只作 argv，不存在命令注入。"""
    if kind == "collect":
        cmd = [sys.executable, "-m", "main", "--real",
               "--frames", "100000",
               "--collect-every", str(int(p.get("collect_every", 1))),
               "--llm", "oracle",
               "--max-elements", "32",
               "--auto-goal",
               "--minutes", str(float(p.get("minutes", 31)))]
        cmd += ["--ocr-scale", str(float(p.get("ocr_scale", 0.5))),
                "--interval", str(float(p.get("interval", 2)))]
        return cmd
    if kind == "train":
        return [sys.executable, "-m", "training.train_minimind",
                "--max-len", str(int(p.get("max_len", 1728))),
                "--epochs", str(int(p.get("epochs", 3))),
                "--batch", str(int(p.get("batch", 2))),
                "--lr", "2e-4"]
    if kind == "demo":
        cmd = [sys.executable, "-m", "demo_locator"]
        g = str(p.get("goal", "")).strip()
        if g:
            cmd += ["--goal", g]
        return cmd
    raise ValueError(f"未知任务类型: {kind}")


def _pump(proc):
    """把子进程输出实时抽进日志缓冲。"""
    try:
        for raw in iter(proc.stdout.readline, b""):
            push(raw.decode("utf-8", "replace").rstrip("\r\n"))
    except Exception as ex:
        push(f"[sys] 日志读取异常: {type(ex).__name__}: {ex}")
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        code = proc.wait()
        push(f"[sys] === 任务结束 exit={code} ===")
        STATE.update(proc=None, kind=None, pid=None)


def start_task(kind: str, params: dict):
    if STATE.get("proc") and STATE["proc"].poll() is None:
        return False, "已有任务在跑，请先停止"
    try:
        cmd = _build_cmd(kind, params)
    except Exception as ex:
        return False, str(ex)

    env = os.environ.copy()
    env["PYTHONPATH"] = SRC
    env["PYTHONIOENCODING"] = "utf-8"     # Windows 中文输出默认是 gbk，会乱码

    push(f"[sys] === 启动 {KIND_LABEL.get(kind, kind)} ===")
    push(f"[sys] $ {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                bufsize=1)
    except Exception as ex:
        push(f"[sys] 启动失败: {type(ex).__name__}: {ex}")
        return False, str(ex)

    STATE.update(proc=proc, kind=kind, pid=proc.pid,
                 t0=time.time(), params=params)
    threading.Thread(target=_pump, args=(proc,), daemon=True).start()
    return True, KIND_LABEL.get(kind, kind)


def stop_task():
    proc, pid = STATE.get("proc"), STATE.get("pid")
    if not proc:
        return False, "没有运行中的任务"
    # 采集会拉起 OCR/YOLO 子进程，必须杀整棵树，否则孤儿进程吃满 CPU
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, timeout=15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    push("[sys] === 已请求停止 ===")
    return True, "已停止"


# ---------------------------------------------------------------------------
# 数据概览（按 mtime 缓存，避免每次轮询都解析 5MB jsonl）
# ---------------------------------------------------------------------------
_CACHE = {"mtime": -1.0, "data": None}


def data_stats() -> dict:
    try:
        mt = os.path.getmtime(DATA)
    except OSError:
        return {"n": 0, "mtime": 0}
    if mt == _CACHE["mtime"] and _CACHE["data"] is not None:
        return _CACHE["data"]

    rows = []
    try:
        with open(DATA, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception as ex:
        return {"n": 0, "mtime": mt, "error": str(ex)}

    # 只读一遍：条数、去重后的唯一 prompt/response、goal 分布一次算完
    n = len(rows)
    uniq_r = len({str(r.get("response", "")) for r in rows})
    uniq_p = len({str(r.get("prompt", "")) for r in rows})
    goals = Counter(str(r.get("goal", "")) for r in rows if r.get("goal"))
    recent = []
    for i, row in enumerate(rows[-8:][::-1], 1):
        c = row.get("action_coords") or [0, 0]
        recent.append({"i": i,
                       "goal": str(row.get("goal", ""))[:24],
                       "coords": f"{c[0]},{c[1]}" if len(c) == 2 else "-",
                       "resp": str(row.get("response", ""))[:70]})
    data = {
        "n": n,
        "unique_prompts": uniq_p,
        "unique_responses": uniq_r,
        "unique_goals": len(goals),
        "goal_top": goals.most_common(15),
        "recent": recent,
        "mtime": mt,
    }
    _CACHE.update(mtime=mt, data=data)
    return data


def latest_demo_img():
    """最新一张 demo 结果图（demo_locator 产物），供页面直接展示。"""
    try:
        cands = [f for f in os.listdir(BLOBS)
                 if f.startswith("demo_") and f.endswith(".png")]
        if not cands:
            return None
        cands.sort(key=lambda f: os.path.getmtime(os.path.join(BLOBS, f)),
                   reverse=True)
        return "/blobs/" + cands[0]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):      # 静默访问日志，免得刷屏
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # ---------------- GET ----------------
    def do_GET(self):
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)

        if path in ("/", "/index.html"):
            try:
                with open(HTML, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                self._send(404, b"index.html not found", "text/plain")
            return

        if path == "/api/status":
            running = bool(STATE.get("proc") and STATE["proc"].poll() is None)
            el = int(time.time() - STATE["t0"]) if running else 0
            kind = STATE.get("kind")
            self._json({
                "running": running,
                "kind": kind,
                "kindLabel": KIND_LABEL.get(kind, kind or ""),
                "elapsed": f"{el//60}分{el%60:02d}秒" if running else "",
                "errors": err_count(),
                "stats": data_stats(),
                "latest_demo": latest_demo_img(),
            })
            return

        if path == "/api/logs":
            try:
                since = int(qs.get("since", ["0"])[0])
            except Exception:
                since = 0
            with LOCK:
                total = len(LINES)
                chunk = LINES[since:] if since < total else []
            self._json({"lines": chunk, "total": total})
            return

        if path.startswith("/blobs/"):
            self._serve_file(os.path.join(BLOBS, path[len("/blobs/"):]))
            return

        self._json({"error": "not found"}, 404)

    def _serve_file(self, rel: str):
        # 防目录穿越：规范化后必须仍在 BLOBS 内
        target = os.path.normpath(os.path.join(BLOBS, rel))
        if not target.startswith(os.path.normpath(BLOBS)) or not os.path.isfile(target):
            self._json({"error": "not found"}, 404)
            return
        ext = os.path.splitext(target)[1].lower()
        ctype = {".png": "image/png", ".jpg": "image/jpeg",
                 ".jpeg": "image/jpeg", ".json": "application/json",
                 ".jsonl": "text/plain; charset=utf-8"}.get(ext, "application/octet-stream")
        try:
            with open(target, "rb") as f:
                self._send(200, f.read(), ctype)
        except OSError:
            self._json({"error": "read failed"}, 404)

    # ---------------- POST ----------------
    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = {}

        if u.path == "/api/start":
            kind = str(body.get("kind", ""))
            if kind not in KIND_LABEL:
                self._json({"ok": False, "msg": f"未知任务 {kind}"})
                return
            ok, msg = start_task(kind, body.get("params") or {})
            self._json({"ok": ok, "msg": msg, "kindLabel": msg})
            return

        if u.path == "/api/stop":
            ok, msg = stop_task()
            self._json({"ok": ok, "msg": msg})
            return

        if u.path == "/api/clear":
            with LOCK:
                LINES.clear()
            self._json({"ok": True})
            return

        self._json({"ok": False, "msg": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    port = args.port
    for attempt in range(20):          # 端口被占就顺延
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    else:
        print("!! 找不到可用端口")
        return 1

    url = f"http://127.0.0.1:{port}"
    print(f"MGA 控制台已启动: {url}")
    print("（Ctrl+C 退出）")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
