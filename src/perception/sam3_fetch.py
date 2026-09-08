"""
sam3_fetch.py —— 从 ModelScope 拉取 SAM3 权重（带断点续传）

为什么不用 HF：
    facebook/sam3 在 HF 是 **gated(manual)** 模型，hf-mirror 实测返回
        403 "Access to model facebook/sam3 is restricted and you are not in the authorized list"
    需要 HF 账号手动申请 + token，而本机 huggingface.co 不可达（见 weights.py）。
    ModelScope 上 facebook/sam3 是 **ApprovalMode=0（免授权）**，且实测支持 Range(206) 断点续传。

体积提示：
    仓库共 ~6.9 GB，但 sam3.pt(3.45G) 与 model.safetensors(3.44G) 是同一权重的两种格式，
    **只需下一种**。本脚本默认只拉 sam3.pt + 全部小文件 → 实际约 3.4 GB。

用法：
    python sam3_fetch.py                      # 拉默认清单到 blobs/weights/sam3
    python sam3_fetch.py --only-small         # 只拉小文件（快速验证连通性）
    python sam3_fetch.py --dest D:/xxx/sam3   # 指定落盘目录（务必放 D 盘）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

BASE = "https://www.modelscope.cn/models/facebook/sam3/resolve/master"

# sam3.pt = 官方 sam3 包(build_sam3_image_model) 用的权重；
# model.safetensors = transformers 路径用的，二选一，默认不下（省 3.4G）。
BIG_FILES = ["sam3.pt"]
SMALL_FILES = [
    "config.json",
    "configuration.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "README.md",
]

_UA = {"User-Agent": "Mozilla/5.0"}


def _head_size(url: str) -> int:
    """拿远端文件大小（用于校验与进度）。失败返回 -1。"""
    try:
        req = urllib.request.Request(url, headers=_UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or -1)
    except Exception:
        return -1


def fetch(name: str, dest_dir: str, retries: int = 3) -> bool:
    """下载单个文件，支持断点续传（Range）。已存在且大小一致则跳过。"""
    os.makedirs(dest_dir, exist_ok=True)
    url = f"{BASE}/{name}"
    final = os.path.join(dest_dir, name)
    part = final + ".part"

    # 已完整存在：校验大小后跳过
    if os.path.isfile(final):
        remote = _head_size(url)
        local = os.path.getsize(final)
        if remote > 0 and local == remote:
            print(f"  SKIP   {name} ({local/1e6:.1f} MB, already complete)")
            return True

    for attempt in range(1, retries + 1):
        try:
            done = os.path.getsize(part) if os.path.isfile(part) else 0
            headers = dict(_UA)
            if done > 0:
                headers["Range"] = f"bytes={done}-"      # 续传
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as r:
                if done > 0 and r.status != 206:          # 服务端不支持续传→重来
                    done = 0
                total = int(r.headers.get("Content-Length") or 0)
                if r.status == 206:
                    cr = r.headers.get("Content-Range") or ""
                    if "/" in cr:
                        total = int(cr.split("/")[-1])
                mode = "ab" if done > 0 else "wb"
                t0, last = time.time(), 0.0
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)           # 1MB
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last > 5.0:               # 每 5s 打一次进度
                            last = now
                            pct = (done / total * 100) if total else 0
                            sp = done / 1e6 / max(now - t0, 1e-6)
                            print(f"    ... {name} {done/1e6:8.1f}/{total/1e6:.1f} MB "
                                  f"({pct:5.1f}%) {sp:.1f} MB/s", flush=True)
            os.replace(part, final)
            print(f"  OK     {name} ({os.path.getsize(final)/1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"  RETRY  {name} attempt {attempt}/{retries}: {type(e).__name__}: {e}")
            time.sleep(3 * attempt)
    print(f"  FAIL   {name}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join("blobs", "weights", "sam3"),
                    help="落盘目录（默认 blobs/weights/sam3，务必在有空间的盘）")
    ap.add_argument("--only-small", action="store_true", help="只拉小文件")
    ap.add_argument("--with-safetensors", action="store_true",
                    help="额外拉 model.safetensors（+3.4G，transformers 路径用）")
    args = ap.parse_args()

    files = list(SMALL_FILES)
    if not args.only_small:
        files += BIG_FILES
        if args.with_safetensors:
            files.append("model.safetensors")

    dest = os.path.abspath(args.dest)
    print(f"SAM3 权重拉取 → {dest}")
    print(f"源：ModelScope (HF 为 gated，403)\n")

    ok = 0
    for name in files:
        if fetch(name, dest):
            ok += 1
    print(f"\n完成 {ok}/{len(files)} 个文件 → {dest}")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
