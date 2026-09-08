"""
sam3_locator.py —— SAM3 开放词汇「概念分割」定位器
============================================================================
为什么需要它：
    detector.py 的 OpenVocabLocator（开放词汇主通道）挂的是 YOLOE，
    但 yoloe-11s.pt 权重 404 → **这条主通道实际是残的**。
    SAM3 是同赛道的上位替代，且额外给：
      · 像素级 mask（点击点可用 mask 质心，比 bbox 中心稳）
      · 跨帧跟踪（SAM2 memory bank；SAM3.1 还能一次跟 16 个物体）
      · 400 万概念的开放词汇（SA-Co 基准）

两种运行模式（自动选择，对上层透明）：
    direct —— 当前解释器能 `import sam3`（如 .venv-sam3 里跑），同进程直接推理。
    worker —— 当前解释器装不了 sam3（如 base python，C 盘告急），
              自动 spawn **常驻子进程** `sam3_worker.py`，走 stdin/stdout JSON。
              模型只加载一次，不会每帧重启。见 sam3_worker.py。

四个已实测的坑（详见 skill `sam3-windows-offline-deploy`）：
    1. HF 上 facebook/sam3 是 gated（403）→ 权重走 ModelScope，见 sam3_fetch.py
    2. pip 版 sam3 不带数据文件 → BPE 词表由 merges.txt 现压（_ensure_bpe）
    3. import sam3 需要 triton → Windows 需 pip install triton-windows
    4. torchvision 必须 CUDA 版，否则 roi_align 报 CUDA NotImplementedError

降级承诺：sam3 不可用 / 权重缺失 / worker 起不来 → detect* 返回 []，
          **绝不抛穿**，由上层 SemanticLocator 退到 YOLOE / ScreenParser。

用法：
    loc = SAM3Locator()                        # 自动选模式，懒加载
    els = loc.detect_named(frame, ["红色按钮", "关闭图标"])
    pt  = loc.click_point(0)                   # mask 质心，推荐点击点
"""
from __future__ import annotations

import atexit
import base64
import io
import json
import os
import queue
import subprocess
import sys
import threading
import uuid
import warnings
from typing import List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # src/perception
_SRC = os.path.dirname(_HERE)                          # src
_ROOT = os.path.dirname(_SRC)                          # 项目根（blobs/weights 在这层）
for _p in (_ROOT, _SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detector import Element, TargetLocator   # noqa: E402

# 权重默认落在项目的 blobs/weights/sam3/（务必在有空间的盘）
DEFAULT_CKPT = os.path.join(_ROOT, "blobs", "weights", "sam3", "sam3.pt")
DEFAULT_BPE = os.path.join(_ROOT, "blobs", "weights", "sam3",
                           "bpe_simple_vocab_16e6.txt.gz")
WORKER_SCRIPT = os.path.join(_HERE, "sam3_worker.py")


def _ensure_bpe(weights_dir: str) -> Optional[str]:
    """拿到 BPE 词表(.txt.gz)；缺失时用同目录的 merges.txt 现压一个。

    坑：pip 装的 sam3 **不带数据文件**（RECORD 里零个 txt/gz），
    它默认的 sam3/assets/bpe_simple_vocab_16e6.txt.gz 根本不存在。
    而 SimpleTokenizer 要的是 gzip 内容 = `#version: 0.2` + 48894 条「词对」行，
    与我们已下载的 CLIP merges.txt **完全一致**（实测 48895 行，逐行对得上）。
    """
    import gzip
    gz = os.path.join(weights_dir, "bpe_simple_vocab_16e6.txt.gz")
    if os.path.isfile(gz):
        return gz
    src = os.path.join(weights_dir, "merges.txt")
    if not os.path.isfile(src):
        return None
    with open(src, "rb") as f_in, gzip.open(gz, "wb") as f_out:
        f_out.write(f_in.read())
    return gz


def _default_worker_python() -> Optional[str]:
    """找装了 sam3 的解释器：项目下的 .venv-sam3。"""
    for cand in (os.path.join(_ROOT, ".venv-sam3", "Scripts", "python.exe"),
                 os.path.join(_ROOT, ".venv-sam3", "bin", "python")):
        if os.path.isfile(cand):
            return cand
    return None


def _to_pil(frame):
    """frame 可以是路径(str) / base64 str / data-url str / numpy 数组 → 统一成 PIL.Image。

    P0-1：支持 base64（data-url 或裸 b64），worker 模式直接用内存帧，不再落盘。
    """
    from PIL import Image
    if isinstance(frame, str):
        if os.path.isfile(frame):
            return Image.open(frame).convert("RGB")
        # 尝试 base64（data:image/png;base64,xxxx 或裸 b64）
        s = frame
        if s.startswith("data:"):
            s = s.split(",", 1)[1] if "," in s else s
        try:
            raw = base64.b64decode(s, validate=False)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return Image.open(frame).convert("RGB")  # 最后兜底（当成路径）
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame if frame.dtype == np.uint8
                               else frame.astype(np.uint8))
    return frame


def _encode_b64(frame) -> str:
    """把帧编码成 base64 PNG 字符串（P0-1：worker 模式内存传帧，不落盘）。"""
    from PIL import Image
    img = _to_pil(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cleanup_tmp(tmp_dir: str) -> int:
    """清掉历史遗留的 sam3_frame_*.png（P0-1）。返回清理数量。"""
    try:
        if not os.path.isdir(tmp_dir):
            return 0
        n = 0
        for fn in os.listdir(tmp_dir):
            if fn.startswith("sam3_frame_") and fn.endswith(".png"):
                try:
                    os.remove(os.path.join(tmp_dir, fn))
                    n += 1
                except OSError:
                    pass
        return n
    except Exception:
        return 0


class SAM3Locator(TargetLocator):
    """SAM3 文本提示定位器：给一句人话，返回所有匹配实例的框 + mask 质心。

    与 YOLOE 的关键差异：
      - YOLOE: set_classes([...]) → 只出 bbox
      - SAM3 : set_text_prompt(...) → 出 bbox **+ mask**，质心可作点击点
    """

    def __init__(self, checkpoint_path: str = DEFAULT_CKPT,
                 bpe_path: Optional[str] = DEFAULT_BPE,
                 device: str = "cuda", conf: float = 0.5,
                 default_prompts: Optional[List[str]] = None,
                 verbose: bool = True,
                 worker_python: Optional[str] = None,
                 tmp_dir: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.device = device
        self.conf = conf
        self.default_prompts = default_prompts or []
        self.verbose = verbose
        self._worker_python = worker_python      # None=自动找；False=禁用 worker
        self.tmp_dir = tmp_dir or os.path.join(_ROOT, "tmp")

        self._model = None
        self._processor = None
        self._proc = None                         # worker 子进程
        self._mode: Optional[str] = None          # "direct" | "worker"
        self._load_error: Optional[str] = None
        self.centroids: List[tuple] = []          # 与最近一次结果按索引对齐

        # P0-2：worker 看门狗
        self._reader: Optional[threading.Thread] = None
        self._rq: "queue.Queue" = queue.Queue()   # reader 线程把 stdout 行塞进来
        self._rq_closed = False                   # reader 已收到 EOF
        self._rpc_timeout = 90.0                   # 单帧推理超时上限（秒）
        self._warned_once = False                  # P1-5：降级告警只响一次

    # ---------------- 状态 ----------------
    @property
    def available(self) -> bool:
        """是否已可用。**会触发懒加载**——只读 _model 会在首次访问时永远为 False，
        把真实失败原因吞掉（踩过：main 里先判 available 再 detect，报 '不可用: None'）。"""
        return self._ensure()

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def mode(self) -> Optional[str]:
        """"direct"（同进程）/ "worker"（子进程）/ None（不可用）。"""
        return self._mode

    @property
    def is_alive(self) -> bool:
        """worker 子进程是否还活着（P1-5：中途死掉上层能感知）。"""
        if self._mode != "worker" or self._proc is None:
            return self._mode == "direct"
        return self._proc.poll() is None

    def status(self) -> dict:
        """P1-5：监控指标。available 是否可用、alive 进程是否存活、mode/error。"""
        return {
            "available": self.available if self._mode else False,
            "alive": self.is_alive,
            "mode": self._mode,
            "error": self._load_error,
        }

    def _warn_once(self, msg: str):
        """P1-5：降级只告警一次，走 stderr 不污染 worker 的 stdout JSON 协议。"""
        if self._warned_once:
            return
        self._warned_once = True
        try:
            print(f"  [SAM3] {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass

    # ---------------- 加载 ----------------
    def _resolve_worker(self) -> Optional[str]:
        if self._worker_python is None:
            self._worker_python = _default_worker_python()
        return self._worker_python or None

    def _ensure(self) -> bool:
        if self._mode:
            return True
        if self._load_error and self._worker_python is False:
            return False

        # 1) 同进程能 import sam3 → 直接模式
        try:
            import sam3  # noqa: F401
            return self._load_direct()
        except Exception as e:
            direct_err = f"{type(e).__name__}: {e}"

        # 2) 否则尝试常驻子进程
        if self._worker_python is False:
            self._load_error = f"同进程加载失败（{direct_err}），且已禁用 worker"
            return False
        if self._start_worker():
            return True

        if not self._load_error:
            self._load_error = (f"同进程加载失败（{direct_err}）；"
                                f"worker 也不可用（{self._resolve_worker() or '未找到 .venv-sam3'}）")
        self._warn_once(f"不可用（降级到 YOLOE/闭集）: {self._load_error}")
        return False

    def _load_direct(self) -> bool:
        try:
            if not os.path.isfile(self.checkpoint_path):
                self._load_error = f"权重不存在: {self.checkpoint_path}"
                if self.verbose:
                    print(f"  [SAM3] 不可用：{self._load_error}")
                return False

            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            dev = self.device
            if dev == "cuda" and not torch.cuda.is_available():
                dev = "cpu"
                if self.verbose:
                    print("  [SAM3] CUDA 不可用 → 回落 CPU（会慢）")

            bpe = self.bpe_path
            if bpe is None or not os.path.isfile(bpe):
                bpe = _ensure_bpe(os.path.dirname(self.checkpoint_path))
            if bpe is None:
                self._load_error = ("缺 BPE 词表：目录下既无 bpe_simple_vocab_16e6.txt.gz "
                                    f"也无 merges.txt @ {os.path.dirname(self.checkpoint_path)}")
                if self.verbose:
                    print(f"  [SAM3] 不可用：{self._load_error}")
                return False

            model = build_sam3_image_model(
                bpe_path=bpe,
                device=dev,
                eval_mode=True,
                checkpoint_path=self.checkpoint_path,
                load_from_HF=False,          # 关键：HF 是 gated，别去撞门
                enable_segmentation=True,
            )
            self._model = model
            self._processor = Sam3Processor(model)
            try:
                self._processor.set_confidence_threshold(self.conf)
            except Exception:
                pass
            self._mode = "direct"
            if self.verbose:
                print(f"  [SAM3] 已加载(direct) {os.path.basename(self.checkpoint_path)} @ {dev}")
            return True
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            if self.verbose:
                print(f"  [SAM3] 同进程加载失败: {self._load_error}")
            return False

    def _start_worker(self) -> bool:
        py = self._resolve_worker()
        if not py or not os.path.isfile(WORKER_SCRIPT):
            return False
        try:
            if self.verbose:
                print(f"  [SAM3] 启动常驻 worker（首次需加载模型，约 1~2 分钟）...",
                      flush=True)
            # 清掉历史遗留的落盘帧（P0-1）
            _cleanup_tmp(self.tmp_dir)
            proc = subprocess.Popen(
                [py, WORKER_SCRIPT],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                bufsize=1, cwd=_ROOT,
            )
            first = proc.stdout.readline()        # 主线程收 ready 握手，避免被 reader 抢
            if not first:
                proc.kill()
                return False
            resp = json.loads(first)
            if not resp.get("ok"):
                proc.kill()
                self._load_error = f"worker 启动失败: {resp.get('error')}"
                return False
            self._proc = proc
            self._mode = "worker"
            self._rq = queue.Queue()
            self._rq_closed = False
            self._reader = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader.start()
            atexit.register(self.close)
            if self.verbose:
                print("  [SAM3] worker 就绪(worker 模式)")
            return True
        except Exception as e:
            self._load_error = f"worker 启动异常: {type(e).__name__}: {e}"
            return False

    def _reader_loop(self):
        """P0-2：后台线程持续把 worker stdout 行塞进队列，EOF 时塞 None 作死信号。"""
        try:
            for line in self._proc.stdout:
                self._rq.put(line)
        except Exception:
            pass
        finally:
            self._rq_closed = True
            try:
                self._rq.put(None)
            except Exception:
                pass

    def _kill_worker(self):
        """P0-2：强制杀掉卡死/已死的 worker，重置状态以便下次 rpc 重启。"""
        self._mode = None
        self._reader = None
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    def close(self):
        """结束常驻 worker（atexit 会自动调）。"""
        proc = self._proc
        self._reader = None
        self._mode = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=5)
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        self._proc = None

    # ---------------- 定位 ----------------
    def detect_named(self, frame, names: List[str]) -> List[Element]:
        """按语义名（文本提示）找所有匹配实例。names 里每个词独立提示。"""
        if not self._ensure():
            return []
        if self._mode == "worker":
            return self._worker_detect(frame, names)
        return self._direct_detect(frame, names)

    def _direct_detect(self, frame, names: List[str]) -> List[Element]:
        out: List[Element] = []
        cents: List[tuple] = []
        try:
            image = _to_pil(frame)
            state = self._processor.set_image(image)
            for name in names:
                res = self._processor.set_text_prompt(state=state, prompt=name)
                boxes, scores, masks = self._unpack(res)
                if boxes is None:
                    continue
                for i, (b, s) in enumerate(zip(boxes, scores)):
                    x1, y1, x2, y2 = [int(v) for v in b]
                    out.append(Element(id=str(uuid.uuid4())[:8], label=name,
                                       bbox=(x1, y1, x2, y2), conf=float(s)))
                    cents.append(self._mask_centroid(masks, i) if masks is not None
                                 else ((x1 + x2) // 2, (y1 + y2) // 2))
        except Exception as e:
            if self.verbose:
                print(f"  [SAM3] 推理失败（降级）: {type(e).__name__}: {e}")
            return []
        self.centroids = cents
        return out

    def _worker_rpc(self, req: dict, retry: bool = True) -> Optional[dict]:
        """P0-2：带超时 + 看门狗的 RPC。读行交给 reader 线程，主线程 queue.get 超时。
        超时/进程死/EOL → 杀掉 worker 重启一次再试，避免永久阻塞冻住整个感知。"""
        if self._proc is None or self._proc.poll() is not None:
            if not (retry and self._start_worker()):
                return None
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            try:
                line = self._rq.get(timeout=self._rpc_timeout)
            except queue.Empty:
                # 超时：worker 卡死（GPU OOM/异常）→ 杀掉，重启再试一次
                self._warn_once(f"worker 推理超时（>{self._rpc_timeout}s）已重启")
                self._kill_worker()
                if retry and self._start_worker():
                    return self._worker_rpc(req, retry=False)
                return None
            if line is None:                  # reader 报 EOF：进程死了
                self._kill_worker()
                if retry and self._start_worker():
                    return self._worker_rpc(req, retry=False)
                return None
            return json.loads(line)
        except Exception:
            self._kill_worker()
            if retry and self._start_worker():
                return self._worker_rpc(req, retry=False)
            return None

    def _worker_detect(self, frame, names: List[str]) -> List[Element]:
        try:
            img_b64 = _encode_b64(frame)      # P0-1：内存帧，不落盘
            resp = self._worker_rpc({"cmd": "detect_named", "image_b64": img_b64,
                                     "names": names, "conf": self.conf})
            if not resp or not resp.get("ok"):
                if self.verbose and resp:
                    print(f"  [SAM3] worker 返回错误: {resp.get('error')}")
                return []
            els, cents = [], []
            for d in resp.get("elements", []):
                els.append(Element(id=d.get("id") or str(uuid.uuid4())[:8],
                                   label=d.get("label", ""),
                                   bbox=tuple(d.get("bbox") or (0, 0, 0, 0)),
                                   conf=float(d.get("conf", 0.0))))
                c = d.get("centroid")
                cents.append(tuple(c) if c else None)
            self.centroids = cents
            return els
        except Exception as e:
            if self.verbose:
                print(f"  [SAM3] worker 调用失败（降级）: {type(e).__name__}: {e}")
            return []

    def detect(self, frame) -> List[Element]:
        """免提示路径：SAM3 必须给提示，故用 default_prompts；没配就返回空。"""
        if not self.default_prompts:
            return []
        return self.detect_named(frame, self.default_prompts)

    def click_point(self, idx: int) -> Optional[tuple]:
        """第 idx 个结果的推荐点击点 = mask 质心（比 bbox 中心稳）。"""
        if 0 <= idx < len(self.centroids):
            return self.centroids[idx]
        return None

    # ---------------- 内部：结果解析 ----------------
    @staticmethod
    def _unpack(res):
        """把 SAM3 输出解析成 (boxes, scores, masks)，兼容 dict/对象多种形态。
        实测输出 dict keys = [original_height, original_width, backbone_out,
        geometric_prompt, masks_logits, masks, boxes, scores]。"""
        boxes = scores = masks = None
        d = res if isinstance(res, dict) else getattr(res, "__dict__", {})
        get = (lambda k: d.get(k)) if isinstance(d, dict) else (lambda k: getattr(res, k, None))

        for k in ("boxes", "bbox", "pred_boxes"):
            if get(k) is not None:
                boxes = get(k)
                break
        for k in ("scores", "score", "pred_scores", "logits"):
            if get(k) is not None:
                scores = get(k)
                break
        for k in ("masks", "pred_masks", "out_masks"):
            if get(k) is not None:
                masks = get(k)
                break
        return _to_np(boxes), _to_np(scores), masks

    @staticmethod
    def _mask_centroid(masks, i):
        """mask 质心（像素坐标）。

        P1-4：优先用 scipy 的 distance transform —— 取**最大连通分量**内离边界最远的
        点，该点必然落在 mask **内部**，修掉凹形/环形目标用非零像素均值时质心被拉到
        体外/洞外的问题。scipy 不可用或失败时回落到像素均值。
        """
        try:
            m = masks[i]
            if hasattr(m, "detach"):
                m = m.detach().cpu().numpy()
            m = np.asarray(m)
            while m.ndim > 2:
                m = m[0]
            m = m > 0.5
            if m.sum() == 0:
                return None
            try:
                from scipy import ndimage as ndi
                labeled, n = ndi.label(m)
                if n >= 1:
                    sizes = ndi.sum(np.ones_like(labeled), labeled,
                                   index=range(1, n + 1))
                    biggest = int(np.argmax(sizes)) + 1
                    comp = (labeled == biggest)
                    dt = ndi.distance_transform_edt(comp)
                    # 取 dt 最大的单个像素（必在 mask 内）：并列最大点多时取 mean 会把
                    # 坐标拉到凹形包围的空区中心而落到体外，故用 argmax 取第一个。
                    flat = int(np.argmax(dt))
                    y0, x0 = np.unravel_index(flat, dt.shape)
                    return (int(x0), int(y0))
            except Exception:
                pass
            ys, xs = np.nonzero(m)
            return (int(xs.mean()), int(ys.mean()))
        except Exception:
            return None


def _to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 冒烟：对一张真实截图按文本提示定位。用法：
    #   python src/perception/sam3_locator.py <图片> "提示1" "提示2"
    args = sys.argv[1:]
    img = args[0] if args else os.path.join(_ROOT, "wechat_full.png")
    prompts = args[1:] or ["icon", "button", "text"]
    print(f"图片: {img}\n提示: {prompts}\n")

    loc = SAM3Locator()
    if not loc.available:
        print("SAM3 不可用（降级）:", loc.load_error)
        sys.exit(2)

    els = loc.detect_named(img, prompts)
    print(f"\n模式={loc.mode}  命中 {len(els)} 个：")
    for i, e in enumerate(els):
        cp = loc.click_point(i)
        print(f"  {e.to_context(i + 1)}  conf={e.conf:.2f}  质心={cp}")
    loc.close()
    print("\nSMOKE_OK" if els else "\n无命中（提示词可能不在画面里）")
