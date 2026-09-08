"""
vision_locate.py — 让大模型"看屏幕"（Vision Computer Use 的感知层）

为什么要有这个模块
--------------------
纯颜色/形状的 CV 在"游戏区定位"这件事上一直翻车：
  - 新版 chrome://dino 背景是纯白（不再是浅灰 canvas）→ 颜色锚失效
  - 用户窗口化、不全屏 → 地平线不横跨全屏 → 全宽线检测失效
  - 桌面任务栏/窗口边也是深灰横线 → 误检成游戏
大模型看一眼就知道"哪个窗口是小恐龙、游戏区在哪"，这正是它擅长的
**低频语义理解**。

但大模型有个物理天花板：**一次调用 1.8~2.6 秒**（实测 hy-vision-2.0-instruct）。
小恐龙从障碍出现到撞上只有 1.5~2 秒，所以**不能让大模型逐帧决策**。

正确分工（System1 / System2）：
  - System2（本模块，大模型）：低频、慢思考 → 定位游戏区 / 判断状态 / 异常复盘
                                 跑一次约 2 秒，成功即存盘复用
  - System1（CV + MotionModel）：高频、快反应 → 每帧预判跳跃（微秒级）

零第三方依赖：只用标准库 urllib + base64 + json（PIL 仅用于缩放）。
密钥只从 .env.local / 环境变量读，绝不打印、绝不入库。
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# 复用 llm_bridge 里的密钥/config 读取（同一套约定，不重复实现）
try:
    from perception.llm_bridge import _load_dotenv, _cfg_runtime
except Exception:                                   # 独立 import 兜底
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from perception.llm_bridge import _load_dotenv, _cfg_runtime
    except Exception:
        def _load_dotenv(path=None):
            pass

        def _cfg_runtime():
            return {}


DEFAULT_VISION_MODEL = "hy-vision-2.0-instruct"

LOCATE_PROMPT = """你是一个屏幕分析助手。这张图是用户电脑屏幕的截图。

任务：找出 Chrome 小恐龙游戏（chrome://dino）的位置。它长这样：
一条**横向的灰色地面线**，线**左侧站着一只像素小恐龙**（深灰色剪影，约 40-50px 高），
恐龙右侧的跑道上会出现仙人掌或飞鸟。背景通常是白色或浅灰。

只输出一行 JSON，不要解释、不要 markdown 代码块：
{"is_game": true, "ground_y": <地面线所在的y像素>, "dino_x": <恐龙中心x像素>, "bbox": [x, y, width, height], "note": "一句话说明"}

要求：
- ground_y 和 dino_x 是最关键的两个锚点，请尽量给准（整数像素）
- bbox 用来圈住游戏区，**紧贴游戏画面，不要把周围大片桌面空白框进来**
- 如果图里没有小恐龙游戏：{"is_game": false, "ground_y": null, "dino_x": null, "bbox": null, "note": "你实际看到的内容"}
"""

PERCEIVE_PROMPT = """这是 Chrome 小恐龙游戏的截图。

只输出一行 JSON，不要解释、不要 markdown 代码块：
{"dino_x": <恐龙中心x>, "obstacles": [{"x": <障碍左边缘x>, "kind": "cactus|bird"}], "action": "jump|duck|none"}

要求：
- 只列出恐龙**右侧**的可见障碍，按 x 从小到大
- kind: 地面上的分叉植物是 cactus，空中会飞的是 bird
- action: 最近的障碍很近(约 150px 内)就 jump；空中的鸟很近就 duck；否则 none
- 所有坐标是这张截图内的整数像素
"""


class VisionLocator:
    """Vision LLM 看屏：定位游戏区 / 理解画面。

    一次调用约 2 秒 —— 设计上只用于低频（启动定位、异常复盘），
    绝不进每帧循环。
    """

    def __init__(self,
                 model: str = None,
                 base_url: str = None,
                 api_key: str = None,
                 timeout: float = 60.0,
                 max_side: int = 1280):
        _load_dotenv()
        cfg = _cfg_runtime()
        self.model = model or os.getenv("MGA_VISION_MODEL") or DEFAULT_VISION_MODEL
        self.base_url = (base_url or os.getenv("MGA_LLM_BASE_URL")
                         or cfg.get("llm_base_url") or "").rstrip("/")
        # 密钥：环境变量优先（.env.local 已注入），绝不使用硬编码
        self.api_key = api_key or os.getenv("MGA_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.timeout = timeout
        self.max_side = max_side
        self.last_latency = 0.0
        self.last_raw = ""

    # ---------------- 底层调用 ----------------
    def _chat_json(self, prompt: str, img) -> Tuple[Optional[dict], float, str]:
        """发一张图 + 提示词，返回 (解析后的dict|None, 耗时秒, 原始文本)。"""
        if not _HAS_PIL:
            raise RuntimeError("需要 pillow：pip install pillow")
        if not self.api_key:
            raise RuntimeError("未配置密钥：请在 .env.local 填 MGA_LLM_API_KEY=你的Key")
        if not self.base_url:
            raise RuntimeError("未配置 llm_base_url（config.json runtime.llm_base_url）")

        img, scale = self._resize(img)
        b64 = self._to_b64(img)
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "max_tokens": 400,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:200]
            except Exception:
                pass
            raise RuntimeError(f"Vision API HTTP {e.code}: {body}")
        except Exception as e:
            raise RuntimeError(f"Vision API 调用失败: {e}")
        dt = time.perf_counter() - t0
        self.last_latency = dt

        try:
            raw = data["choices"][0]["message"]["content"] or ""
        except Exception:
            raw = ""
        self.last_raw = raw
        return _parse_json(raw), dt, raw

    # ---------------- 高层能力 ----------------
    def locate(self, screen) -> Optional[Dict[str, Any]]:
        """在大屏/全屏图里找小恐龙游戏区。

        返回 dict: {"region": (x,y,w,h), "ground_y": int|None, "dino_x": int|None,
                    "note": str, "latency": float}
        找不到（或模型说不是游戏）返回 None。

        注意：视觉模型的**像素级定位不准**（实测 bbox 误差可达数百 px），
        所以它只负责"粗定位"，精确边界交给 CV 形状佐证去修（见 refine）。
        """
        img, scale = self._resize(screen)
        res, dt, raw = self._chat_json(LOCATE_PROMPT, screen)
        if not res or not res.get("is_game"):
            return None

        def _i(v):
            try:
                return int(float(v))
            except Exception:
                return None

        gy, dx = _i(res.get("ground_y")), _i(res.get("dino_x"))
        # 锚点还原到原图尺度
        if gy is not None:
            gy = int(round(gy / scale))
        if dx is not None:
            dx = int(round(dx / scale))

        bb = res.get("bbox")
        region = None
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            try:
                x, y, w, h = [int(float(v)) for v in bb]
                if w > 0 and h > 0:
                    region = tuple(int(round(v / scale)) for v in (x, y, w, h))
                    region = (max(0, region[0]), max(0, region[1]),
                              max(1, region[2]), max(1, region[3]))
            except Exception:
                region = None
        return {"region": region, "ground_y": gy, "dino_x": dx,
                "note": str(res.get("note", ""))[:120],
                "latency": dt, "raw": raw[:200]}

    def perceive(self, frame) -> Dict[str, Any]:
        """理解一帧游戏画面：恐龙位置 / 障碍 / 建议动作。

        ⚠️ 每帧调用会慢到无法游玩（~2s/次）——仅用于低频校验或演示。
        """
        res, dt, raw = self._chat_json(PERCEIVE_PROMPT, frame)
        out = {"dino_x": None, "obstacles": [], "action": "none",
               "latency": dt, "raw": raw, "ok": res is not None}
        if not res:
            return out
        try:
            out["dino_x"] = float(res.get("dino_x")) if res.get("dino_x") is not None else None
        except Exception:
            pass
        obs = res.get("obstacles") or []
        if isinstance(obs, list):
            for o in obs:
                if isinstance(o, dict) and o.get("x") is not None:
                    try:
                        out["obstacles"].append({"x": float(o["x"]),
                                                 "kind": str(o.get("kind", "cactus"))})
                    except Exception:
                        continue
        act = str(res.get("action", "none")).lower()
        out["action"] = act if act in ("jump", "duck", "none") else "none"
        return out

    # ---------------- 工具 ----------------
    def _resize(self, img):
        """缩到 max_side 以内（省 token、提速），返回 (新图, 缩放比)。"""
        if not _HAS_PIL:
            return img, 1.0
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img) if hasattr(img, "shape") else img
        w, h = img.size
        m = max(w, h)
        if m <= self.max_side:
            return img.convert("RGB"), 1.0
        scale = self.max_side / float(m)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return img.convert("RGB").resize((nw, nh), Image.LANCZOS), scale

    def _to_b64(self, img) -> str:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_json(text: str) -> Optional[dict]:
    """从可能带 ```json 围栏 / 前后废话的文本里抠出第一个 JSON 对象。"""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip()
    s = re.sub(r"```$", "", s).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    frag = s[start:end + 1]
    try:
        return json.loads(frag)
    except Exception:
        # 容忍尾随逗号等小瑕疵
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", frag))
        except Exception:
            return None


def locate_game_region(screen, model: str = None, verbose: bool = True
                       ) -> Optional[Dict[str, Any]]:
    """便捷入口：全屏图 → 大模型粗定位结果 dict（含 region/ground_y/dino_x）。

    失败返回 None。返回的 region 是**粗框**，建议再用 CV 形状佐证精修。
    """
    try:
        vl = VisionLocator(model=model)
        if verbose:
            print(f"  [Vision] 让大模型看一眼屏幕定位游戏区（{vl.model}，约 1-3 秒）...")
        info = vl.locate(screen)
        if verbose:
            if info:
                print(f"  [Vision] 找到了：region={info['region']} "
                      f"ground_y={info['ground_y']} dino_x={info['dino_x']} "
                      f"（耗时 {vl.last_latency:.2f}s）")
                if info.get("note"):
                    print(f"  [Vision] 模型说：{info['note']}")
            else:
                print(f"  [Vision] 大模型说这不是小恐龙游戏 "
                      f"(耗时 {vl.last_latency:.2f}s)")
                if vl.last_raw:
                    print(f"  [Vision] 原文: {vl.last_raw[:120]}")
        return info
    except Exception as e:
        if verbose:
            print(f"  [Vision] 定位失败：{e}")
        return None
