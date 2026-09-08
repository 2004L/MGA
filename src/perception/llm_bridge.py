"""
llm_bridge.py —— MGA M3「YOLO 说话 → LLM 听懂 → 直接动手」的翻译桥

对应你贴的三步走：
  步骤1 结构化提取 : Element 已是 YOLO 的标准化输出（见 detector.py）
  步骤2 转自然语言   : format_scene() 把元素列表拼成 LLM 易消化的文本
  步骤3 传入 LLM     : decide_action() 调用 LLM，并强制 JSON 结构化输出

本文件补上你贴文里「进阶策略」的工程实现：
  - 强制 JSON 输出（Structured Outputs）：LLM 回 {"action","target","coordinates"}
    程序直接解析执行，不再靠正则抠散文。
  - 多模态入口：Ultralytics 官方 LLM 接口 llm(prompt, image=...) 一步到位；
    备用：标准 API（OpenAI SDK 风格）发送。
  - 多轮对话（Multi-shot）：树莓派等弱算力下，小模型分多轮逐步达成目标。

设计原则（与系统一致）：
  - LLM 是「听懂 + 决策」端，只接结构化上下文，不碰原始像素理解。
  - 所有外部依赖（ultralytics / openai）懒加载，缺失即降级，零硬依赖可自检。
  - JSON 解析健壮：兼容 markdown 代码块、前后废话、字段缺失自动回退。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# 1. 可执行动作：LLM 的「听懂」最终产物（直接喂 Computer Use）
# ---------------------------------------------------------------------------
@dataclass
class Action:
    action: str = "none"            # click / type / scroll / wait / none
    target: Optional[str] = None    # 语义目标："确认按钮" / "用户名输入框"
    coordinates: Optional[tuple] = None   # (x, y) 像素坐标（无则靠 target 解析）
    text: Optional[str] = None      # type 动作要输入的内容
    raw: dict = field(default_factory=dict)  # LLM 原始 JSON，便于调试

    def executable(self) -> bool:
        """Computer Use 能否直接执行：有明确动作 + 坐标或可追溯目标。"""
        return self.action != "none" and (self.coordinates is not None or self.target)


# ---------------------------------------------------------------------------
# 2. 步骤2：结构化场景 → 自然语言（Prompt Chaining 关键一步）
# ---------------------------------------------------------------------------
# 屏幕上百个元素全塞进 prompt ≈ 7500 token，远超小模型上下文（MiniMind2-Small
# 训练窗口取 1024）。截断必须发生在**本函数**而不是训练脚本：
# 训练侧截断会让落盘的 prompt 与推理期 prompt 不一致，直接毁掉微调（见 build_prompt 契约）。
# 实测 MiniMind tokenizer 下约 27 token/元素，32 个元素 ≈ 917 token，可塞进 1024。
DEFAULT_MAX_ELEMENTS = 32


def format_scene(elements: List, goal: str, max_elements: int = 0) -> str:
    """把 YOLO 元素列表拼成 LLM 最易消化的场景描述文本。

    elements: List[Element]（detector.py 定义），每个含 label/center()/text/conf
    max_elements: >0 时限制进 prompt 的元素数（0=不限制）。

    截断策略：**命中 goal 的元素优先保留**，其余按原顺序。
    绝不能简单地取前 N 个——那样目标元素很可能被截掉，模型被迫对看不见的
    目标编坐标，等于在教它幻觉（这类数据训完才发现就晚了）。
    """
    if not elements:
        return f"目标：{goal}\n屏幕：未检测到任何 UI 元素。"
    els = list(elements)
    truncated = 0
    if max_elements and len(els) > max_elements:
        g = (goal or "").strip()
        hit, rest = [], []
        for e in els:
            t = getattr(e, "text", None) or ""
            lab = getattr(e, "label", "") or ""
            (hit if (g and (g in t or g in lab)) else rest).append(e)
        els = (hit + rest)[:max_elements]
        truncated = len(elements) - len(els)
    lines = [f"目标：{goal}", "当前屏幕包含以下 UI 元素："]
    for i, e in enumerate(els, 1):
        txt = f"「{e.text}」" if getattr(e, "text", None) else ""
        lines.append(f"{i}. {e.label}{txt} 中心坐标={e.center()} 置信度={e.conf:.2f}")
    if truncated:
        # 必须告诉模型列表是部分的，否则它会以为屏幕只有这些元素
        lines.append(f"（共检测到 {len(elements)} 个元素，"
                     f"上面只列出最相关的 {len(els)} 个，省略 {truncated} 个）")
    return "\n".join(lines)


JSON_SCHEMA_HINT = (
    '请严格只输出一个 JSON 对象（不要解释、不要代码块），格式：\n'
    '{"action": "click"|"type"|"scroll"|"wait"|"none", '
    '"target": "语义目标名(如确认按钮)", '
    '"coordinates": [x, y] 或 null, '
    '"text": "需要输入的文字或 null"}\n'
    '规则：若你知道目标坐标就填 coordinates；若只知道目标名就填 target，'
    '程序会按目标名从检测结果里解析坐标。'
)


# ---------------------------------------------------------------------------
# 3. LLM 桥：封装「调用 LLM + 强制 JSON」的两种接入方式
# ---------------------------------------------------------------------------
def build_prompt(scene: str = "", goal: str = "") -> str:
    """拼装喂给大脑的完整文本 prompt。

    关键契约：generate() 与轨迹落盘（trajectory.record 的 prompt 字段）**必须共用本函数**。
    训练数据与推理时 prompt 若不同源，分布不一致会直接毁掉微调效果，
    且这种错误往往要到训完才发现，等于白洗一遍数据。

    注意：format_scene() 已在首行写入「目标：{goal}」，此处**不可再前置目标**，
    否则同一个目标在 prompt 里出现两次，等于喂给模型冗余信号、污染训练数据。
    仅在 scene 为空时退回纯 goal。"""
    return (scene + "\n\n" + JSON_SCHEMA_HINT) if scene else (goal or "")


class LLMBridge:
    """LLM 接入协议。子类实现 respond()。"""

    is_real = False  # 是否真·外部大模型（如实标记，避免把本地推理冒充 LLM）

    def respond(self, prompt: str, image=None) -> str:
        raise NotImplementedError

    # ---- 统一的「决策」入口：喂场景 → 拿可执行 Action ----
    def decide(self, elements: List, goal: str, image=None,
               multimodal: bool = False, extra_context: str = "") -> Action:
        scene = format_scene(elements, goal)
        ctx_block = f"\n\n背景知识（来自知识层 GraphRAG）：\n{extra_context}" if extra_context else ""
        prompt = scene + ctx_block + "\n\n" + JSON_SCHEMA_HINT
        reply = self.respond(prompt, image=image if multimodal else None)
        data = _extract_json(reply)
        action = Action(raw=data)
        action.action = str(data.get("action", "none")).lower()
        action.target = data.get("target")
        action.text = data.get("text")
        coords = data.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            action.coordinates = (int(coords[0]), int(coords[1]))
        # 坐标缺失但有目标名 → 从检测元素里解析坐标（兜底）
        if action.coordinates is None and action.target and elements:
            action.coordinates = _resolve_coords_by_target(elements, action.target)
        return action

    # ---- 接 ② 投影层的新入口：generate(llm_feats) -> str ----
    def generate(self, llm_feats=None, scene: str = "", goal: str = "",
                 multimodal: bool = False) -> str:
        """大脑层统一入口，直接吃 ② 投影层的输出。

        llm_feats: ② 投影层产出 (n_patches, llm_dim) 的真实视觉特征。
            当前文本后端用 scene 文本推理，llm_feats 仅占位（多模态大脑 MiniMind
            上线后设 multimodal=True 直接喂入，无需额外文本）。
        scene: 由 detector.detect 的结构化元素拼成的自然语言（format_scene）。
        返回 LLM 原始响应字符串（含结构化 JSON 动作），交给 parse_llm_response 解析。
        """
        # 与轨迹落盘共用 build_prompt，保证训练/推理 prompt 逐字同源
        prompt = build_prompt(scene, goal)
        image = llm_feats if (multimodal and llm_feats is not None) else None
        return self.respond(prompt, image=image)

    def parse_llm_response(self, response: str, elements=None) -> Action:
        """把 generate 的响应字符串解析成可执行 Action（委托模块级函数）。"""
        return parse_llm_response(response, elements)


# ---------------------------------------------------------------------------
# 3.6 模块级解析：把 LLM 响应字符串解析成可执行 Action（generate 的配套）
# ---------------------------------------------------------------------------
def parse_llm_response(response: str, elements=None) -> Action:
    """把 generate() 的响应字符串解析成可执行 Action（JSON 抠取 + 坐标回退）。"""
    data = _extract_json(response)
    action = Action(raw=data)
    action.action = str(data.get("action", "none")).lower()
    action.target = data.get("target")
    action.text = data.get("text")
    coords = data.get("coordinates")
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        action.coordinates = (int(coords[0]), int(coords[1]))
    if action.coordinates is None and action.target and elements:
        action.coordinates = _resolve_coords_by_target(elements, action.target)
    return action


class MockLLM(LLMBridge):
    """自检用假 LLM：根据 goal 关键词返回结构化动作，验证整条链路不崩。"""

    def respond(self, prompt: str, image=None) -> str:
        # 从场景文本里找「确认/重试/取消」等语义词决定点哪个
        if "重试" in prompt:
            return json.dumps({"action": "click", "target": "重试按钮",
                               "coordinates": [660, 380], "text": None},
                              ensure_ascii=False)
        if "取消" in prompt:
            return json.dumps({"action": "click", "target": "取消按钮",
                               "coordinates": [200, 380], "text": None},
                              ensure_ascii=False)
        # 默认点「确认」
        return json.dumps({"action": "click", "target": "确认按钮",
                           "coordinates": [530, 320], "text": None},
                          ensure_ascii=False)


class OracleLLM(LLMBridge):
    """离线采集用的「规矩老师」：不调任何外部模型，直接按 goal 在检测结果里找最匹配
    的元素，输出带真实坐标的点击动作。产出 (prompt, response) 场景对齐，可直接当
    ③ MiniMind 的 SFT 监督对；外部真 VLM 上线后把 llm_backend 换掉即可，链路不变。

    为什么不直接用 MockLLM 采集：MockLLM 永远回固定坐标([530,320])，与场景无关，
    训出来的模型学到的不是「看屏→点对应按钮」，而是「无脑点固定点」，毫无价值。"""

    def generate(self, llm_feats=None, scene: str = "", goal: str = "",
                 multimodal: bool = False, elements=None) -> str:
        els = list(elements or [])
        # 提取 goal 目标关键词：「点击搜索」→「搜索」；「点击确定按钮」→「确定」
        kw = goal.replace("点击", "").replace("按钮", "").strip() or goal
        best, best_score = None, -1
        for e in els:
            label = (getattr(e, "text", None) or getattr(e, "label", None) or "")
            score = 0
            if kw and kw in label:
                score = 3
            elif kw and kw.lower() in label.lower():
                score = 2
            if score == 0 and label and goal and label in goal:
                score = 1
            if score > best_score:
                best_score, best = score, e
        # 没匹配到：退一步点第一个可见可点元素，保证有真实坐标（数据仍可用）
        if best is None and els:
            for e in els:
                if getattr(e, "label", None) in ("Button", "Utility Button", "Link", "Text"):
                    best = e
                    break
            best = best or els[0]
        if best is None:
            return json.dumps({"action": "wait", "target": None,
                               "coordinates": None, "text": None}, ensure_ascii=False)
        cx, cy = best.center()
        return json.dumps({"action": "click",
                           "target": (getattr(best, "text", None) or getattr(best, "label", None)),
                           "coordinates": [int(cx), int(cy)], "text": None},
                          ensure_ascii=False)


class ReasonerLLM(LLMBridge):
    """本地规则式「大脑替身」：无外部模型，但从 System2 的富文本 prompt 里读懂局面，
    返回与真 LLM 同构的 JSON（决策/计划/纠错三态）。用于离线演示「System2 充分推理」
    的完整闭环；接真模型时换 --llm-backend api，链路不变。

    诚实标记 is_real=False：它确实在「读 prompt → 结构化推理 → 写经验/调旋钮」，
    但用的是本地规则而非大模型，绝不冒充真 LLM。"""

    is_real = False

    def respond(self, prompt: str, image=None) -> str:
        # 按 prompt 关键词判别三类意图（纠错优先：其 prompt 也含「宏观调控旋钮」字样）
        if "诊断" in prompt or "纠正" in prompt or "撞死" in prompt or "失误" in prompt:
            return self._correct(prompt)
        if "宏观调控" in prompt or "上一段执行遥测" in prompt:
            return self._plan(prompt)
        return self._decide(prompt)

    # ---- 决策态：读局面 → 跳/蹲 + 时机补偿 ----
    def _decide(self, prompt: str) -> str:
        import re as _re
        kind = "cactus"
        m = _re.search(r"障碍类型[=\s]*(\w+)", prompt)
        if m and "bird" in m.group(1):
            kind = "bird"
        pair = "紧跟障碍=True" in prompt
        action = "squat" if kind == "bird" else "jump"
        rd = -0.03 if pair else 0.0
        return json.dumps(
            {"action": action, "react_delta": rd,
             "reason": f"本地推理：{kind}→{action}" + ("；双障碍略早跳以覆盖两个" if pair else "")},
            ensure_ascii=False)

    # ---- 计划/宏观调控态：读遥测与当前旋钮 → 调参 ----
    def _plan(self, prompt: str) -> str:
        import re as _re
        gnum = lambda name, d: (lambda m: float(m.group(1)) if m else d)(
            _re.search(name + r"=([\d.]+)", prompt))
        react, airtime, sh = gnum("reaction", 0.22), gnum("airtime", 0.60), gnum("s2_horizon", 1.20)
        deaths = 0
        m = _re.search(r"死亡=(\d+)", prompt)
        if m:
            deaths = int(m.group(1))
        speed = 0.0
        m = _re.search(r"平均速度=(\d+)", prompt)
        if m:
            speed = float(m.group(1))
        reason = []
        if deaths > 0:
            react = max(0.15, react - 0.01)
            airtime = max(0.40, airtime - 0.05)
            reason.append(f"窗口内死亡{deaths}次→略晚跳/允许更快补跳")
        if speed > 600:
            sh = min(1.80, sh + 0.10)
            reason.append(f"速度{speed:.0f}偏快→更早请示高层")
        if not reason:
            reason.append("执行平稳→维持当前旋钮")
        return json.dumps(
            {"steps": ["维持节奏：地面障碍跳、飞鸟蹲", "速度变快时提前请示高层",
                       "遇新情形交给高层学习"],
             "reaction": round(react, 3), "airtime": round(airtime, 3),
             "s2_horizon": round(sh, 3), "horizon": 8.0,
             "reason": "本地宏观调控：" + "；".join(reason)},
            ensure_ascii=False)

    # ---- 纠错态：读死亡上下文 → 诊断 + 修正 ----
    def _correct(self, prompt: str) -> str:
        import re as _re
        key = ""
        m = _re.search(r"情形键[=:\s]+([^\n,]+)", prompt)
        if m:
            key = m.group(1).strip().strip("'\"")
        # 纠错 prompt 不含"障碍类型"字段，靠情形键判定鸟/地刺（bird 情形→应蹲）
        kind = "bird" if "bird" in key else "cactus"
        return json.dumps(
            {"diagnosis": f"本地纠错：{kind} 情形触发时机偏晚/被误导经验带偏",
             "fix": "challenge", "key": key,
             "action": "jump" if kind != "bird" else "squat",
             "react_delta": -0.02,
             "knobs": {"reaction": 0.20, "airtime": 0.55},
             "rewrite": True,
             "reason": "死亡后反思：降权误导经验并写入更早发现时机，略早跳兜底"},
            ensure_ascii=False)


class UltralyticsLLM(LLMBridge):
    """首选方案：Ultralytics 官方 LLM 接口，多模态一步到位。
    用法：llm = UltralyticsLLM("gpt-5.6-luna")；llm.decide(els, goal, image=img, multimodal=True)
    依赖懒加载：未装 ultralytics 时 __init__ 不报错，调用才报错（上层可降级）。"""

    is_real = True

    def __init__(self, model: str = "gpt-5.6-luna"):
        self.model = model
        self._llm = None

    def _ensure(self):
        if self._llm is None:
            from ultralytics import LLM
            self._llm = LLM(self.model)
        return self._llm

    def respond(self, prompt: str, image=None) -> str:
        llm = self._ensure()
        if image is not None:
            return llm(prompt, image=image)
        return llm(prompt)


class APILLM(LLMBridge):
    """备用方案：标准 API（OpenAI SDK 风格）。把拼好的文本 + 指令发过去。
    适配任何兼容 chat.completions 的服务（OpenAI / 本地 vLLM / Ollama）。"""

    is_real = True

    def __init__(self, model: str = "gpt-4o", api_key: str = "", base_url: str = ""):
        self.model, self.api_key, self.base_url = model, api_key, base_url
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key or "sk-noauth"}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def respond(self, prompt: str, image=None) -> str:
        client = self._ensure()
        msgs = [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(model=self.model, messages=msgs,
                                              temperature=0.2)
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# 3.5 后端工厂：按 config 的 llm_backend 选择大脑（mock/ollama/openai/ultralytics）
# ---------------------------------------------------------------------------
# 项目根（src/perception/llm_bridge.py → src/perception → src → 根）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_dotenv(path: str = None) -> None:
    """从项目根的 .env.local 读密钥到环境变量（文件不入库，已在 .gitignore）。

    - 已存在的环境变量优先，不被文件覆盖（命令行/CI 显式设置时以它为准）
    - 只认 KEY=VALUE；# 开头为注释
    - 绝不打印任何值
    """
    path = path or os.path.join(_PROJECT_ROOT, ".env.local")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except FileNotFoundError:
        pass


_CFG_CACHE = None


def _cfg_runtime() -> dict:
    """读 config.json 的 runtime 段（带缓存），让任何入口都能拿到同一套大脑配置。"""
    global _CFG_CACHE
    if _CFG_CACHE is None:
        try:
            with open(os.path.join(_PROJECT_ROOT, "config.json"), "r",
                      encoding="utf-8") as f:
                _CFG_CACHE = json.load(f).get("runtime", {})
        except Exception:
            _CFG_CACHE = {}
    return _CFG_CACHE


def build_llm(backend: str = "mock", model: str = None,
              base_url: str = None, api_key: str = None) -> LLMBridge:
    _load_dotenv()          # 密钥从 .env.local 注入（不打印、不入库）
    cfg = _cfg_runtime()    # model / base_url 兜底读 config.runtime
    """llm_backend 映射：
    - mock        : 零依赖假 LLM（自检用，返回结构化 JSON 动作）
    - oracle      : 按 goal 查真值坐标（采集对齐数据用，不是推理）
    - ultralytics : Ultralytics 官方 LLM 接口（多模态，需 ultralytics）
    - ollama      : 本地 Ollama（默认 11434）
    - openai/api  : 任意 OpenAI 兼容接口（DeepSeek / 通义 / 混元 / 中转站…）

    参数优先级：显式入参 > 环境变量 > 默认值。
    环境变量：MGA_LLM_MODEL / MGA_LLM_BASE_URL / MGA_LLM_API_KEY
              （api_key 额外兼容 OPENAI_API_KEY）
    密钥只从环境变量读，绝不写进代码、配置或日志。
    """
    b = (backend or "mock").lower()
    if b == "mock":
        return MockLLM()
    if b == "oracle":
        return OracleLLM()
    if b == "reasoner":
        return ReasonerLLM()
    if b == "ultralytics":
        return UltralyticsLLM(model or os.getenv("MGA_LLM_MODEL") or "gpt-5.6-luna")
    if b in ("gpt6", "gpt-6", "gpt-6-astra"):
        # L2 兜底 grounding/反思层（桌面 CU 方案 §8）：直连 GPT-6 Astra。
        # 密钥只从环境变量/.env.local 读（_load_dotenv 已注入），绝不写代码/日志。
        # 无 key 时不崩：APILLM 实例化成功，仅真正调用 respond 时才报网络错 → 降级 L3。
        m = model or os.getenv("MGA_LLM_MODEL") or "gpt-6-astra"
        u = base_url or os.getenv("MGA_LLM_BASE_URL") or cfg.get("llm_base_url") or ""
        k = api_key or os.getenv("MGA_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        return APILLM(model=m, api_key=k, base_url=u)
    if b in ("ollama", "openai", "api", "custom"):
        m = (model or os.getenv("MGA_LLM_MODEL") or cfg.get("llm_model")
             or ("llama3" if b == "ollama" else "gpt-4o"))
        u = (base_url or os.getenv("MGA_LLM_BASE_URL") or cfg.get("llm_base_url")
             or ("http://localhost:11434/v1" if b == "ollama" else ""))
        k = api_key or os.getenv("MGA_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        return APILLM(model=m, api_key=k, base_url=u)
    if b in ("hy3", "hunyuan", "hunyuan3", "tencent"):
        # 混元 / 其他 OpenAI 兼容模型：走 APILLM，模型名/端点/密钥从环境变量或
        # 配置读，绝不写进代码。无 key 时不崩：调用 respond 才报网络错 → 上层降级。
        m = model or os.getenv("MGA_LLM_MODEL") or cfg.get("llm_model") or "hy3"
        u = base_url or os.getenv("MGA_LLM_BASE_URL") or cfg.get("llm_base_url") or ""
        k = api_key or os.getenv("MGA_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        return APILLM(model=m, api_key=k, base_url=u)
    raise ValueError(f"未知 llm_backend: {b}")


# ---------------------------------------------------------------------------
# 4. 多轮对话（Multi-shot）：弱算力下小模型分多轮逐步达成目标
# ---------------------------------------------------------------------------
def multi_shot_decide(bridge: LLMBridge, elements: List, goal: str,
                      image=None, max_turns: int = 3) -> Action:
    """把复杂任务拆成多轮：
    第1轮：问「应该对哪类元素做什么动作？」拿到 target。
    第2轮：给定 target，问「它的坐标是多少？」补 coordinates。
    仍失败则第3轮用整个场景兜底。每轮都强制 JSON。
    """
    scene = format_scene(elements, goal)
    # 轮1：意图拆解
    r1 = bridge.respond(scene + "\n\n只回答操作意图JSON："
                              '{"action":"...","target":"..."}', image=image)
    d1 = _extract_json(r1)
    action = Action(action=str(d1.get("action", "none")).lower(),
                     target=d1.get("target"), raw=d1)
    # 轮2：补坐标
    if action.coordinates is None and action.target:
        r2 = bridge.respond(f"目标元素={action.target}。从场景里解析它的中心坐标，"
                            '只回答JSON：{"coordinates":[x,y]}', image=image)
        d2 = _extract_json(r2)
        coords = d2.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            action.coordinates = (int(coords[0]), int(coords[1]))
    # 兜底：整场景一次性决策
    if action.coordinates is None and elements:
        action.coordinates = _resolve_coords_by_target(elements, action.target)
    return action


# ---------------------------------------------------------------------------
# 5. JSON 解析 + 坐标回退（健壮性，避免 LLM 胡说导致链路崩）
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """从 LLM 回复里抠出第一个 JSON 对象，兼容代码块/前后废话。"""
    if not text:
        return {}
    # 优先匹配 ```json ... ``` 或 ``` ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 退化为首个 { 到末个 }
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except Exception:
            pass
    return {}


def _resolve_coords_by_target(elements: List, target: Optional[str]) -> Optional[tuple]:
    """按语义名从检测元素里找坐标（LLM 给 target 但没给坐标时兜底）。"""
    if not target or not elements:
        return None
    t = target.lower()
    for e in elements:
        label = (getattr(e, "text", None) or e.label or "").lower()
        if t in label or label in t:
            return e.center()
    return None


# ---------------------------------------------------------------------------
# 6. 适配器：把结构化桥接进现有 correction_pipeline（不改其契约）
# ---------------------------------------------------------------------------
def make_json_decider(bridge: LLMBridge):
    """返回 correction_pipeline 兼容的 llm_decide(ctx, goal) 调用。
    LLM 回结构化 Action → 转成 Element（target 当 label，coordinates 当中心），
    原管线照旧执行点击 + 记忆回写。"""
    from detector import Element

    def decider(ctx: str, goal: str):
        # 从 ctx 可能无法直接拿到 elements，故让 bridge 用空元素列表走兜底解析；
        # 真实部署应把 elements 透传给 bridge.decide(elements, goal)。
        # 这里演示：用 goal 关键词 + 解析 ctx 里的坐标。
        action = bridge.decide([], goal)
        cx, cy = action.coordinates or (0, 0)
        half = 20
        return Element(id="llm", label=action.target or "target",
                       bbox=(cx - half, cy - half, cx + half, cy + half),
                       conf=1.0)
    return decider


# ---------------------------------------------------------------------------
# 7. 自检（python llm_bridge.py，零外部依赖）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from detector import Element  # 复用同一份结构化元素定义

    # 模拟 YOLO 检测输出（步骤1：结构化提取，已是机器可读对象）
    els = [
        Element(id="e1", label="Button", bbox=(500, 300, 560, 340), conf=0.92, text="确定"),
        Element(id="e2", label="Button", bbox=(620, 360, 700, 400), conf=0.90, text="重试"),
        Element(id="e3", label="Text Input", bbox=(100, 100, 300, 130), conf=0.80),
    ]

    print("=== 步骤1+2：YOLO 结构化输出 → 自然语言场景 ===")
    scene = format_scene(els, goal="点击重试按钮")
    print(scene)

    bridge = MockLLM()

    print("\n=== 步骤3a：单轮强制 JSON 决策（听懂 → 可执行 Action）===")
    act = bridge.decide(els, goal="点击重试按钮")
    print("LLM 决策：", act.raw)
    print(f"可执行？{act.executable()}  动作={act.action} 坐标={act.coordinates}")

    print("\n=== 步骤3b：多轮对话（弱算力小模型逐步决策）===")
    act2 = multi_shot_decide(bridge, els, goal="点击确定按钮")
    print("多轮决策：", act2.raw, "→ 坐标", act2.coordinates)

    print("\n=== 鲁棒性：LLM 返回带废话的 JSON 也能抠出来 ===")
    messy = '好的，这是结果：\n```json\n{"action":"click","target":"取消","coordinates":[200,380]}\n```\n请执行。'
    print("解析结果：", _extract_json(messy))

    print("\n=== 接进 correction_pipeline 闭环（结构化决策 → 点击 → 记忆）===")
    from detector import correction_pipeline, Element as _E, TargetLocator

    class MockVisual(TargetLocator):
        def detect(self, frame):
            return [
                _E(id="e1", label="Button", bbox=(500, 300, 560, 340), conf=0.92, text="确定"),
                _E(id="e2", label="Button", bbox=(620, 360, 700, 400), conf=0.90, text="重试"),
            ]

    decider = make_json_decider(bridge)
    loc = MockVisual()
    sys1 = {"e1": (530, 320)}
    out = correction_pipeline("fake.png", sys1.get, loc, decider,
                               lambda x, y: (print(f"  [CU] 点击 {x},{y}") or True))
    print("闭环结果：", out["status"], "→", out["target"], out["center"])

    print("\n结论：YOLO 说话（bbox 列表）→ 桥转自然语言 → LLM 回 JSON 动作 → "
          "Computer Use 直接执行 → 记忆回写。整条链路零外部依赖可跑。")
