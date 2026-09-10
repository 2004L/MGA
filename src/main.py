"""
MGA · 端到端最小闭环编排（src/main.py）
====================================
    把已落地的模块串成一个能跑的 MGA 雏形：
    memory_system.py       (④ 知识层：经验记忆，自研)
    perception/detector.py (M3 定位器：集成 ScreenParser，带三级降级)
    perception/encoder.py  (① 视觉编码器：截图→特征网格，接真骨干/像素降级)
    projection/projector.py(② 投影层：MLP/Q-Former，特征网格→LLM 空间)
    perception/llm_bridge.py (YOLO→LLM 翻译桥：强制 JSON 动作)
    predictive/frame.py    (⑤ System1 预判帧：自研，可训练反射)
    economy/token.py       (内在动机：token 生存经济，自研)
    exec/computer_use.py   (⑥ 执行层：Computer Use 真封装 + 安全护栏)
    trajectory.py          (③ MiniMind 训练数据：真实反馈轨迹收集)

后端可插拔（build_backends）：
    use_real=False  默认 → 全 Mock 后端，零依赖可直接 python src/main.py 自检
    use_real=True   → 接真 ScreenParser + UltrALyticsLLM + ComputerUse（需装依赖）
    live=True       → 关闭 DRY_RUN 真动鼠标（需先确认护栏/区域）

闭环（触发式：平日 System1 静默，偏差才唤醒关键帧）：
    每帧: System1 物理预判 -> 与实测比较
        ├─ 无偏差/置信度高/非分布偏移 -> SILENT（纯数学，零 NN）
        └─ 有偏差/分布偏移 -> KEYFRAME（关键帧才唤醒贵模块，平日零开销）
              -> ① 定位器锁坐标 -> ①→② 投影层(真实视觉特征→LLM空间)
              -> ③ 大脑(当前文本路径；llm_feats 已留待 MiniMind) -> ④ Computer Use 点击
              -> ④ 经验记忆回写 + ⑤ 补偿样本回流训练 System1 残差
              -> ⑤ 轨迹落盘(图像→响应→动作→结果，供③ MiniMind 自训练)

运行：
    python src/main.py              # Mock 自检
    python src/main.py --real       # 接真实后端（DRY_RUN 不真动）
    python src/main.py --real --live # 真动鼠标（危险动作仍需人工确认）
"""

from __future__ import annotations
import sys, os, json, uuid, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_system import MemorySystem
from perception.detector import (TriLevelLocator, ScreenParserBackend,
                                  OCRBackend, UIALocator, Element, TargetLocator)
from perception.llm_bridge import (UltralyticsLLM, format_scene,
                                    parse_llm_response, build_llm, build_prompt,
                                    DEFAULT_MAX_ELEMENTS)
from predictive.frame import SelfMotionFrame
from economy.token import TokenEconomy
from exec.computer_use import ComputerUse, SafetyGuard
from knowledge.graphrag import KnowledgeGraph, LLMExtractor
from projection.projector import build_projector              # ② 投影层（MLP/Q-Former）
from perception.encoder import build_vision_encoder          # ① 真实视觉编码器（特征网格）
from perception.events import SemanticEventDetector           # ① 语义事件触发（几何之外的内容变化唤醒）
from trajectory import TrajectoryCollector                    # ③ MiniMind 训练数据收集


# 任务目标：原先是散落各处的字面量，训练时根本不知道在学哪个任务。
# 现统一为此常量并随轨迹落盘（schema 的 goal 字段），避免目标漂移。
GOAL = "点击确定按钮"


# ---------------------------------------------------------------------------
# Mock 后端（零依赖自检用）
# ---------------------------------------------------------------------------
class MockScreen:
    def __init__(self, n_ticks=40):
        self.vel = (1.0, 0.2)
        self.n_ticks = n_ticks

    def actual_pos(self, t: float) -> tuple:
        x = 100 + self.vel[0] * t * 100
        y = 100 + self.vel[1] * t * 100
        if 1.5 <= t < 2.0:                 # 世界突变：物体被推了一下
            x += 60
            y += 40
        return (x, y)

    def measure(self, t: float) -> np.ndarray:
        return np.array(self.actual_pos(t), dtype=float)

    def shot(self, t: float):
        return {"t": t, "pos": self.actual_pos(t)}


class MockVisual:
    def detect(self, frame):
        px, py = frame["pos"]
        return [Element(id="btn_confirm", label="Button",
                        bbox=(int(px) - 30, int(py) - 20, int(px) + 30, int(py) + 20),
                        conf=0.95, text="确定")]


class MockComputerUse:
    def click(self, x, y) -> bool:
        print(f"        [Computer Use] 点击 ({x}, {y})")
        return True


# ---------------------------------------------------------------------------
# 真实后端
# ---------------------------------------------------------------------------
class RealScreen:
    """真实环境：每帧截图 → 定位器找被跟踪物体 → 返回其中心作为 System1 实测。
    （YOLO 毫秒级，远便宜于 LLM；定位器仅在测量/关键帧调用，符合触发式原则）"""

    def __init__(self, cu: ComputerUse, locator: TargetLocator,
                 track_text: str = "确定"):
        self.cu, self.locator, self.track_text = cu, locator, track_text
        self.last = np.array([0.0, 0.0])
        self.last_elements = []            # 本帧检测结果缓存（关键帧分支复用，省一次推理）

    def shot(self, t: float):
        try:
            return self.cu.screenshot()       # 截图路径/数组，喂给定位器
        except Exception:
            # 无截屏依赖（pyautogui 未装/无显示）→ 合成一张随时间变化的确定性截图，
            # 闭环仍可跑（编码器/定位器会各自降级），用于 headless 验证层集成。
            return self._synthetic_shot(t)

    def _synthetic_shot(self, t: float) -> np.ndarray:
        """兜底合成截图：480x640 渐变 + 一个随时间漂移的亮块，保证帧间有真实差异，
        让 ①→② 投影层能展示对截图的真实响应（非随机）。"""
        H, W = 480, 640
        base = np.linspace(0, 255, W, dtype=np.float32)[None, :].repeat(H, 0)
        img = np.stack([base, base * 0.6, base * 0.3], axis=-1)
        bx = int((W // 2) + 120 * np.sin(t))
        by = int((H // 2) + 80 * np.cos(t * 0.7))
        img[by - 25:by + 25, bx - 40:bx + 40] = (255, 230, 60)
        return img.astype("uint8")

    def measure(self, t: float) -> np.ndarray:
        frame = self.shot(t)
        try:
            els = self.locator.detect(frame)
        except Exception:
            return self.last.copy()
        self.last_elements = els          # 供关键帧分支复用，避免同一帧重复推理
        if not els:
            return self.last.copy()
        # 匹配条件：文字(OCR) 或 类别名(如 ScreenParser 的 Button) 命中 track_text。
        # 原实现只比 text，而 ScreenParser 只给类别名、OCR 未装时 text 恒为 None，
        # 结果测量值永远不更新 → 跟踪器收敛 → 0 关键帧 → 采不到任何轨迹。
        cands = [e for e in els
                 if (self.track_text and (getattr(e, "text", None) == self.track_text
                                          or getattr(e, "label", None) == self.track_text))
                 or e.id == "btn_confirm"]
        if not cands:
            return self.last.copy()
        # 同类元素往往几十个（实测一屏 51 个 Button）。若每帧都取第一个，
        # 匹配到的可能是完全不同的按钮，位置乱跳 = 全是假关键帧。
        # 故按「离上一帧最近」挑选，保证跟踪的时间连续性。
        if len(cands) > 1 and np.any(self.last):
            cands.sort(key=lambda e: np.linalg.norm(np.array(e.center()) - self.last))
        self.last = np.array(cands[0].center(), dtype=float)
        return self.last.copy()


class JsonLLM:
    """（保留兼容）把 llm_bridge 包成 .decide(elements, goal)->Element。
    新链路统一走 llm.generate(llm_feats, scene, goal) + parse_llm_response，见 run() 关键帧分支。"""

    def __init__(self, bridge):
        self.bridge = bridge

    def decide(self, elements, goal: str, context: str = None, **kwargs) -> Element:
        act = self.bridge.decide(elements, goal, extra_context=context or "")
        cx, cy = act.coordinates or (0, 0)
        half = 20
        return Element(id="llm", label=act.target or "target",
                       bbox=(int(cx - half), int(cy - half),
                             int(cx + half), int(cy + half)), conf=1.0)


def build_backends(use_real: bool = False, live: bool = False,
                   track_text: str = "确定", llm_backend: str = "mock",
                   ocr_scale: float = 1.0, llm_model: str = None,
                   llm_base_url: str = None):
    """工厂：返回 (screen, locator, llm, cu)。Mock 零依赖；real 接真依赖。
    llm_backend 按 config 选择大脑（mock/oracle/ollama/openai/api/ultralytics）。
    llm_model / llm_base_url 透传给 build_llm（密钥只走环境变量，不在此传）。"""
    if not use_real:
        return (MockScreen(n_ticks=40),
                TriLevelLocator(visual=MockVisual(), ocr=OCRBackend(), uia=UIALocator()),
                build_llm(llm_backend, model=llm_model, base_url=llm_base_url),
                MockComputerUse())

    # 真实后端
    # 执行层策略化：约束全在 config.exec（默认全放开 + 急停）。
    # --live 只决定"这一跑是否真动鼠标"，策略里的 dry_run 是默认值，
    # 命令行 --live 优先。注意原逻辑 require_confirm=live 是反的（真跑才弹确认），已废。
    guard = SafetyGuard.from_policy(dry_run=(not live))
    cu = ComputerUse(guard)
    # UIA 作为三级降级（YOLO → OCR → UIA）的运行时兜底元件：
    # 仅在关键帧定位器调用时产出元素、注入 LLM prompt，不参与训练数据落盘。
    locator = TriLevelLocator(visual=ScreenParserBackend(),
                              ocr=OCRBackend(scale=ocr_scale), uia=UIALocator())
    llm = build_llm(llm_backend, model=llm_model,
                    base_url=llm_base_url)       # ③ 大脑：按 llm_backend 选
    screen = RealScreen(cu, locator, track_text=track_text)
    return screen, locator, llm, cu


def load_runtime() -> dict:
    """读取 config.json 的 runtime 段（use_real/auto_execute/visual_backend/...）。
    CLI 参数优先于配置文件；配置文件缺段时返回空。"""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f).get("runtime", {})
    except Exception:
        return {}


def verify_realtime_chain() -> None:
    """无外部依赖验证 ④ 项：检测链路连通 / 投影对齐 / LLM 稳定 / 端到端耗时。
    用真实像素编码器（无真骨干降级）+ 已训练投影器 + Mock LLM.generate，
    证明『真实图像→真实特征→投影对齐→LLM输出』数据流通，且基准延迟可记录。"""
    print("=== 真实闭环链路自检（无需 ultralytics/pyautogui）===")
    import time
    import numpy as np
    from perception.detector import Element

    enc = build_vision_encoder(real=True, enc_dim=64, n_patches=16)
    proj = build_projector("mlp", enc_dim=64, out_dim=64)
    try:
        proj.load(os.path.join("blobs", "projector_mlp.pkl"))
    except Exception as e:
        # 不再静默：权重加载失败 = 用的是随机初始化的 projector，
        # 表现"能跑通但效果不对"，极易误判成模型问题
        print(f"  [main] ⚠️ projector 权重加载失败（将使用随机初始化）："
              f"{type(e).__name__}: {e}")
    llm = build_llm("mock")

    shot = np.random.randint(0, 255, (480, 640, 3), dtype="uint8")  # 模拟真实截图
    t0 = time.time()
    # 1) 检测链路连通：vision_feats 非空且 seq_len>0
    vf = enc.encode(shot)
    assert vf.size > 0 and vf.shape[0] > 0, "❌ vision_feats 为空"
    print(f"  [1] 检测链路连通: vision_feats 非空, shape={vf.shape}")
    # 2) 投影对齐：末维 == 投影器 out_dim
    lf = proj.forward(vf)
    assert lf.shape[1] == proj.out_dim, "❌ 投影输出维度不匹配"
    print(f"  [2] 投影对齐: llm_feats{lf.shape} 末维={lf.shape[1]} == out_dim={proj.out_dim}")
    # 模拟 M3 结构化元素 → scene 文本（真实模式由 detector.detect 产出）
    els = [Element(id="e1", label="Button", bbox=(100, 100, 160, 140),
                   conf=0.9, text="确定")]
    scene = format_scene(els, GOAL)
    # 3) LLM 生成稳定：非空格、可解析动作
    resp = llm.generate(lf, scene=scene, goal=GOAL, multimodal=False)
    assert isinstance(resp, str) and len(resp.strip()) > 0, "❌ LLM 响应为空/乱码"
    action = parse_llm_response(resp, els)
    print(f"  [3] LLM 生成稳定: 响应='{resp}' → 动作={action.action} 坐标={action.coordinates}")
    # 4) 端到端耗时：截图→输出 基准
    dt = (time.time() - t0) * 1000
    print(f"  [4] 端到端耗时(截图→输出): {dt:.1f} ms（本机基线，供后续优化对比）")
    print("✅ 真实闭环链路自检通过：检测连通 / 投影对齐 / LLM 稳定 / 已记录基准延迟")


# ---------------------------------------------------------------------------
# 编排主循环
# ---------------------------------------------------------------------------
CLICKABLE_LABELS = ("Button", "Icon", "CheckBox", "RadioButton",
                    "MenuItem", "Tab", "Input", "Link")


def pick_auto_goal(els, cursor: int = 0):
    """自动出题：从当前屏幕**实际存在**的带文字元素里挑一个当 goal。

    为什么必须自动：长时程采集时用户会切各种软件，写死的 goal（如「搜索」）
    在别的页面上根本不存在 → oracle 找不到目标 → 采到一堆「目标不存在」的
    垃圾样本。从屏幕实际内容出题，(题目, 答案) 在任何页面都天然成立，
    而且屏幕一变、可出的题就变，多样性是白送的。

    cursor 递增轮换：同一屏连续采多帧时会出不同的题，避免 100 帧只围着
    同一个按钮转（这正是之前 111 条只有 3 种答案的根因）。
    """
    cands = []
    for e in els:
        t = (getattr(e, "text", None) or "").strip()
        if len(t) < 2 or len(t) > 12:
            continue
        # OCR 噪声过滤：实测会挑出 "0-?"「拍 拜」这类误识字。拿它当题目，
        # 模型学到的就是噪声映射，白占一条样本。要求至少 2 个有效字符且无问号。
        if "?" in t or "？" in t:
            continue
        n_ok = sum(1 for ch in t if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if n_ok < 2:
            continue
        # 文字来源：uia 零误差优先，ocr 残留仅作兜底
        src = getattr(e, "text_src", "ocr") or "ocr"
        cands.append((getattr(e, "label", "") or "", t, n_ok, src))
    if not cands:
        return None
    # 优先级：① UIA 零误差文字 → ② 可点控件 → ③ 短文字（更像按钮标题）
    cands.sort(key=lambda x: (x[3] != "uia", x[0] not in CLICKABLE_LABELS, len(x[1])))
    return cands[cursor % len(cands)][1]


def run(use_real: bool = False, live: bool = False, minimal: bool = False,
        track_arg: str = "", frames: int = 0, threshold: float = 15.0,
        collect_every: int = 0, goal_arg: str = "", llm_arg: str = "",
        llm_model_arg: str = "", llm_base_url_arg: str = "",
        max_elements_arg: int = 0, auto_goal: bool = False,
        interval: float = 0.0, ocr_scale: float = 1.0,
        minutes: float = 0.0):
    mem = MemorySystem("mga_demo.db")
    eco = TokenEconomy(init=1000.0, safe=400.0, min_=100.0)
    kg = KnowledgeGraph()                 # ④ 外部知识层（GraphRAG，numpy+networkx 自研）
    # 知识层：**采集模式强制用 MockExtractor**（零依赖），仅真实使用模式上真 LLM 抽取。
    # 历史教训（2026-09-03 控制台实战）：之前只要 use_real 就建 LLMExtractor，关键帧
    # 每帧调一次，没 API key 时 except 一条错误刷屏，把采集日志全淹没。采集的目的是
    # 攒训练数据，不是用知识增强——本就不该被外部 LLM 拖累，也刷屏污染日志。
    # MockExtractor 零依赖符合 MGA 降级哲学；接口完全一致，KnowledgeGraph 无需改。
    is_collect_mode = bool(collect_every > 0
                           or (llm_arg or "").lower() == "oracle")
    if use_real and not is_collect_mode:
        try:
            from perception.llm_bridge import UltralyticsLLM
            kg = KnowledgeGraph(extractor=LLMExtractor(UltralyticsLLM("gpt-5.6.luna")))
            print(f"[kg] 真实模式：知识层使用 LLMExtractor({llm_backend})")
        except Exception as ex:
            print(f"[kg] 真实 LLM 抽取初始化失败，降级 MockExtractor: {ex}")
            kg = KnowledgeGraph()         # 默认 MockExtractor
    elif is_collect_mode:
        print(f"[kg] 采集模式：知识层用 MockExtractor（不调外部 LLM）")
    # 用 M5 经验数据预热知识层（演示「经验→外部知识」打通，关键帧时 Local Search 可召回）
    for m in mem.search_memory("关键帧", top_k=20):
        kg.insert_document(m.get("summary", ""), doc_id=m.get("memory_id", ""))

    # ---- 运行时配置：config.json runtime 段 与 CLI 参数合并（CLI 优先）----
    runtime = load_runtime()
    use_real = bool(use_real or runtime.get("use_real", False))
    auto_execute = bool(live or runtime.get("auto_execute", False))
    llm_backend = llm_arg or runtime.get("llm_backend", "mock")
    # 外部大模型：模型名 / base_url 走 config 或 CLI；api_key 只从环境变量读
    llm_model = llm_model_arg or runtime.get("llm_model", "")
    llm_base_url = llm_base_url_arg or runtime.get("llm_base_url", "")
    visual_backend = runtime.get("visual_backend", "screenparser")
    capture_source = runtime.get("capture_source", "screen")
    # ① 感知层跟踪目标：System1 靠它的位移产生预测误差来触发关键帧。
    # 匹配不上 → 测量值恒定 → 跟踪器自适应收敛 → 0 关键帧 → 采不到任何轨迹。
    # CLI --track 优先（config 里 track_text 的说明写了 COCO 权重该如何填）。
    track_text = track_arg if track_arg else runtime.get("track_text", "确定")
    # 采集用目标：默认 GOAL，可用 --goal 覆盖为屏幕上真实存在的元素（如「搜索」），
    # 否则 trajectory 的 goal 字段与目标屏不匹配，训练出的模型会去点不存在的按钮。
    goal = goal_arg or GOAL
    # 进 prompt 的元素数上限。全屏上百元素 ≈7500 token，远超小模型训练窗口，
    # 必须在 format_scene 侧统一截断（训练/推理同源），不能只在训练脚本里截。
    max_elements = int(max_elements_arg or runtime.get("max_elements", 0)
                       or DEFAULT_MAX_ELEMENTS)
    screen, locator, llm, cu = build_backends(use_real, live,
                                              track_text=track_text,
                                              llm_backend=llm_backend,
                                              ocr_scale=ocr_scale,
                                              llm_model=llm_model,
                                              llm_base_url=llm_base_url)
    if use_real:
        print(f"    [① 跟踪目标] track_text='{track_text}'"
              f"（若采不到轨迹，多半是它没匹配上；用 --track '类别名' 改）")

    # ② 投影层：加载已训练投影器（与编码器维度对齐）；缺失则新建未训练版
    proj_path = os.path.join("blobs", "projector_mlp.pkl")
    try:
        proj = build_projector("mlp", enc_dim=64, out_dim=64)
        proj.load(proj_path)
        proj_src = f"已加载 {proj_path}"
    except Exception:
        proj = build_projector("mlp", enc_dim=64, out_dim=64)
        proj_src = f"未找到 {proj_path}，新建未训练投影器（接真骨干后需先对齐训练）"
    enc_dim = getattr(proj, "enc_dim", 64)
    # ① 视觉编码器：real=True 走真实截图/真骨干，real=False 用零依赖占位
    encoder = build_vision_encoder(real=use_real, enc_dim=enc_dim,
                                 n_patches=16, backend=visual_backend)
    # ⑤ 轨迹收集：闭环产出的真实反馈数据，供 ③ MiniMind 自训练
    traj = TrajectoryCollector(path=os.path.join("blobs", "trajectories.jsonl"))
    print(f"    [② 投影层] {proj_src}  enc_dim={enc_dim}  "
          f"编码器={'真实/像素降级' if use_real else 'Mock'}  "
          f"轨迹→{traj.path}")

    frame = SelfMotionFrame(pos=np.array([100.0, 100.0]),
                             vel=np.array([100.0, 20.0]),
                             threshold=threshold, dim=2)

    # ① 语义事件触发：几何偏差外，画面「内容」变化（某类数量突变/关注类出现消失）
    # 也唤醒 System2。静止桌面几何偏差恒为 0，但 UI 元素一直变（实测每帧类数都变），
    # 这是解开上轮「采不到轨迹」死结的关键层。零额外推理：复用本帧已有的检测框计数签名。
    # watch_classes 含 Button —— 「按钮数量变化/出现消失」都是操控相关的高价值事件。
    sem = SemanticEventDetector(count_delta=3, watch_classes=["Button"])
    goal_cursor = 0      # --auto-goal 轮换游标：同一屏连续采多帧也出不同的题

    # 真实模式下循环用时间驱动（每 0.1s 一帧），Mock 用固定 ticks
    n_ticks = frames if frames > 0 else (8 if use_real else (20 if minimal else 40))
    dt = 0.5 if use_real else 0.1
    # --minutes 墙钟时长：优先于 --frames，不受会话中断影响进程本身（脱离会话 run 时）
    deadline = (time.time() + minutes * 60) if minutes > 0 else None

    sys1_ticks = sys2_ticks = 0
    mode_tag = (f"{'真实' if use_real else 'Mock'}后端"
                f"[{visual_backend}/capt={capture_source}]"
                f" · 大脑={llm_backend}"
                f" · {'AUTO_EXEC' if auto_execute else '规划(不真点)'}"
                f"{'' if not use_real else (' (DRY_RUN)' if not live else ' (LIVE)')}")
    print(f"=== MGA 最小闭环演示 [{mode_tag}] 触发式：平日 System1 静默，偏差才唤醒关键帧 ===\n")
    for i in range(n_ticks):
        t = i * dt
        if deadline and time.time() > deadline:
            print(f"\n--minutes 到时（{minutes:.0f}min），停止采集。")
            break
        try:
            shot = screen.shot(t)                   # 截图（真=图像/路径，Mock=占位 dict）
            meas = screen.measure(t)                # System1 实测（真=定位器/YOLO）
        except Exception as ex:
            print(f"t={t:4.1f} [采集异常，跳过本帧] {type(ex).__name__}: {str(ex)[:80]}")
            if interval > 0:
                time.sleep(interval)
            continue
        r = frame.step(t, meas)                     # System1 物理预判 + 比较
        # ① 语义事件：复用本帧检测结果的类别计数签名做 diff（几何触发之外的第二路唤醒）
        sem_hit, sem_why = sem.check(getattr(screen, "last_elements", []) or [])
        # 采集模式：--collect-every N 强制每 N 帧一个关键帧，解决「静止屏采不到轨迹」死结。
        # 几何/语义触发在静止屏上永远不醒；采集阶段需要主动采样真实「感知→决策」样本供训练。
        collect_hit = collect_every > 0 and (i % collect_every == 0)
        fired = r.trigger or sem_hit or collect_hit
        # ①→② 真实视觉特征：真实模式每帧跑（验证链路连通 + 记录基准），平日 System1 静默但投影层仍低成本校验对齐
        vision_feats = llm_feats = None
        if use_real:
            vision_feats = encoder.encode(shot)
            llm_feats = proj.forward(vision_feats)
            if i % 5 == 0:
                print(f"t={t:4.1f} [①→②] vision_feats{vision_feats.shape} "
                      f"→ llm_feats{llm_feats.shape}  # token 数≈{llm_feats.shape[0]}")

        if not fired:
            sys1_ticks += 1
            mode = eco.act(use_sys2=False, env_reward=5.0)
            if i % 5 == 0:
                print(f"t={t:4.1f} SILENT(System1) err={r.error:6.2f} "
                      f"conf={r.confidence:.2f} {mode}")
        else:
            sys2_ticks += 1
            mode = eco.act(use_sys2=True, env_reward=0.0)
            if r.trigger:
                tag, why = "几何", f"几何偏差={r.error:.2f}"
            elif sem_hit:
                tag, why = "语义", f"语义事件：{sem_why}"
            else:
                tag, why = "采集", f"采集采样(每{collect_every}帧)"
            print(f"t={t:4.1f} KEYFRAME(System2/{tag}) {why} "
                  f"conf={r.confidence:.2f} {mode}")
            # ① 定位器锁坐标：优先复用本帧 measure 已算出的检测结果。
            #    ScreenParser 在 CPU 上单次约 2.8s，同一帧重复推理会直接拖垮采集速度。
            els = getattr(screen, "last_elements", None)
            if not els:
                els = locator.detect(shot)          # 三级降级
            if llm_feats is None:  # Mock 模式：关键帧才算投影
                vision_feats = encoder.encode(shot)
                llm_feats = proj.forward(vision_feats)
            if llm_feats is not None:
                print(f"        [①→②] vision_feats{vision_feats.shape} "
                      f"→ llm_feats{llm_feats.shape}  # token 数≈{llm_feats.shape[0]}")
            kg.upsert_elements(els)                   # M3 元素 → 知识图实体节点
            kg_ctx = kg.local_search(GOAL, top_k=2, hops=1)  # 关键帧才查，平日零开销
            print(f"        [知识层] Local Search 检索到背景：\n{kg_ctx}")
            # 自动出题：从本屏实际存在的元素里挑目标，保证任何页面都有合法题目
            if auto_goal:
                g = pick_auto_goal(els, goal_cursor)
                goal_cursor += 1
                if g:
                    goal = g
            # ③ 大脑：generate 吃 ② 投影层的真实 llm_feats（现文本后端用 scene，llm_feats 留待 MiniMind）
            scene = format_scene(els, goal, max_elements=max_elements)
            response = llm.generate(llm_feats, scene=scene, goal=goal,
                                    multimodal=False, elements=els)
            print(f"        [OUTPUT] {response}")
            action = parse_llm_response(response, els)    # ③ 解析成可执行 Action
            coords = action.coordinates or (0, 0)
            # ④ 执行层：auto_execute=False 只规划不真点（先观察响应是否合理）
            #    outcome 语义：已执行→执行是否成功；仅规划→规划是否命中真实目标。
            #    （旧写法规划一律记 False，会让 schema 里 sample_weight_or_filter=outcome
            #     把全部采集样本过滤掉，故必须区分 executed / planned_grounded）
            if auto_execute and action.coordinates:
                executed = True
                ok = cu.click(*coords)               # 过安全闸
                print(f"        [执行] 真实点击 {coords} -> {ok}")
            else:
                executed = False
                # 规划有效性：目标非空 + 坐标有效（非 (0,0) 占位）
                ok = bool(action.target) and bool(action.coordinates) and coords != (0, 0)
                print(f"        [规划] {action.action} @ {coords} "
                      f"（auto_execute={auto_execute}，未真正点击，规划有效={ok}）")
            mem.write_memory(
                title=f"关键帧纠正@{t:.1f}",
                summary=(f"物体偏移时锁定{action.target or 'target'}并点击"
                         f"{'成功' if ok else '未执行'}"),
                file_pointer=f"blobs/fix_{uuid.uuid4().hex[:6]}.md",
                tags=["correction", "gui", "keyframe"],
            )
            frame.step(t, meas, sys2_correction=meas)   # ⑤→① 补偿回流，System1 残差更新
            # ⑤ 轨迹落盘：真实「图像→响应→动作→结果」，供 ③ MiniMind 自训练
            traj.record(t=t, screenshot_ref=shot, llm_feats=llm_feats,
                        scene=[e.to_context(i + 1) for i, e in enumerate(els)],
                        response_target=action.target or "target",
                        action_coords=coords, outcome=ok, executed=executed,
                        goal=goal,
                        prompt=build_prompt(scene, goal),   # 与 generate() 同源
                        response=response,                   # SFT 监督目标 y
                        action=action.action,
                        max_elements=max_elements,
                        backend={"visual": visual_backend, "llm": llm_backend})
            print(f"        -> System2 纠正回流；轨迹已记录 #{traj.n}")

        # 采集节流：ScreenParser+OCR 都是 CPU 密集，全速跑会把用户电脑拖卡，
        # 而采集恰恰需要用户正常用电脑才能拿到多样化的屏幕。
        if interval > 0:
            time.sleep(interval)

    total = sys1_ticks + sys2_ticks
    print("\n=== 统计 ===")
    if total:
        print(f"总帧数={total}  System1静默={sys1_ticks} ({sys1_ticks/total:.0%})  "
              f"System2关键帧={sys2_ticks} ({sys2_ticks/total:.0%})")
    print(eco.report())
    print("记忆条目数:", len(mem.search_memory("关键帧", top_k=100)))
    print("轨迹条数(供③ MiniMind训练):", traj.n, "→", traj.path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="接真实后端（ScreenParser+LLM）")
    ap.add_argument("--live", action="store_true", help="关闭 DRY_RUN 真动鼠标（=auto_execute）")
    ap.add_argument("--minimal", action="store_true", help="Mock 少跑几帧")
    ap.add_argument("--verify", action="store_true",
                    help="仅跑真实闭环链路自检（检测连通/投影对齐/LLM稳定/基准延迟），不进入主循环")
    ap.add_argument("--track", default="",
                    help="① 感知层跟踪的目标文字/类别名，覆盖 config.runtime.track_text。"
                         "ScreenParser 用类别名(Button/Text...)，COCO 权重只认 laptop/person 等")
    ap.add_argument("--frames", type=int, default=0,
                    help="主循环帧数（采集时调大）。注意静止屏幕永远触发不了关键帧："
                         "System1 只在预测偏差超阈值时唤醒，这是刻意设计")
    ap.add_argument("--collect-every", type=int, default=0,
                    help="采集模式：每 N 帧强制一个关键帧（采样真实感知→决策），"
                         "解决静止屏几何/语义触发都不醒、采不到轨迹的死结。0=关闭（纯触发式）")
    ap.add_argument("--goal", default="",
                    help="覆盖采集目标（写进 trajectory.goal）。默认 GOAL='点击确定按钮'；"
                         "若屏幕上没有「确定」按钮，传屏幕上真实存在的元素名（如 搜索）以采到对齐数据")
    ap.add_argument("--llm", default="",
                    help="覆盖大脑后端（config.runtime.llm_backend）。采集真实对齐数据用 "
                         "--llm oracle（按 goal 在检测元素里找真实坐标，无需外部模型）；"
                         "默认 mock（固定响应，仅用于链路自检，不适合当训练数据）")
    ap.add_argument("--llm-model", default="",
                    help="外部大模型名（OpenAI 兼容接口），覆盖 config.runtime.llm_model。"
                         "也可用环境变量 MGA_LLM_MODEL。例：--llm openai --llm-model hy3")
    ap.add_argument("--llm-base-url", default="",
                    help="外部大模型 API 地址（OpenAI 兼容），覆盖 config.runtime.llm_base_url。"
                         "也可用环境变量 MGA_LLM_BASE_URL。"
                         "密钥一律走环境变量 MGA_LLM_API_KEY（不写进命令行/配置，防泄露）")
    ap.add_argument("--max-elements", type=int, default=0,
                    help=f"进 prompt 的 UI 元素数上限（默认 {DEFAULT_MAX_ELEMENTS}）。"
                         f"全屏上百元素≈7500 token 会撑爆小模型训练窗口；"
                         f"截断在 format_scene 内完成且与推理同源，命中 goal 的元素优先保留。")
    ap.add_argument("--auto-goal", action="store_true",
                    help="自动出题：从当前屏幕实际存在的带文字元素里轮换挑目标。"
                         "长时程采集必开——写死的 goal 在你切到别的软件后就不存在了，"
                         "会采到一堆『目标不存在』的垃圾样本")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="每帧之间 sleep 的秒数。采集需要你正常用电脑，"
                         "而 ScreenParser+OCR 吃满 CPU，不节流会把机器拖卡（建议 2~3）")
    ap.add_argument("--ocr-scale", type=float, default=1.0,
                    help="OCR 输入缩放（0.5 约把 1080p 全图 OCR 从 18s 降到 ~5s）。"
                         "识别率略降，但用于『按目标名找按钮』足够")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="墙钟采集时长（分钟）。优先于 --frames，按真实时间停止，"
                         "适合『挂着让你正常用电脑 N 分钟』的采集模式。建议 30")
    ap.add_argument("--threshold", type=float, default=15.0,
                    help="System1 触发阈值(像素)。调小更敏感、关键帧更多；"
                         "调小意味着样本从『显著异常』变成『轻微抖动』，会影响数据分布，慎用")
    args = ap.parse_args()
    if args.verify:
        verify_realtime_chain()
    else:
        run(use_real=args.real, live=args.live, minimal=args.minimal,
            track_arg=args.track, frames=args.frames, threshold=args.threshold,
            collect_every=args.collect_every, goal_arg=args.goal,
            llm_arg=args.llm, max_elements_arg=args.max_elements,
            llm_model_arg=args.llm_model, llm_base_url_arg=args.llm_base_url,
            auto_goal=args.auto_goal, interval=args.interval,
            ocr_scale=args.ocr_scale, minutes=args.minutes)
