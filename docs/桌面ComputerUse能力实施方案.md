# 桌面 Computer Use 能力实施方案（Path B · 双系统）

> 版本：2026-09-08 v5 ｜ 状态：🟢 M1–M6 闭环 + M7 小恐龙自操作 + M8 微信自主聊 + M9 真机 OCR 读消息；L2 待真实 API key；发送红线用户 2026-09-08 拍板有意识放宽（范围受控）
> 关联：`GTA5适配技术方案.md`（同双系统，桌面优先于 GTA5 推进）｜ `GTA5感知端实施计划.md`

## 0. 结论

把"桌面上非高危重复任务快速自动化"做成**双系统 + 安全门禁**的可落地能力：

- **S1（快）**＝ YOLOE 开放词汇 UI 定位 + 残差（本地、CPU、逐帧）。
- **S2（慢）**＝ LLM 任务规划 + 失败反思 + 纠错；看不懂/失败时升档调 **GPT-6 Astra** 做 grounding/反思（L2 兜底，仅关键帧、节流、按 token 计费）。
- **执行层**＝ 复用已存在的 `src/exec/computer_use.py`（`ComputerUse` + `SafetyGuard` + `SendInput`/`PyAutoGUI` 后端 + 窗口管理）。**不重造**。
- **安全门禁**＝ 新增 `IntentGate`（意图级）：`删除/发送` 出自动化范围直接拦截（连准备都不做）；`支付` 卡人工确认点；其余放行后再过 `SafetyGuard` 的 region/危险文本闸。

诚实缺口（v2 更新）：主视觉通道已替换为 **ScreenParser**（`docling-project/ScreenParser`，YOLO11-Large，25.4M 参数，55 类 GUI 控件），M2 已真通验证（权重下载落盘 + 模型加载 + 推理 + 管线标 `screenparser`，EXIT=0）。**YOLOE 开放词汇权重在 dino 侧仍 404 未下载**，降级为可选升级通道（代码 `OpenVocabLocator` 已就位，权重不可达时不谎报 `active_backend`）。GPT-6 Astra（2026-09-03 发布、灰度未全开）的 L2 接 Code 已完成（`build_llm("gpt6")`），但**真实兜底验证待真实 API key**——无 key 时 L2 常驻「代码就绪、不调用」状态，不假装可用。

## 1. 现状（架构真相）

| 层 | 状态 | 真实位置 |
|---|---|---|
| 双系统框架 S1/S2/记忆/纠错 | ✅已接通 | `src/agent/{system1,system2,orchestrator,memory}.py`，dino 实证闭环 |
| 执行层（键鼠/窗口/拟人化/审计/急停） | ✅已接通 | `src/exec/{computer_use,backends,window,tools}.py` |
| 执行安全闸（confirm/region/dangerous-text/急停） | ✅已接通 | `src/exec/computer_use.py::SafetyGuard` |
| LLM 桥（mock/oracle/reasoner/api/ultralytics/gpt6） | ✅已接通 | `src/perception/llm_bridge.py`（`APILLM` 可直连 GPT-6 Astra，加 `gpt6` 分支） |
| **桌面 UI 检测**（55 类控件） | ✅ ScreenParser 真通（M2）｜ YOLOE 可选(404) | `src/perception/desktop_perceive.py` + `detector.py::ScreenParserBackend`（L1 主通道）；`OpenVocabLocator` 代码就位但权重 404 |
| **意图级安全门禁**（delete/send 出范围、payment 确认） | ✅ M5 红队通过 | `src/safety/intent_gate.py`（新增 `IntentGate`，含否定式 + 盲区加固） |
| **桌面 Adapter**（GameAdapter 子类） | ✅ M1+M3 完成 | `src/agent/desktop_adapter.py`（`DesktopAdapter(GameAdapter)`，几何换算已接线） |
| **L2 接入 GPT-6 Astra** | 🟡 代码就绪，待真 key | `build_llm("gpt6")` → `APILLM(gpt-6-astra)`；真实兜底验证待 key |
| **双系统闭环**（perceive→plan→gate→act→verify） | ✅ M6 沙箱全过 | `src/agent/desktop_agent.py`（DesktopAgent 编排 + DesktopBrain(S2) + 经验记忆；每步过 IntentGate 最外层闸） |

## 2. 边界锁定（用户拍板）

1. **高危自动化范围**：`删除/发送` 出自动化范围（永不自动执行，连准备都不做）。
2. **支付**：可准备但**必须人工确认点**才注入（与"永不自动化"相比是放宽——仅支付放开到此步）。
3. **主攻方向**：桌面**非高危重复任务**——填表 / 跨系统搬运 / 文件整理（只移动/重命名，不删除）。
4. **感知主通道**：**YOLOE**（开放词汇 UI grounding）+ 多模态 LLM / 几何 CV 两级降级。
5. **架构**：双系统（S1/S2 + 经验记忆 + 纠错），直接复用，不另起炉灶。

> ⚠️ 微调确认点：原硬约束写"删除/发送/支付 永不自动化"。本次放宽为"支付可准备+人工确认、删除/发送直接出范围"。若本意是支付也彻底出范围，门禁改成三条全拦截。

## 3. 模块布局（路径写实）

复用：
- `src/agent/game_adapter.py` —— `GameAdapter` 基类（`locate/grab/perceive/act/tick`），`Scene` 在 `src/agent/types.py`。
- `src/exec/computer_use.py` —— `ComputerUse`（执行编排）+ `SafetyGuard`（动作级闸）。
- `src/exec/backends.py` —— `SendInputBackend`(原生) / `PyAutoGUIBackend`(兜底)；`build_backend()`。
- `src/exec/window.py` —— `WindowManager`（聚焦/列窗）。
- `src/perception/llm_bridge.py` —— `LLMBridge`/`APILLM`/`build_llm`（L2 后端）。
- `src/perception/weights.py` —— YOLOE 权重加载 + 404 兜底。
- `src/perception/detector.py` —— `Element`（结构化检测输出，含 `.center()/.label/.text/.conf`）。

新增：
- `src/perception/desktop_perceive.py` —— `DesktopScene` + 三级降级 UI 感知管线（L1 ScreenParser / L2 多模态 LLM / L3 CV），每帧带 `active_backend`。
- `src/agent/desktop_adapter.py` —— `DesktopAdapter(GameAdapter)`：`locate`(窗口定位) / `grab`(截帧) / `perceive`(调 `desktop_perceive`) / `act`(调 `ComputerUse`) / `to_screen`(客户区坐标换算)。
- `src/safety/intent_gate.py` —— `IntentGate`：意图分类（normal/delete/send/payment）+ 拦截/确认决策 + 否定式 + 盲区关键词。
- `src/agent/desktop_tasks.py` —— M4 三类任务模板（填表/跨系统搬运/文件整理），每步编译期过 `IntentGate`。
- `src/perception/desktop_perceive_selftest.py` —— M1 沙箱可跑的降级逻辑自检（5 段全过）。
- `src/perception/m2_screenparser_verify.py` —— M2 ScreenParser 真通验证（EXIT=0）。
- `src/perception/m3_exec_acceptance.py` —— M3 沙箱执行工具集验收（6 维全过）。
- `src/perception/m3_realmachine_acceptance.py` —— M3 真机验收（`--real` 真注入门，默认只读安全）。
- `src/agent/desktop_tasks_selftest.py` —— M4 模板自检（11 项全过）。
- `src/safety/intent_gate_redteam.py` —— M5 红队（10 例 + 否定式 3 例，全过；当场修掉 2 个真漏洞）。

## 4. 感知管线（YOLOE 主 + 两级降级，活标注）

`DesktopScene`：
```
@dataclass DesktopScene:
    frame: np.ndarray
    elements: List[UIElement]      # 控件：button/input/menu/icon/link/text
    active_backend: str            # "yoloe" | "llm" | "cv"  ← 强制标注
    confidence: float
    meta: dict
```

三级（顺序降级，命中即停，记 `active_backend`）：
- **L1 ScreenParser（主通道，M2 已真通）**：`detector.py::ScreenParserBackend` 加载 `docling-project/ScreenParser`（YOLO11-Large，55 类 GUI 控件），检测通用控件，`active_backend="screenparser"`。权重经 hf-mirror 已下载落盘。
- **L2 多模态 LLM（GPT-6 Astra 兜底，待真 key）**：把截帧 + goal 送 `APILLM(model="gpt-6-astra")`（或本地 VLM），返回带坐标的 `UIElement`，`active_backend="llm"`。贵，仅关键帧/失败升档用；无 key 时该层不调用。
- **L3 CV 兜底**：按钮矩形 + 文字 OCR、输入框边界、菜单条检测，保证"至少能点"，`active_backend="cv"`。
- **（可选升级）YOLOE 开放词汇**：`OpenVocabLocator` 代码就绪，但 dino 侧权重 404 不可达；权重可达时作为 L1 的开放词汇补充，不可达时绝不谎报 `active_backend`。

`UIElement` 实现 `.center()/.label/.text/.conf`，与 `detector.Element` 接口同构，可直接喂 `LLMBridge.decide` / `format_scene`。

## 5. 执行层（复用，不重造）

`DesktopAdapter.act(action)` → 转成 `ComputerUse` 调用：
- `click(x,y)` / `double_click` / `right_click` / `move`
- `type(text)`（Unicode，支持中文）
- `drag(x1,y1,x2,y2)` / `scroll(clicks)`
- `press(key)` / `hotkey(...)`
- `focus_window(title)` / `list_windows()`

动作级护栏已由 `SafetyGuard` 提供：`require_confirm`、`allowed_region`、`block_dangerous_action`、`block_dangerous_text`(含 `rm -rf`/`del /f`/`format` 等)、`failsafe`(急停文件 `STOP`)。**本期只把它当作"最后一道物理闸"，新增的 `IntentGate` 在其之前做语义拦截。**

## 6. 任务规划（S2 跨应用编排，先三类模板）

`DesktopAdapter` 把截帧 + 目标喂 `LLMBridge.decide(elements, goal)`（S2），产出 `Action{action,target,coordinates,text}`。S2 职责：
- **填表**：识别表单字段 → 填值 → 校验。
- **跨系统搬运**：从 A 应用取结构化数据 → 落到 B 应用。
- **文件整理**：按规则分类归位（只移动/重命名，删除出范围）。
- 每步后看截图确认；失败/卡住唤醒 `ReasonerLLM`/`APILLM` 反思 + `memory.challenge` 纠错（复用 dino 已实证闭环）。

## 7. 安全门禁（意图级，本期新增 `IntentGate`）

`IntentGate.classify(step)` → 意图 ∈ {normal, delete, send, payment}：
- **delete / send** → 直接 `BLOCK`（出自动化范围，连准备都不做，回拒给用户）。
- **payment** → `NEED_CONFIRM`（生成待确认摘要，暂停等真人 checkpoint；确认后才放行给 `ComputerUse`）。
- **normal** → 放行，再过 `SafetyGuard`（region/危险文本/急停）。

三个护栏层次（从外到内）：
1. `IntentGate`（语义层，本期新）
2. `SafetyGuard`（动作层，已存在）
3. 急停文件 `STOP` + 拟人化轨迹（已存在）

## 8. L2 接入 GPT-6 Astra（兜底 grounding / 反思层）

- `build_llm("gpt6")` 别名 → `APILLM(model="gpt-6-astra", base_url=env, api_key=env)`（OpenAI 兼容）。密钥只从 `.env.local`/环境变量读，绝不入库/打印（沿用 `_load_dotenv`）。
- **只做 S2**：关键帧/失败升档才调；逐帧不调（成本 $10/$50 每百万 token、API 延迟、灰度未全开）。
- **节流**：每任务 ≤ N 次 L2 调用；`active_backend` 记 `"llm"`；失败回退 L3。
- **验收**：接真 API key 后跑 1 个"看不懂的杂乱 UI"实例，确认 L2 能补 grounding。代码分支（`build_llm("gpt6")`）已完成，待 key 即联调。

## 9. 任务拆解（带归属/降级/验收）

| # | 任务 | 归属 | 验收门槛 | 状态 |
|---|---|---|---|---|
| M1 | 感知+执行接口骨架（沙箱可跑） | 感知+执行 | 降级逻辑自检通过（L1→L2→L3 链路 + IntentGate 三态） | ✅ `desktop_perceive_selftest.py` 5 段全过 |
| M2 | 桌面 UI 检测真通 | 感知 L1 | 模型加载+推理+产出元素+管线标真后端 | ✅ ScreenParser 真通（标题误写 YOLOE，主通道实为 ScreenParser）；定量召回>90% 待真机截图门 |
| M3 | 执行工具集验收（复用 ComputerUse） | 执行 | 动作齐全/非stub + dry_run 安全 + 几何接线 + 坐标换算 + 中文输入 | ✅ 沙箱 6 维全过；真机脚本 `--real` 门就绪待触发 |
| M4 | 三类任务模板（填表/搬运/整理） | S2 规划 | 每类编译期过 IntentGate（无 delete/send 原语） | ✅ `desktop_tasks.py` + 自检 11 项全过 |
| M5 | 安全门禁红队（IntentGate） | 安全 | 红队 10 例（含 delete/send/payment）+ 否定式全判对 | ✅ 红队通过；当场修掉否定式误拦 + 盲区绕过 2 个真漏洞 |
| L2 | 接入 GPT-6 Astra 兜底层 | S2/L2 | 接 key 后 1 个杂乱 UI 实例补 grounding 成功 | 🟡 `build_llm("gpt6")` 代码就绪，待真实 API key |
| M6 | 双系统闭环接通（PCD loop） | 编排 | 6 用例沙箱全过（填表跑通/删除BLOCK/支付确认/S2反思/开放规划/编译期BLOCK） | ✅ `desktop_agent.py` + 自检 25/25 PASS |

## 10. 里程碑

- **M1**（沙箱）：感知+执行接口骨架 + 降级自检。✅ 已完成。
- **M2**（模型级）：ScreenParser 真通验证（加载+推理+标注）。✅ 已完成；定量召回>90% 待真机截图门。
- **M3**（执行）：执行工具集验收。✅ 沙箱 6 维全过；真机 `--real` 注入门就绪待触发。
- **M4**（S2）：三类任务模板跑通。✅ 已完成（编译期安全）。
- **M5**（安全）：安全门禁红队验收。✅ 已完成（修掉 2 个真漏洞）。
- **L2**（GPT-6 Astra）：接真 key 后兜底验证。🟡 代码就绪，待 key。
- **M6**（双系统闭环）：perceive→plan→gate→act→verify 接通。✅ 沙箱 6 用例/25 项全过。
- 不达标不进下个里程碑（同 GTA5 原则）。已达标里程碑均经沙箱/真机脚本实证，非设计宣称。

## 11. 风险

- ~~YOLOE 权重 404~~ 已解：主通道改为 **ScreenParser**（dino 侧权重可达、已下载落盘、M2 真通）；YOLOE 降为可选升级（权重 404 时绝不谎报）。
- GPT-6 Astra 灰度未全开 → L2 代码已就绪（`build_llm("gpt6")`），但真实兜底验证**待真实 API key**；无 key 时 L2 常驻「代码就绪、不调用」。
- "完美"不真实：OSWorld 2.0 报 72.6%（GPT-6 Astra），长程仍需确认点。
- 跨应用语义靠 LLM，贵且偶错 → 每步截图确认 + 纠错兜底。
- 桌面操作默认关：首次运行需人工开启，`desktop` 模式开关独立于游戏模式（执行层 `dry_run` 默认开，不真注入）。
- **已闭合的真机验收门**（需真桌面显式触发）：M3 真机注入成功率>95%（UIPI/DPI/多屏边界）、窗口几何还原召回率、拟人化轨迹落点误差；M2 定量召回>90%。脚本均就绪，尊重默认关不自动跑。

## 12. 你点名的四块：现状 / 交付物 / 验收证据

### 12.1 桌面 UI 检测（感知主通道）
- **状态**：✅ M2 真通（主通道 = ScreenParser，非 YOLOE）。
- **交付物**：`src/perception/desktop_perceive.py`（`DesktopScene` + 三级降级管线）、`detector.py::ScreenParserBackend`（55 类）、`m2_screenparser_verify.py`。
- **证据**：权重落盘 `blobs/weights/ScreenParser/best.pt`；合成桌面帧推理出 6 元素（Text Input×2 / Button×2 / Text / Search Field）；`perceive_pipeline` 正确标 `active_backend="screenparser"`，EXIT=0。
- **YOLOE 真相关**：`yoloe-11s.pt` 在 dino 侧 404 不可达 → 降为可选升级通道，代码 `OpenVocabLocator` 就位但不谎报后端。
- **待真机**：定量召回 >90% 需真实桌面截图 + ground truth。

### 12.2 意图级安全门禁（IntentGate）
- **状态**：✅ M5 红队通过，且红队当场逼出并修掉 2 个真漏洞。
- **交付物**：`src/safety/intent_gate.py`（`IntentGate` + `GateDecision`）、`m5` 红队脚本。
- **证据**：10 例正样本（delete/send/payment 全判对）+ 3 例否定式（「不发送/非删除」不再误拦）全过。
- **红队修掉的真漏洞**：① 朴素子串把「不发送」误 BLOCK → 加否定式 `_negated()`；② 口语「发给/转发」原漏判 NORMAL → 扩充发/删/付关键词（发给/转发/移除/注销/提交订单/下单 等）。

### 12.3 桌面 Adapter（GameAdapter 子类）
- **状态**：✅ M1 接口骨架 + M3 执行工具集验收。
- **交付物**：`src/agent/desktop_adapter.py`（`DesktopAdapter(GameAdapter)`），含 `locate/grab/perceive/act/to_screen`。
- **证据**：M3 沙箱 6 维全过（10 动作路由到真实后端、dry_run 不真注入、几何工具接线、坐标换算 `to_screen(10,20)→(110,220)`、中文逐字送达拼回 `你好世界`）。
- **接线补强**：M3 补齐 `ComputerUse` 的 `window_rect/client_rect/foreground_window/is_window_minimized` 并注册进 18 工具的工具表；`locate()` 从占位 `待 M3 对齐` 改为真实返回客户区矩形。

### 12.4 L2 接入 GPT-6 Astra（兜底 grounding / 反思）
- **状态**：🟡 代码就绪，待真实 API key。
- **交付物**：`src/perception/llm_bridge.py` 的 `build_llm` 加 `gpt6` 分支 → `APILLM(model="gpt-6-astra")`；`desktop_perceive._try_llm` 已接 `l2_bridge`。
- **密钥纪律**：只从 `.env.local`/环境变量读，绝不入库/打印。
- **待办**：接 key 后跑 1 个「看不懂的杂乱 UI」实例验证 L2 补 grounding；当前无 key 时 L2 不调用、不假装可用（`active_backend` 不标 `llm`）。
- **上下文事实**：GPT-6 Astra 2026-09-03 发布，1.05M token，OSWorld 2.0 = 72.6%，灰度未全开；按 token 计费（$10/$50 每百万），仅关键帧/失败升档调用并节流。

### 12.5 双系统闭环（M6，把零件接成能跑的 Agent）
- **状态**：✅ 沙箱 6 用例 / 25 项断言全过（PASS=25 FAIL=0）。
- **交付物**：`src/agent/desktop_agent.py`（`DesktopAgent` 编排 + `DesktopBrain`(S2) + `DesktopResult`/`DesktopStats`）、`src/agent/desktop_agent_selftest.py`。
- **闭环结构（PCD）**：
  1. **perceive**：`DesktopAdapter.perceive` → 三级降级产出 `Scene`（含 `elements` + `active_backend` 真标注）。
  2. **plan**：S1=模板快路径（`build_task`，不调 LLM，99% 常规任务）；S2=`DesktopBrain.plan`（LLM 开放目标规划，无 LLM 诚实返回 no-plan）。
  3. **gate（最外层）**：每步过 `IntentGate`——delete/send→整任务 BLOCK、payment→标 need_confirm 卡人工确认；过闸才进 `ComputerUse` 动作级物理闸。
  4. **act**：`DesktopAdapter.act` → `ComputerUse`（dry_run 安全、拟人化、审计、急停）。
  5. **verify**：失败→`DesktopBrain.reflect` 纠正→重试；每步写 `ExperienceMemory`。
- **统计（架构真相）**：`s1_calls / s2_calls / s2_reflect / executed / blocked / need_confirm` 如实区分 S1/S2 分工。
- **真 bug 修掉**：`Scene` 原缺 `elements` 字段，`DesktopAdapter.perceive` 一直传 `elements=` → 运行时必崩；已补字段，perceive 现在真能产出带 UI 元素的 Scene。
- **诚实边界**：`AgentOrchestrator`(恐龙专用) 不复用，桌面走独立 `DesktopAgent`；开放目标规划需真 LLM；桌面操作默认关（enable 才接真实桌面）。

### 12.6 M7 桌面自操作小恐龙（用户 2026-09-08：「自己找 Chrome 打开网站去玩」）
- **状态**：✅ 脚本写好（`src/desktop_play_dino.py`），安全验证过（默认只打印计划，`--real` 才真碰桌面）。
- **复用（不重造）**：`DesktopAdapter` 做"找 Chrome 开网站"前置段（Win+R→chrome→Ctrl+L→chrome://dino）；`demo_dino_real.ChromeDinoPilot`（已真机验证：自动定位游戏区 + 发空格/下键跳跃 + 撞死自动重开）接管玩。
- **事实**：本项目早就有能真机自玩小恐龙的 `ChromeDinoPilot`/`demo_dino_agent`，缺的只是"自动开 Chrome 并导航到 dino"这段前置，本次补齐并接到桌面 CU。
- **安全**：默认无 `--real` 只打印 5 步计划；`--real` 才启动 Chrome+导航+接管键鼠；尊重桌面默认关、不自动跑。

### 12.7 M8 微信自主聊（用户 2026-09-08 拍板放开发送红线）
- **状态**：✅ 脚本 + 沙箱自测 12/12 过（全 mock，不碰真桌面）。
- **用户约束（已编成硬代码，不偷偷放宽）**：
  1. 只替 `--contact` 指定的**单一**联系人回；换人须重新跑并确认（检测到换人即停）。
  2. **真人鼠标一动（点击/移动）立即自动取消**（`HumanOverrideMonitor` 轮询 cursor 位置 + 左右键状态，无需全局钩子）。
  3. 删除/支付永远不碰（IntentGate 红线：DELETE 永远 BLOCK、PAYMENT 永远 NEED_CONFIRM）。
  4. 发送仅 `--real` + 预授权范围内才发（门禁标记 `authorized-send` 便于审计）。
  5. 披露默认带「(AI代回)」；`--no-disclosure` 可关（关掉=完全静默冒充，伦理责任归用户，不拦但标注）。
- **红线放松（架构真相）**：`IntentGate` 加 `allow_send` 开关，默认 `False`（SEND 仍 BLOCK）；仅调用方**显式** `True` 且范围受控才 ALLOW。`DELETE`/`PAYMENT` 无论 `allow_send` 如何都保持 BLOCK/NEED_CONFIRM——**物理红线永不放松**。这是用户 2026-09-08 有意识拍板，非默认行为。
- **交付物**：`src/desktop_wechat_chat.py`（`WeChatChatAgent` + `HumanOverrideMonitor` + `FileInbox`/`MockInbox` + `EchoLLM`）+ `desktop_wechat_chat_selftest.py`。
- **待真机（沙箱无法判定，需你真桌面触发）**：① 微信 UI 坐标（搜索框启发式 `l+70,t+45` 需真实窗口微调）；② *读消息已接 OCR 抓屏（见 M9）*，仍依赖真窗口微调；③ LLM 草稿需真实 key（`--echo` 可先占位自测；默认后端已改 `hy3`）。

### 12.8 M9 真机 OCR 读微信消息 + hy3 草稿后端（2026-09-08）
- **状态**：✅ 沙箱自测 13/13 过（`desktop_wechat_chat_selftest_ocr.py`，零真机依赖）；原 12 例红队/安全自测无回归。
- **OCR 读消息（读仅，不发送）**：
  - `detector.py` 的 `OCRBackend` 新增 `read_lines(frame)`：整图 OCR 一次，返回按 y 排序的文本行 `(cx,cy,text,conf)`；复用指纹缓存（屏没变不重复推理），缺 easyocr/读不到图安全降级返回 `[]`。`_load_img`/`_readtext` 抽出复用，`recognize` 不改行为。
  - `desktop_wechat_chat.py` 新增 `OCRInbox`：`grab` 微信窗口 → 按 `chat_frac` 裁聊天区（排除左联系人列/顶栏/底输入框）→ `read_lines` → **去重（已见集合）+ 跳过自带「(AI代回)」前缀的自己消息** → 返回最新一条未读。`ack` 标刚返回那条为已见，保证不丢消息、不重复回。
  - 用法：`python desktop_wechat_chat.py --contact "张三" --real --ocr`（读仅抓屏；配合 `--echo` 可先占位自测不发真 LLM）。
- **hy3 草稿后端**：`build_llm` 加 `hy3`/`hunyuan`/`hunyuan3`/`tencent` 别名 → `APILLM`（OpenAI 兼容），模型名/端点/密钥从 `.env.local` 或环境变量读（如 `MGA_LLM_MODEL`/`MGA_LLM_BASE_URL`/`MGA_LLM_API_KEY`），**绝不写进代码**。无 key 时不崩（仅实例化，调用才报网络错由 `run_once` 容错跳过）。CLI 默认 `--llm-backend hy3`。
- **诚实边界**：微信 UI 布局因版本/缩放而异，`chat_frac` 是启发式（真机需微调）；OCR 较慢（整图 easyocr CPU 数秒~18s，靠缓存缓解）；首轮把屏上可见消息全标已见（不回历史），之后只回真正新来的。

### 12.9 当前进度总表
| 阶段 | 状态 | 交付物 |
|---|---|---|
| M1 接口骨架 | ✅ | `desktop_adapter.py` |
| M2 视觉通道真通 | ✅ | ScreenParser 主通道 |
| M3 执行工具集验收 | ✅ | 18 工具表 + 坐标换算 |
| M4 三类任务模板 | ✅ | `desktop_tasks.py` |
| M5 安全门禁红队 | ✅ | `intent_gate.py` + 红队 |
| M6 双系统闭环 | ✅ | `desktop_agent.py` |
| M7 小恐龙自操作 | ✅(脚本) | `desktop_play_dino.py` |
| M8 微信自主聊 | ✅(沙箱) | `desktop_wechat_chat.py` + 红线放松 |
| M9 OCR 读消息 + hy3 | ✅(沙箱) | `OCRInbox` + `OCRBackend.read_lines` + `build_llm("hy3")` |
| L2 GPT-6 Astra | 🟡 | 待真实 API key |
