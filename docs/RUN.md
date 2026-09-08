# MGA 自跑手册

回答你问的：**后台在哪、怎么跑、怎么训练、demo 怎么用、有没有能点按钮的界面**。

---

## 0. 控制台（推荐：一键操作 + 看数据 + 看报错）

一条命令启动本地面板，浏览器里点按钮就行，**不用记任何命令**：

```bash
cd /d/workbuddy/2026-09-02-17-12-49
PYTHONPATH=src python -m ui_server
# 默认自动打开 http://127.0.0.1:7860 ；--port 8000 换端口，--no-browser 不自动开
```

| 区块 | 能干什么 |
|------|---------|
| **左侧一键操作** | ▶开始采集（时长/间隔/OCR；**默认合规全链路**；⚠「UIA-only」勾选框**仅用于验证降级链路，产出数据不可训练**）、▶开始训练（epochs/max-len/batch 可调）、▶跑一次定位（填目标名） |
| **顶部状态灯** | 绿=运行中（显示已跑时长），灰=空闲；右侧「■ 停止」随时中断 |
| **实时日志** | 子进程 stdout+stderr 合并流式显示；**错误行自动标红**，可勾「只看错误」；红色数字角标=错误条数 |
| **训练数据** | 轨迹条数、唯一响应数、响应多样性（<5% 标红告警）、不同目标数、最近 8 条样本（目标/坐标/响应） |
| **统计** | goal 分布 Top15 条形图——一眼看出数据是不是被 OCR 噪声污染了 |

实现：`src/ui_server.py`（后端，纯标准库 `http.server`，**零第三方依赖**）+ `src/ui/index.html`（前端）。
停止用 `taskkill /F /T /PID` 杀**进程树**——采集会拉起 OCR/YOLO 子进程，只杀父进程会留孤儿吃满 CPU。

> 这个服务由**你自己的终端**持有，不受沙箱后台任务被回收的影响。这才是长期正确的用法。
> 沙箱里我起的那个实例随时可能被回收，失效了就自己跑上面这条命令。

---

## 1. 后台任务"放在哪里、怎么跑"的

本沙箱里，**后台 ≠ Linux daemon**，而是 WorkBuddy 工具自己的后台任务机制：

| 机制 | 说明 |
|------|------|
| **Bash 工具的 `run_in_background=true`** | 启动后立刻返回 task id（如 `FvRVh8`），不阻塞对话。任务完成时自动发 `<task-notification>` 提醒 |
| **采集器 `mode="a"` 追加写** | 任务被沙箱中断杀掉后，**已落盘的数据不会丢**；重拉同一命令自动累积 |
| **`nohup` / `setsid` 在本 Windows Git Bash 不可靠** | 工具一返回就连带回收子进程，日志 0 字节。所以别再用 `nohup &` 了 |
| **被 429 限流影响** | 限流事件下，沙箱可能回收后台任务；但因为数据已追加落盘，重拉即可累积 |
| **孤儿进程排查** | `tasklist \| grep python.exe`；杀进程走专用工具，不在 bash 里直接调 |

实际产出全部落盘：
- **轨迹**：`blobs/trajectories.jsonl`（每行一条，含 prompt/response/坐标/goal）
- **截图**：`blobs/shots/*.png`（真实屏幕截图，时间戳命名）
- **LoRA 适配器**：`blobs/weights/mga-lora/`（peft 标准格式）

---

## 2. 你自己操作的话怎么跑、怎么训练

### 2.1 后台挂 30 分钟采集（你自己用电脑，脚本顺带采真实屏幕）

```bash
cd /d/workbuddy/2026-09-02-17-12-49

PYTHONPATH=src python -m main --real --frames 100000 \
  --llm oracle --max-elements 32 --auto-goal \
  --ocr-scale 0.5 --interval 2 --minutes 31
```

每个参数含义：

| 参数 | 为什么这么设 |
|------|------------|
| `--real` | 接真实 ScreenParser+YOLO（不加就是 Mock 自检） |
| `--llm oracle` | 接地老师：按 goal 在检测结果里找真实坐标，产出训练对齐数据 |
| `--max-elements 32` | 进 prompt 的元素数上限（必须 32，否则训练/推理 prompt 不同源 = 白训） |
| `--auto-goal` | 从屏幕**实际存在**的元素里轮换出题——写死的 goal 切到别的软件就不存在了 |
| `--ocr-scale 0.5` | 1080p OCR 从 ~18s 降到 ~5s；识别率略降但够用 |
| `--interval 2` | 每帧间 sleep 2s；不节流 CPU 被吃满，你电脑卡 |
| `--minutes 31` | 墙钟自停。`--frames 100000` 是兜底（正常到时间停） |

> 用对话里的 Bash 工具启动时把 `run_in_background: true` 打开，就不阻塞你了。完成时会自动通知。

### 2.1b ⚠ UIA-only —— 仅验证降级链路，产出数据**不可训练**

> **2026-09-04 方向性纠偏。** 此前把 `--uia-only` 当「快速采训练数据」是错的，两条硬伤：
>
> 1. **缺 X_visual**：它跳过「截图 → 编码器 → ②投影层」，落盘 `llm_feats_shape=[0]`。
>    而 `data/trajectory_schema.json` 的契约是 `X_visual = llm_feats`，缺这一维训不出多模态大脑，
>    只能训出「文字 → 坐标」的查表。实测对照：合规全链路落 `[16,64]`（64 个非零分量），UIA-only 落 `[0]`。
> 2. **目标本身无意义**：UIA 已能**零误差、实时**给出坐标，再训模型去拟合它，
>    等于造一个更慢、更差、只会背答案的 UIA。此前「110 条 100% 命中 / 0px」是背答案，不是泛化。
>
> **UIA 的正确角色**（2026-09-02 定的集成姿势）：三级降级 `YOLO → OCR → UIA` 里的**兜底定位器**，
> 把结构化 bbox 直接注入 LLM 上下文去选，**不是训练数据的标注源**。

**要训练数据就用 §2.1 合规全链路**：实测 **≈7.2s/帧**（`--ocr-scale 0.5` + OCR 指纹缓存），
半小时约 250 条，落盘 `X_visual=[16,64]` 可直接喂③ MiniMind。

只有当你想验证「UIA 降级链路还通不通」时才用下面这条（**采完请丢弃数据，勿喂训练**）：

```bash
cd /d/workbuddy/2026-09-02-17-12-49

PYTHONPATH=src python -m main --real --uia-only \
  --llm oracle --max-elements 32 --auto-goal \
  --collect-every 1 --minutes 5
```

| 参数 | 说明 |
|------|------|
| `--uia-only` | 跳过 ScreenParser/OCR/截图/编码器/投影层；不加 `--ocr-scale`/`--interval` |
| ⚠ 落盘数据 | `llm_feats_shape=[0]`，**禁止喂给训练** |

**已知粒度问题**：复合 `TextControl`（如抖音顶部导航栏把整行文字当一个控件 Name）会让 `oracle`
按子串把短 goal（"小游戏"）命中到整行控件、坐标落在行中心。标准独立按钮/输入框不受影响。

### 2.2 看采集数据质量

```bash
# 行数（=轨迹条数）
wc -l blobs/trajectories.jsonl

# 多样性（重点看 unique_responses / unique_goals；多样性 < 5% 说明屏幕是静止的）
PYTHONPATH=src python -m training.sft_dataset

# prompt token 长度分布（若全超训练 max_len，提示窗口要调）
PYTHONPATH=src python -m training.train_minimind --report-lengths
```

### 2.3 训练（全量 LoRA 微调）

```bash
PYTHONPATH=src python -m training.train_minimind \
  --max-len 1728 --epochs 3 --batch 2 --lr 2e-4
```

| 参数 | 为什么 |
|------|--------|
| `--max-len 1728` | prompt 真实长度 ~1200-1545 token；默认 512 会 100% 截断 |
| `--epochs 3` | 151 样本 ×3 epoch ≈ 453 步，CPU 上 ~13 分钟；GPU 几分钟 |
| `--lora`（默认开） | 只训 0.82% 参数（212,992 个），快且不毁基座 |

跑完适配器自动存到 `blobs/weights/mga-lora/`。

---

## 3. Demo（看效果用）

刚写好的 `src/demo_locator.py`，一条命令就能验证模型有没有真学会：

### 3.1 先看屏幕上有哪些候选目标（避免 OCR 误识的噪声串）

```bash
PYTHONPATH=src python -m demo_locator --list
```

打印屏幕实际存在的、**可作目标**的带文字元素（自动过滤纯 ASCII 短串等 OCR 噪声）。

### 3.2 跑指定目标，最干净的 demo

```bash
# 自动从屏幕挑（要求是真实存在的元素，不能是 'on'/'STS' 这种 OCR 噪声）
PYTHONPATH=src python -m demo_locator

# 自己指定目标
PYTHONPATH=src python -m demo_locator --goal '搜索'

# 对已有截图跑（不重新截屏，省 20s）
PYTHONPATH=src python -m demo_locator --shot blobs/shots/xxx.png --goal '搜索'
```

### 3.3 怎么看输出

demo 做了一件事：**同一张图三色对比**。

| 颜色 | 含义 |
|------|------|
| 🔴 红圈 | 训练后（+LoRA）模型预测点 |
| 🔵 蓝圈 | 训练前（base）模型预测点——通常是空的 |
| 🟢 绿圈 + 绿框 | 真实元素中心 + bbox（标准答案） |

三者重合 + 信息面板写"误差 0px 命中=是" → 模型真的学会了定位。
红圈明显偏离绿圈 → 模型还在背坐标或泛化失败。

---

## 4. 我怀疑你在架空我？——怎么自查

后台跑完你最不踏实的就是"我是不是嘴上说跑了实际没跑"。自查姿势：

1. **产物文件存在性**：`blobs/weights/mga-lora/` 6 个文件（adapter_model.safetensors + config + tokenizer...），`blobs/trajectories.jsonl` N 行
2. **截图真实**：随便打开 `blobs/shots/*.png` 任意一张，是你自己屏幕的快照
3. **时间戳跨度**：第 1 帧 ts ≈ 开始时间，最后一帧 ts ≈ 结束时间，跨度约 = 你设置的 minutes
4. **adapter_config.json**：`r=8/alpha=16/target_modules=[q_proj,k_proj,v_proj,o_proj]` 精确等于训练代码参数——这是我编不出来对应不上的
5. **独立统计**：用 Grep 数 `"response":` 在 jsonl 出现次数 = `wc -l` 的行数

---

## 5. 已知坑与下一步

- **UIA 角色（2026-09-04 纠偏）**：`UIALocator` 从系统无障碍树拿真实控件文字（零依赖、零误差），在采集链路主动融合降噪（A/B 验证 30%→7%）——**这部分是对的，保留**。
  但 **`--uia-only` 不能当训练数据采集模式**：它跳过编码器/投影层，落盘 `X_visual=[0]`，违反 schema 契约；
  且 UIA 本身零误差实时给坐标，训模型拟合它 = 造更慢更差、只会背答案的 UIA。
  **正确定位：UIA 是三级降级 `YOLO → OCR → UIA` 的兜底定位器，不是标注源。** 详见 §2.1b
- **miniMind2-Small 26M 太小**：复杂 UI 上可能仍崩；接 VLM（UltralyticsLLM/真 VLM）才是终局
- **demo 暴露的真问题**：loss 低 ≠ 泛化好；要用 `--list` + `--goal` 手动挑干净目标验证；当前 demo 已能验证

## 6. 小恐龙 System1 Demo（核心创新 · 端到端验证）

```bash
cd /d/workbuddy/2026-09-02-17-12-49
PYTHONPATH=src python -m demo_dino
```

纯数学预判帧端到端验证（**注意：这是核心创新演示，不是训练数据源**）：System1 用
`Track`/RLS 残差算障碍物到达恐龙的剩余时间 ETA，仅接近恐龙时唤醒 System2 选 `jump`/`squat`，
System2 纠正样本回流训练 System1（S2→S1 蒸馏）。输出 System1 静默 / System2 关键帧比例 + Token 经济。
回到原始设计正路：90% 帧从 NN 降级为纯数学，token 经济逼着 agent 优先用物理预判。

---

## 7. 真实 Chrome 小恐龙 Pilot（端到端 · 真屏 CV + System1 + 真按键）

> 这是**能录屏给别人看的第一个端到端演示**：真实桌面上的 Chrome 小恐龙被系统自动控制，
> System1 纯数学预判（无神经网络介入）的实测延迟可录屏展示。
> 链路：截图(游戏区) → 单色 CV 检测恐龙/障碍 → System1 `Track`/RLS 算 ETA → 临门一脚唤醒
> System2 选 `jump`(空格)/`squat`(下键) → `pyautogui` 真按键 → 闭环回流(S2→S1 蒸馏)。

### 7.1 先跑 sim 自测（沙箱可跑，验证 CV+System1+闭环逻辑）

```bash
cd /d/workbuddy/2026-09-02-17-12-49
PYTHONPATH=src python -m demo_dino_real --sim
```

输出要点（实测）：

```
System1静默=384(96.0%)  System2关键帧=16(4.0%)
触发决策=16  成功躲过=16  漏判撞车=0
[System1 纯数学预判延迟] avg=30.50µs  p50=30.70µs  p99=51.90µs  max=66.70µs  (零神经网络介入)
```

- 96% 帧 System1 静默（纯物理运算，**零神经网络**）；4% 关键帧才唤醒 System2（贵调用）
- **System1 纯数学预判延迟 ≈30µs**（微秒级），这是「无 NN 介入」的硬证据，可截屏/录屏展示
- 多 seed 稳定：16 触发 / 16 躲过 / 0 漏判（seed=1,2,3,7 全一致）

### 7.2 真机运行（你自己的电脑，开 Chrome 小恐龙）

**前置**：
1. 浏览器打开 `chrome://dino`（断网或地址栏输 `chrome://dino` 回车即开始），让小恐龙站在左侧
2. 装 `pyautogui`：`pip install pyautogui`（仅真机按键需要；sim 不依赖）
3. 运行前**点一下 Chrome 窗口**让它获得焦点（`pyautogui.press` 需要窗口在前台）

**游戏区坐标不用你量**：真机不传 `--region` 时，端侧自动量——全屏截图里找
`chrome://dino` 那块浅灰画布 `#f7f7f7`（页面其余是纯白 `#fff`，阈值干净区分），
bounding box 即 region。只有自动量失败时才用手动 `--region` 覆盖。

```bash
cd /d/workbuddy/2026-09-02-17-12-49

# 真机：自动量 region + 自动控制（开录屏跑这条即可）
PYTHONPATH=src python -m demo_dino_real --frames 1200

# 手动覆盖（仅自动量失败时用）：PYTHONPATH=src python -m demo_dino_real --region 360,240,600,150
```

| 参数 | 说明 |
|------|------|
| `--region X,Y,W,H` | 游戏画布**屏幕绝对像素**（可选覆盖；不传则端侧自动量浅灰 canvas） |
| `--frames` | 帧数（默认 400；想长录屏就 `--frames 1200`） |
| `--dt` | 每帧间隔秒（默认 0.05；真机按你机器截图速度定，慢就调 0.08） |
| `--sim` | 合成截图自测，不碰真实桌面 |

**真机预期结果**：小恐龙被自动控制连续跳跃/下蹲；终端实时打印 `KEYFRAME(S2) eta=...→jump`；
结束时给出 System1 延迟（avg/p50/p99/max µs）+ Token 经济；`漏判撞车=0` 即全程没撞。

**真机会翻车的点（先说清楚）**：
- 游戏加速后速度变快，`Track` 每障碍重新校准速度，能自适应；但若帧率太慢（dt 太大）会漏判
- 撞一次就 Game Over，detect 找不到恐龙 → 自动停在当前帧；重开小恐龙重跑即可
- 区域量偏了（截到白边/黑边）会让 `detect` 把边框当障碍；量紧一点，只框游戏画布
- 恐龙下蹲时精灵变矮，仍是 `#535353`，检测不受影响；鸟在上方，检测判 `is_bird→squat`

### 7.3 这套验证了什么（对应原始架构）

| 原始设计点 | 本 demo 落点 |
|-----------|-------------|
| ① 感知(检测器) | 单色 CV 阈值检测，`mask = (Σ|像素−#535353| < 60)`，纯 numpy+PIL，零 cv2 |
| ② 投影层/③ 大脑 | **故意不接**（这是验证 System1 的 demo，不是训练数据源） |
| 预判帧 System1（核心创新） | `Track`+`RLS` 残差纯数学算 ETA，**90%+ 帧静默，零 NN**，延迟 µs 级 |
| token 经济 | `TokenEconomy`：静默 cost≈0.1、关键帧 cost=100，逼 agent 优先物理预判 |
| S2→S1 蒸馏 | 每帧 `sys1.update(..., sys2_correction=eta−reaction_time)` 回流残差 |
| ④ 执行 | `pyautogui.press("space"/"down")` 真按键（sim 走合成闭环） |

源码：`src/demo_dino_real.py`（`detect`/`synthetic_shot`/`ChromeDinoPilot`）+ `src/demo_dino.py`（`DinoSystem1`）+ `src/predictive/frame.py`（`Track`/`RLS`/`CalibrationWindow`）。
