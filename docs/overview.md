# MGA 训练闭环验证报告

> ⚠️ **2026-09-04 纠偏声明 —— 本报告核心结论已降级，请先读这段**
>
> 本报告验证的「GPU 训练 + 100% 命中」闭环，**数据链路是错的**：
> - 训练数据走的是 UIA 融合 / `--uia-only`，落盘 `llm_feats_shape=[0]`（**X_visual 缺失**），
>   违反 `data/trajectory_schema.json` 的契约 `X_visual = llm_feats`
>   → 训出来的不是多模态大脑，只是「文字 → 坐标」查表。
> - UIA 本身已能零误差、实时给出坐标，再训模型去拟合它 = 造一个更慢、更差、只会背答案的 UIA。
>   **「110 条 100% 命中 / 0px 误差」是背答案，不是泛化能力。**
>
> **仍然有效的部分**：① `UIALocator` 降噪（OCR 噪声 30%→7%）✓；
> ② GPU 环境修复（torch cu132，提速 24×）✓；③ 控制台 ✓。
> **作废的部分**：拿 UIA 数据做 grounding 自训这条路线。
> 合规训练采集走全链路（实测 ≈7.2s/帧，落盘 `X_visual=[16,64]`），见 `RUN.md` §2.1 / §2.1b。

## 核心结论
- 发现用户机器有 **NVIDIA GeForce RTX 3060 Laptop GPU**（CUDA 13.2），但原环境 `torch==2.14.0+cpu` 导致 GPU 空转。
- 成功切到 **CUDA 版 torch（2.14.0+cu132）**。
- 用 110 条干净 UIA 真值数据在 GPU 上完成 3 epoch LoRA 训练，**训练耗时仅 33 秒**；对比此前 CPU 版的 13 分 31 秒，**提速约 24 倍**。
- **训练集回放验证**：110 条中 99 条有效预测，**100% 命中（<50px），平均误差 0.0px**。
- **实时 Demo 验证**：截取当前屏幕，目标「朋友」，训练后预测 (240,290) 与真值中心 (240,290) 完全重合，**误差 0px**。

## 关键改动
| 文件 | 改动 |
|------|------|
| `src/perception/detector.py` | 修复 `UIALocator`（递归遍历、过滤离屏控件），新增 UIA 与视觉检测融合，用系统无障碍树真值覆盖 OCR 噪声文字。 |
| `src/main.py` | 采集模式强制使用 `MockExtractor`，避免外部 LLM API 失败刷屏。 |
| `src/training/train_minimind.py` | 自动选择 `cuda`/`cpu`，模型/数据 `.to(device)`，`sample_gen` 跟随模型设备。 |
| `src/demo_locator.py` | 推理加载 GPU 模型；`pick_auto_goal` 去歧义（优先屏上唯一名目标，避免同名多元素导致虚假误差）。 |
| `src/ui_server.py` + `src/ui/index.html` | 控制台「开始采集」按钮补 `--collect-every 1`，新增 Demo 结果图展示。 |

## 训练与验证结果
- **数据**：`blobs/trajectories.jsonl` 110 条，0% OCR 噪声，66.4% 响应多样性，68 个不同目标。
- **Loss**：1.4088 → 0.0738 → 0.0229
- **LoRA 适配器**：`blobs/weights/mga-lora`
- **训练集回放**：100% 命中，0px 误差
- **Demo 验证图**：`blobs/demo_1788425822.png`（红圈=训练后预测，绿圈/绿框=真值，二者重合）

## 踩坑与解决方案
1. **pip 在线安装失败**：`torch==2.14.0+cu132` 下载到 1.8/2.0 GB 时网络中断 5 次，最终因临时文件锁 `WinError 32` 失败。  
   **解决**：用 `curl -C -` 断点续传把 whl 拉取到本地，再用 `pip install --no-deps --force-reinstall 本地whl` 成功安装。
2. **demo 首轮 900px 误差**：目标「精选」在屏幕上出现多次（顶部标签和底部导航），真值与模型分别选中不同实例。  
   **解决**：`pick_auto_goal` 增加去歧义逻辑，只选屏上唯一名的目标；重跑后目标「朋友」0px 命中。

## 后续可选
- ④ VLM 级联：用视觉大模型补 ScreenParser 漏检的小控件，喂给同一套 LoRA。
- ③ temporal context：让模型看「上一帧点了哪、发生了什么」，支持多步任务。

## 正路验证：预判帧 System1 小恐龙 Demo（2026-09-04）

纠偏后回到核心创新（见顶部声明）。新增 `src/demo_dino.py`，用纯数学 ETA 预判跑通端到端闭环：
- System1（`DinoSystem1` + `Track`/RLS 残差）纯数学跟踪障碍 X，预测到达恐龙的剩余时间 ETA，**零神经网络**。
- 仅当 `ETA < 反应时间(0.45s)` 或 `距离 < 阈值(70px)` 才唤醒 System2（`system2_decide` 选 jump/squat）。
- System2 纠正样本回流训练 System1 残差（S2→S1 蒸馏，使「习惯」可习得）。
- Token 经济联动：静默帧 cost 0.1，关键帧 cost 100，体现「能靠物理公式解决的绝不动用大模型」。

实测（300 帧）：System1 静默 291/300（**97.0%**），System2 关键帧 9/300（3.0%），**0 漏判撞车**，
校准速度 vx=-319.4（真值 -320.0），eff=0.32。运行：`PYTHONPATH=src python -m demo_dino`。
（关键帧比例受单障碍串行 Demo 限制；机制已证——System1 独立承担 97% 帧，仅障碍接近恐龙时才唤醒 System2。）

### 真实 Chrome 小恐龙 Pilot（端到端 · 2026-09-04）

新增 `src/demo_dino_real.py`：**真屏 CV 检测 + System1 纯数学预判 + `pyautogui` 真按键**，
这是能录屏给别人看的第一个端到端演示。`detect` 按 `#535353 on #f7f7f7` 单色阈值工作（纯 numpy+PIL），
`--sim` 合成截图自测、`--region X,Y,W,H` 接真实游戏区。

sim 自测实测（400 帧，seed=1/2/3/7 一致）：**16 触发 / 16 躲过 / 0 漏判**，System1 静默 96% /
System2 关键帧 4%，**System1 纯数学预判延迟 avg=30.5µs（p50=30.7 / p99=51.9 / max=66.7 µs，零神经网络）**。
真机运行：`PYTHONPATH=src python -m demo_dino_real --region X,Y,W,H`（需 `pip install pyautogui` + 窗口在前台）。
详见 RUN.md §7。

