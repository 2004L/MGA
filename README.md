# MGA — Multi-modal GUI Agent（ComputerUse 双系统）

> 分层决策智能体 · 感知（System2 视觉大脑）+ 预判（System1 纯数学预判帧）+ 执行 三件套的桌面 ComputerUse 系统。
> 核心创新：**预判帧 System1**——90%+ 的帧用纯数学 ETA 预判（零神经网络介入，µs 级延迟），只有临门一脚才唤醒昂贵的 System2（LLM），并用 Token 经济逼着 agent 优先物理预判。

## 架构（双系统）

| 层 | 模块 | 说明 |
|----|------|------|
| ① 感知 | `src/perception/` | 检测器 `detector.py`：三级降级 `YOLO → OCR → UIA`，UIA 用系统无障碍树做零误差兜底定位器；后续集成 SAM3 做像素级分割 |
| ② 投影层 + ③ 大脑 | `src/agent/`, `src/training/` | LLM/VLM grounding；LoRA 微调（miniMind 基座，只训 0.82% 参数） |
| 预判帧 System1（核心创新） | `src/predictive/frame.py` | `Track`/`RLS` 残差纯数学算障碍 ETA；`TokenEconomy` 静默 cost≈0.1 / 关键帧 cost=100 |
| ④ 执行 | `src/exec/computer_use.py`, `src/exec/gui_agent.py` | `pyautogui` 真按键 / 桌面操作；高危操作（删除/发送/支付）**永不自动化** |
| 闭环回流 | `src/agent/system1.py` | S2→S1 蒸馏：System2 纠正样本回流训练 System1 残差 |

源码入口：`src/main.py`（采集）、`src/ui_server.py`（控制台）、各 `demo_*` 模块。

## 目录结构

```
MGA/
├── src/                  # 全部源码（agent / exec / perception / economy / knowledge / predictive / training / ui ...）
├── config.json           # 运行配置（密钥走环境变量，见下）
├── setup.sh              # 环境准备
├── docs/                 # 设计/实施方案/讲稿/自跑手册/能力审计
│   ├── 分层决策智能体-双系统架构设计.md
│   ├── 桌面ComputerUse能力实施方案.md
│   ├── 桌面ComputerUse能力_演示讲稿.md
│   ├── RUN.md            # 自跑手册（怎么跑/训练/demo/自查）
│   ├── overview.md       # 训练闭环验证报告
│   ├── CAPABILITY_AUDIT.md
│   └── 录屏操作清单.md
├── screenshots/          # 精选演示截图（约 25 张，入库）
└── screenshots-full/     # 完整截图归档（743 张，本地保留，不入库，见 .gitignore）
```

## 快速开始

```bash
# 控制台（浏览器点按钮，零命令）
cd MGA
PYTHONPATH=src python -m ui_server        # http://127.0.0.1:7860

# 采集训练数据（合规全链路，≈7.2s/帧，落盘 X_visual=[16,64]）
PYTHONPATH=src python -m main --real --frames 100000 \
  --llm oracle --max-elements 32 --auto-goal --ocr-scale 0.5 --interval 2 --minutes 31

# 训练 LoRA
PYTHONPATH=src python -m training.train_minimind --max-len 1728 --epochs 3 --batch 2 --lr 2e-4

# 小恐龙 System1 预判帧 demo（端到端，真屏 CV + 真按键）
PYTHONPATH=src python -m demo_dino_real --sim        # 合成自测
PYTHONPATH=src python -m demo_dino_real --frames 1200  # 真机（需 pyautogui + chrome://dino 前台）
```

> ⚠️ UIA-only 模式（`--uia-only`）**仅用于验证降级链路**，落盘 `X_visual=[0]`，产出数据禁止喂训练。

## 安全约束

- 高危操作（删除 / 发送 / 支付）**永不自动化**；桌面操作默认关闭，需显式开启。
- `config.json` 不含密钥明文，LLM API Key 走环境变量（`runtime.llm_api_key_env`）。
- `.gitignore` 已排除 `.env.local`、`*.db`、`__pycache__`、`blobs/`、`screenshots-full/`。

## 权重与截图说明

- **LoRA 权重 `mga-lora` 不在本仓库**（约 900KB，按需从发布/训练产物获取，放 `blobs/weights/mga-lora/`）。
- 演示截图：仓库内 `screenshots/` 为精选约 25 张；完整 743 张录制帧在 `screenshots-full/`（本地保留，未入库，避免仓库膨胀）。
