#!/usr/bin/env bash
# setup.sh —— MGA 真实模式依赖安装（幂等，可反复执行）
# ============================================================================
# 用途：装上跑「真实模式」所需的依赖，然后就可以
#           python src/main.py --real
#       采集真实轨迹（喂给 ③ MiniMind 训练）。
#
# 设计原则（与全系统一致：零依赖自检 + 满血运行）：
#   - 幂等：已装的包 pip 会跳过，可放心重复执行。
#   - 默认 CPU 版 torch：CUDA 版 wheel 动辄 2GB+，本机无独显时用 CPU 索引省时省盘。
#   - 装完逐项 import 自检：缺哪个一眼看到，而不是等跑起来才崩。
#   - 不装也能跑：主程序所有重依赖都是懒加载，缺依赖会降级而非崩溃。
#
# 用法：
#   bash setup.sh                 # 默认 CPU 版 torch
#   bash setup.sh --gpu           # 有 NVIDIA 独显，装 CUDA 版 torch
#   bash setup.sh --venv .venv    # 装进虚拟环境（推荐，避免污染全局）
#   bash setup.sh --check         # 只检查当前缺什么，不安装
#   bash setup.sh --minimal       # 只装真实模式最小集（截图+检测），不含 transformers
#
# Windows 说明：在 Git Bash 里执行本脚本即可。若用 PowerShell，请先手动创建并
#   激活 venv：  python -m venv .venv  &&  .venv\Scripts\Activate.ps1
#   再执行：      bash setup.sh
# 注意：pyautogui 需要图形界面（远程桌面/无头会话会失败，此时主程序会走合成截图兜底）。
# ============================================================================
set -euo pipefail

# ---- 默认参数 ----
TORCH_INDEX="https://download.pytorch.org/whl/cpu"   # 默认 CPU 版（体积小）
TORCH_LABEL="CPU"
VENV_DIR=""
CHECK_ONLY=0
MINIMAL=0
PY="python"

# ---- 参数解析 ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)     TORCH_INDEX=""; TORCH_LABEL="CUDA(默认索引)"; shift ;;
    --venv)    VENV_DIR="${2:?--venv 需要给出目录名}"; shift 2 ;;
    --check)   CHECK_ONLY=1; shift ;;
    --minimal) MINIMAL=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（用 --help 查看用法）"; exit 1 ;;
  esac
done

echo "=== MGA 依赖安装 ==="
echo "torch 版本 : ${TORCH_LABEL}"
echo "安装模式   : $([[ $MINIMAL -eq 1 ]] && echo 最小集 || echo 完整) / $([[ $CHECK_ONLY -eq 1 ]] && echo 仅检查 || echo 安装)"

# ---- 虚拟环境 ----
if [[ -n "$VENV_DIR" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "--- 创建虚拟环境 $VENV_DIR ---"
    "$PY" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  if [[ -f "$VENV_DIR/Scripts/activate" ]]; then   # Windows
    source "$VENV_DIR/Scripts/activate"
  else                                            # Linux/macOS
    source "$VENV_DIR/bin/activate"
  fi
  echo "已激活虚拟环境: $VENV_DIR"
fi

PY="$PY"
echo "使用解释器: $(command -v "$PY" 2>/dev/null || echo "$PY")"
"$PY" --version

# ---- 需要检查的模块（模块名 -> 用途）----
# 说明：pyautogui 截图、ultralytics 检测、PIL 存图、numpy 全流程、
#      easyocr 给 UI 元素补文字（否则 ScreenParser 只框不识字，goal 无法落地）、
#      transformers/accelerate 留给 mage-vl 主干与后续 MiniMind。
CORE_MODULES=(numpy PIL pyautogui ultralytics easyocr)
EXTRA_MODULES=(torch transformers accelerate)
ALL_MODULES=("${CORE_MODULES[@]}" "${EXTRA_MODULES[@]}")
if [[ $MINIMAL -eq 1 ]]; then
  ALL_MODULES=("${CORE_MODULES[@]}" torch)
fi

# ---- 依赖检查函数 ----
check_deps() {
  "$PY" - "$@" <<'PYEOF'
import importlib, sys
missing = []
for m in sys.argv[1:]:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK]      {m:<14} {ver}")
    except Exception as e:
        print(f"  [MISSING] {m:<14} ({type(e).__name__})")
        missing.append(m)
sys.exit(1 if missing else 0)
PYEOF
}

echo ""
echo "=== 依赖检查 ==="
if check_deps "${ALL_MODULES[@]}"; then
  echo "✅ 所有依赖已就绪，无需安装。"
else
  if [[ $CHECK_ONLY -eq 1 ]]; then
    echo ""
    echo "上面标 [MISSING] 的就是缺的。执行 bash setup.sh 安装（加 --gpu 可装 CUDA 版 torch）。"
    exit 1
  fi
  echo "有缺失，开始安装..."
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
  echo ""
  echo "（--check 模式，不安装）"
  exit 0
fi

# ---- 安装 ----
echo ""
echo "=== 升级 pip ==="
"$PY" -m pip install --upgrade pip

echo ""
echo "=== 安装基础依赖（numpy / Pillow / pyautogui）==="
# Pillow 必须装：截图要靠它落盘成 png（否则轨迹里存的是内存地址，无法回查原图）
"$PY" -m pip install numpy Pillow pyautogui

echo ""
echo "=== 安装 torch (${TORCH_LABEL}) ==="
if [[ -n "$TORCH_INDEX" ]]; then
  "$PY" -m pip install torch --index-url "$TORCH_INDEX"
else
  "$PY" -m pip install torch
fi

echo ""
echo "=== 安装检测器 ultralytics（① 感知层真实骨干）==="
"$PY" -m pip install ultralytics

echo ""
echo "=== 安装 easyocr（① 感知层补文字：ScreenParser 只框不识字，OCR 让 goal 真正可执行）==="
"$PY" -m pip install easyocr

if [[ $MINIMAL -eq 0 ]]; then
  echo ""
  echo "=== 安装 transformers / accelerate（mage-vl 主干 + 后续 ③ MiniMind 训练）==="
  "$PY" -m pip install transformers accelerate
fi

# ---- 装后自检 ----
echo ""
echo "=== 安装后自检 ==="
if check_deps "${ALL_MODULES[@]}"; then
  echo ""
  echo "✅ 依赖安装完成。"
else
  echo ""
  echo "⚠️  仍有模块导入失败（见上方 [MISSING]）。"
  echo "   常见原因：pyautogui 在无图形界面的会话里装不上/导不进。"
  echo "   好消息：主程序对缺依赖是**降级**而非崩溃——无 pyautogui 会走合成截图兜底，"
  echo "   无 ultralytics 会降级为像素特征。但那样采到的是模拟数据，不能用于训练。"
  exit 1
fi

# ---- 下一步提示 ----
echo ""
echo "=== 下一步 ==="
cat <<'NEXT'
1) 先确认链路通（不进主循环，纯自检）：
     PYTHONPATH=src python src/main.py --verify

2) 采真实轨迹（默认 auto_execute=false，只规划不真点，安全）：
     PYTHONPATH=src python src/main.py --real

   config.json 的 runtime 段控制行为：
     use_real        true  = 真实截图+真实检测（本脚本装的就是为它服务）
     auto_execute    false = 只规划；确认动作靠谱后再改 true 真点
     visual_backend  screenparser / yolo / mage-vl
     llm_backend     mock / ollama / openai / ultralytics

3) 校验采到的数据是否符合 v1 契约（data/trajectory_schema.json）：
     PYTHONPATH=src python -c "import sys;sys.path.insert(0,'src');from trajectory import validate;validate('blobs/trajectories.jsonl')"

   采够 >=500 条真实轨迹后再做 ③ MiniMind LoRA 自训练。
NEXT
