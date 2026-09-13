#!/usr/bin/env bash
# setup_perception.sh —— MGA 感知层（A supervision 事件 / B 事件闸 / C Frigate 采集 / 真 LLM 桥）依赖固化安装
# ============================================================================
# 设计原则（与全系统一致：真实可复现 + 零依赖自检 + 满血运行）：
#   - 幂等：.venv 已存在就跳过创建；pip install 对已装包自动跳过。可反复执行。
#   - 锁定版本：从 requirements.lock.txt 装精确版本，任何人/任何机器装出来一致。
#   - C 盘满兜底：本机 C 盘曾仅剩 59M，pip 临时目录与 venv 都放 D 盘（repo 在 D 盘）。
#   - 建 venv 优先用 Windows `py` 启动器：规避某些 Git-Bash 下「托管 python 建 venv 静默失败」的坑。
#   - 装后逐项 import 自检：缺哪个一眼看到，而不是等跑起来才崩。
#   - 不装也能跑：感知层所有重依赖都是懒加载 + 降级；但本脚本装的是真实链路所需。
#
# 用法（在 MGA 仓库根目录执行）：
#   bash setup_perception.sh                  # 建 .venv 并装锁定依赖
#   bash setup_perception.sh --check          # 只检查当前缺什么，不安装
#   VENV_DIR=myenv bash setup_perception.sh   # 装到自定义目录（默认 .venv）
#
# Windows 说明：Git Bash 里直接 bash setup_perception.sh；PowerShell 则跑 setup_perception.ps1。
# 注意：脚本会创建 .venv 在仓库内（已被 .gitignore 忽略，绝不入库）。
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
VENV="$VENV_DIR"
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（用 --help 查看用法）"; exit 1 ;;
  esac
done

# ---- C 盘满兜底：pip 临时目录挪到 D:/tmp（repo 在 D 盘，有空间）----
PIP_TMP="${TMPDIR:-D:/tmp}"
mkdir -p "$PIP_TMP"
export TMPDIR="$PIP_TMP" TEMP="$PIP_TMP" TMP="$PIP_TMP"

# ---- 选「建 venv」用的解释器：优先 Windows py 启动器 ----
CREATOR="python"
if command -v py >/dev/null 2>&1; then
  if py -3.12 --version >/dev/null 2>&1; then CREATOR="py -3.12"
  elif py -3 --version >/dev/null 2>&1; then CREATOR="py -3"; fi
fi

echo "=== MGA 感知层依赖安装 ==="
echo "仓库根目录 : $ROOT"
echo "虚拟环境   : $VENV"
echo "pip TEMP   : $PIP_TMP (C 盘满兜底)"
echo "venv 创建器: $CREATOR"

# ---- 虚拟环境 ----
if [[ $CHECK_ONLY -eq 0 ]]; then
  if [[ ! -d "$VENV" ]]; then
    echo "--- 创建虚拟环境 $VENV ---"
    $CREATOR -m venv "$VENV"
  fi
  # 断言 venv 真正建出来了（某些 python 会静默建空目录）
  if [[ ! -f "$VENV/Scripts/activate" && ! -f "$VENV/bin/activate" ]]; then
    echo "⚠️ venv 创建失败（未发现 activate）。请改用系统 python 3.12 手动建：" >&2
    echo "    py -3.12 -m venv .venv" >&2
    exit 1
  fi
fi

# ---- 解析 venv 内的解释器 ----
VENV_PY="$VENV/Scripts/python.exe"
if [[ ! -f "$VENV_PY" ]]; then VENV_PY="$VENV/bin/python"; fi

if [[ $CHECK_ONLY -eq 1 ]]; then
  echo ""
  echo "=== 仅检查（不安装）==="
  if [[ ! -f "$VENV_PY" ]]; then
    echo "⚠️ .venv 不存在，先执行 bash setup_perception.sh 创建。"
    exit 1
  fi
  selfcheck() { :; }
  "$VENV_PY" - <<'PYEOF'
import importlib
core = ["supervision","numpy","cv2","paho.mqtt","requests"]
opt  = ["amqtt"]
missing = []
for m in core:
    try:
        mod = importlib.import_module(m)
        print(f"  [OK]      {m:<12} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  [MISSING] {m:<12} ({type(e).__name__})")
        missing.append(m)
for m in opt:
    try:
        mod = importlib.import_module(m)
        print(f"  [OK](opt) {m:<12} {getattr(mod,'__version__','?')}")
    except Exception:
        print(f"  [skip]    {m:<12} 未装（仅真 broker 联调需要）")
if missing:
    print("\n上面 [MISSING] 即缺失项。执行 bash setup_perception.sh 安装。")
    raise SystemExit(1)
print("\n✅ 感知层依赖就绪")
PYEOF
  exit 0
fi

echo "使用解释器: $($VENV_PY --version 2>&1)"

# ---- 安装 ----
echo ""
echo "=== 升级 pip ==="
"$VENV_PY" -m pip install --upgrade pip

echo ""
echo "=== 安装锁定依赖 (requirements.lock.txt) ==="
"$VENV_PY" -m pip install -r "$ROOT/requirements.lock.txt"

echo ""
echo "=== 安装后自检 ==="
"$VENV_PY" - <<'PYEOF'
import importlib
core = ["supervision","numpy","cv2","paho.mqtt","requests"]
opt  = ["amqtt"]
missing = []
for m in core:
    try:
        mod = importlib.import_module(m)
        print(f"  [OK]      {m:<12} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  [MISSING] {m:<12} ({type(e).__name__})")
        missing.append(m)
for m in opt:
    try:
        mod = importlib.import_module(m)
        print(f"  [OK](opt) {m:<12} {getattr(mod,'__version__','?')}")
    except Exception:
        print(f"  [skip]    {m:<12} 未装（仅真 broker 联调需要）")
if missing:
    raise SystemExit("依赖缺失，安装未完全成功")
print("\n✅ 感知层依赖就绪")
PYEOF

echo ""
echo "=== 下一步（跑真探针，全部已真跑通）==="
echo "  $VENV_PY src/perception/event_gate.py            # B 事件闸自检"
echo "  $VENV_PY src/perception/frigate_source.py        # C 离线路径自检"
echo "  $VENV_PY src/perception/frigate_broker_probe.py  # C 真 broker 联调"
echo "  $VENV_PY src/perception/gate_llm_link.py --backend openai --key-file <你的.env>  # 真 LLM 闭环"
echo ""
echo "密钥纪律：MGA_LLM_API_KEY 从环境变量 / .env.local / --key-file 注入，绝不写代码或聊天。"
