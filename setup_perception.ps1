# setup_perception.ps1 —— MGA 感知层依赖固化安装（Windows PowerShell 版）
# 与 setup_perception.sh 行为一致：建仓库内 .venv，从 requirements.lock.txt 装锁定版本，
# C 盘满时把 pip TEMP 挪到 D:/tmp，装后逐项 import 自检。
# 用法（在 MGA 仓库根目录的 PowerShell 中）：
#   .\setup_perception.ps1
#   .\setup_perception.ps1 -Check        # 只检查缺什么，不安装
$ErrorActionPreference = 'Stop'

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV = Join-Path $ROOT '.venv'

param(
  [switch]$Check
)

# ---- C 盘满兜底：pip 临时目录挪到 D:/tmp ----
$PIP_TMP = if ($env:TMPDIR) { $env:TMPDIR } else { 'D:/tmp' }
New-Item -ItemType Directory -Force -Path $PIP_TMP | Out-Null
$env:TMPDIR = $PIP_TMP
$env:TEMP   = $PIP_TMP
$env:TMP    = $PIP_TMP

Write-Host "=== MGA 感知层依赖安装 ==="
Write-Host "仓库根目录 : $ROOT"
Write-Host "虚拟环境   : $VENV"
Write-Host "pip TEMP   : $PIP_TMP (C 盘满兜底)"

function SelfCheck {
  param([bool]$CheckOnly)
  $code = @'
import importlib, sys
check_only = sys.argv[1] == "1"
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
    if check_only:
        print("`n上面 [MISSING] 即缺失项。执行 .\setup_perception.ps1 安装。")
    sys.exit(1)
print("`n✅ 感知层依赖就绪")
'@
  $args = @(if ($CheckOnly) { "1" } else { "0" })
  & "$VENV/Scripts/python.exe" -c $code @args
  if ($LASTEXITCODE -ne 0) { throw "依赖自检未通过" }
}

if ($Check) {
  Write-Host "`n=== 仅检查（不安装）==="
  if (-not (Test-Path "$VENV/Scripts/python.exe")) {
    Write-Host "⚠️ .venv 不存在，先执行 .\setup_perception.ps1 创建。"
    exit 1
  }
  SelfCheck -CheckOnly $true
  exit 0
}

# ---- 创建虚拟环境 ----
if (-not (Test-Path $VENV)) {
  Write-Host "--- 创建虚拟环境 .venv ---"
  python -m venv $VENV
}

Write-Host "已激活虚拟环境: $VENV"
Write-Host "使用解释器: $(& "$VENV/Scripts/python.exe" --version 2>&1)"

# ---- 安装 ----
Write-Host "`n=== 升级 pip ==="
& "$VENV/Scripts/python.exe" -m pip install --upgrade pip

Write-Host "`n=== 安装锁定依赖 (requirements.lock.txt) ==="
& "$VENV/Scripts/python.exe" -m pip install -r (Join-Path $ROOT 'requirements.lock.txt')

Write-Host "`n=== 安装后自检 ==="
SelfCheck -CheckOnly $false

Write-Host "`n=== 下一步（跑真探针）==="
Write-Host "  $VENV/Scripts/python.exe src/perception/event_gate.py"
Write-Host "  $VENV/Scripts/python.exe src/perception/frigate_source.py"
Write-Host "  $VENV/Scripts/python.exe src/perception/frigate_broker_probe.py"
Write-Host "  $VENV/Scripts/python.exe src/perception/gate_llm_link.py --backend openai --key-file <你的.env>"
Write-Host "`n密钥纪律：MGA_LLM_API_KEY 从环境变量 / .env.local / --key-file 注入，绝不写代码或聊天。"
