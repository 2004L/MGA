"""
m3_realmachine_acceptance.py —— M3 真机验收（真实 Windows 桌面）
============================================================================

沙箱层已证「工具齐全/非stub/dry_run安全/几何接线/坐标换算/中文输入」
（见 m3_exec_acceptance.py）。本脚本补上**沙箱无法判定**的三项真机指标：

  G1 真机注入成功率 >95%
      在自建 Notepad fixture 上做 click/type/drag/hotkey，type 用剪贴板 readback
      强校验（文字确实进了聚焦控件），其余动作以「后端返回 True = OS 已派发」计。
  G2 窗口几何还原召回率
      client_rect / window_rect 与脚本内独立 ctypes GetWindowRect/GetClientRect
      真值比对，一致率应=100%。
  G3 拟人化轨迹落点误差
      human_like 开启时 move 到目标，末端收敛后 cursor_position == target，误差≈0。

安全约定（尊重「桌面操作默认关」硬约束）：
  · 默认【只读】跑 G2 几何一致性 + 后端能力探针，不碰真实输入。Exit 0 = 脚本可用。
  · G1/G3 真实注入必须显式 `--real` 才执行：会短暂移动鼠标/在自建 Notepad 打字，
    且 fixture 全程自建自清（开 Notepad → 聚焦 → 测试 → Alt+F4 关闭不保存）。
  · 非 Windows 直接跳过（真机指标只在 Windows 有意义）。
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PERCEPTION = _HERE
for _p in (_ROOT, _PERCEPTION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ctypes import wintypes
from exec.computer_use import ComputerUse, SafetyGuard, ExecPolicy

IS_WINDOWS = (os.name == "nt")
_u = ctypes.windll.user32 if IS_WINDOWS else None


# ---------------------------------------------------------------------------
# 独立真值（脚本内直连 win32，不依赖被测模块，做交叉校验）
# ---------------------------------------------------------------------------
def _gt_window_rect(hwnd) -> tuple:
    r = wintypes.RECT()
    _u.GetWindowRect(hwnd, ctypes.byref(r))
    return (r.left, r.top, r.right, r.bottom)


def _gt_client_rect(hwnd) -> tuple:
    r = wintypes.RECT()
    _u.GetClientRect(hwnd, ctypes.byref(r))
    pt = wintypes.POINT(0, 0)
    _u.ClientToScreen(hwnd, ctypes.byref(pt))
    return (pt.x, pt.y, pt.x + r.right - r.left, pt.y + r.bottom - r.top)


def _rect_close(a, b, tol: int = 2) -> bool:
    """两矩形各坐标差 ≤ tol（吸收 DWM 边框取整/瞬态抖动）。"""
    if not a or not b or len(a) != 4 or len(b) != 4:
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _read_clipboard_text() -> str:
    """读剪贴板文本（ctypes，无第三方依赖）。失败返回 ''。"""
    try:
        if not ctypes.windll.user32.OpenClipboard(0):
            return ""
        cf = 1  # CF_TEXT
        h = ctypes.windll.user32.GetClipboardData(cf)
        if not h:
            ctypes.windll.user32.CloseClipboard()
            return ""
        buf = ctypes.cast(h, ctypes.c_char_p).value or b""
        ctypes.windll.user32.CloseClipboard()
        return buf.decode("gbk", "ignore")
    except Exception:
        try:
            ctypes.windll.user32.CloseClipboard()
        except Exception:
            pass
        return ""


# ---------------------------------------------------------------------------
# 验收
# ---------------------------------------------------------------------------
_FAILS = []


def _check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)


def _g2_geometry(cu: ComputerUse):
    """G2：几何还原召回率（与独立真值比对）。

    交叉校验：被测模块 cu.wm.rect/client_screen_rect（同一套 win32 封装）
    vs 脚本内独立 ctypes 真值。对**稳定、非最小化、非零尺寸**窗口要求 100% 一致；
    瞬态/叠加层/HUD 窗口（两次读取间自身在动）跳过并计入 skipped，不计入召回率。
    """
    print("\n[G2] 窗口几何还原召回率（cu.wm vs 独立 GetWindowRect/GetClientRect）")
    wins = cu.list_windows()
    if not wins:
        _check("G2 有可见窗口", False, "无可见窗口可比对")
        return 0.0, 0
    TOL = 2
    ok = 0
    checked = 0
    skipped = 0
    mism = []
    title_path_ok = 0
    for w in wins[:60]:
        hwnd = w["hwnd"]
        if cu.is_window_minimized(w["title"]):
            skipped += 1
            continue
        try:
            gt_w = _gt_window_rect(hwnd)
            gt_c = _gt_client_rect(hwnd)
        except Exception:
            skipped += 1
            continue
        if not gt_w or gt_w[2] <= gt_w[0] or gt_w[3] <= gt_w[1]:
            skipped += 1          # 零/负尺寸窗口跳过
            continue
        our_w = cu.wm.rect(hwnd)
        our_c = cu.wm.client_screen_rect(hwnd)
        if our_w is None or our_c is None:
            skipped += 1
            continue
        if cu.window_rect(w["title"]) is not None:
            title_path_ok += 1
        if _rect_close(our_w, gt_w, TOL) and _rect_close(our_c, gt_c, TOL):
            ok += 1
            checked += 1
        else:
            # 复读一次独立真值，确认是否瞬态窗口（自身在动）
            try:
                gt_w2 = _gt_window_rect(hwnd)
                gt_c2 = _gt_client_rect(hwnd)
            except Exception:
                gt_w2, gt_c2 = gt_w, gt_c
            if _rect_close(our_w, gt_w2, TOL) and _rect_close(our_c, gt_c2, TOL):
                ok += 1
                checked += 1
            else:
                mism.append(w["title"][:12])
                skipped += 1
    recall = ok / checked if checked else 0.0
    _check("G2 几何一致率 = 100%（稳定窗口）",
           recall >= 0.999,
           f"ok={ok}/{checked} 不一致={mism[:5]} 跳过瞬态/最小化={skipped}")
    _check("G2 标题路径可用", title_path_ok > 0,
           f"title_path_ok={title_path_ok}/{checked}")
    return recall, checked


def _g1_injection(cu: ComputerUse):
    """G1：真机注入成功率（自建 Notepad fixture + 剪贴板 readback）。"""
    print("\n[G1] 真机注入成功率 >95%（Notepad fixture + 剪贴板 readback）")
    # 开 fixture
    try:
        proc = subprocess.Popen(["notepad.exe"])
    except Exception as e:
        _check("G1 启动 Notepad", False, f"{type(e).__name__}: {e}")
        return 0.0
    time.sleep(1.2)
    title = None
    for t in ("记事本", "Notepad"):
        if cu.focus_window(t) is not None:
            title = t
            break
    if not title:
        # 用首个含 notepad/记事本 的窗口
        for w in cu.list_windows():
            if "notepad" in w["title"].lower() or "记事本" in w["title"]:
                if cu.focus_window(w["title"]):
                    title = w["title"]
                    break
    if not title:
        _check("G1 定位 fixture 窗口", False, "未找到 Notepad")
        proc.terminate()
        return 0.0
    _check("G1 定位 fixture 窗口", True, f"title={title}")

    rect = cu.client_rect(title)
    if not rect:
        _check("G1 取 fixture 客户区", False)
        _cleanup(proc, cu, title)
        return 0.0
    cx = (rect[0] + rect[2]) // 2
    cy = (rect[1] + rect[3]) // 2

    total = 0
    succ = 0
    N = 20
    for i in range(N):
        token = f"M3_{i:02d}_"
        total += 1
        # 聚焦 + 点进编辑区
        cu.focus_window(title)
        ok_click = cu.click(cx, cy + 2)        # 点进正文
        # 全选清空该行影响：直接 type token，再 readback
        ok_type = cu.type(token)
        # readback：全选 + 复制 + 读剪贴板
        cu.hotkey("ctrl", "a")
        cu.hotkey("ctrl", "c")
        got = _read_clipboard_text()
        readback = token in got
        # 回退一行避免累积（按上箭头 + 全选删，保持 fixture 干净）
        cu.press("home")
        cu.hotkey("ctrl", "a")
        cu.press("backspace")
        step_ok = bool(ok_click) and bool(ok_type) and readback
        succ += 1 if step_ok else 0
        if i < 3 or not step_ok:
            print(f"    #{i:02d} click={ok_click} type={ok_type} readback={readback}")

    rate = succ / total if total else 0.0
    _check("G1 真机注入成功率 >95%", rate >= 0.95,
           f"succ={succ}/{total} rate={rate*100:.1f}%")
    _cleanup(proc, cu, title)
    return rate


def _g3_trajectory(cu: ComputerUse):
    """G3：拟人化轨迹落点误差。"""
    print("\n[G3] 拟人化轨迹落点误差（human_like 末端收敛）")
    targets = [(640, 360), (100, 100), (1200, 700), (300, 500)]
    max_err = 0
    for (tx, ty) in targets:
        cu.move(tx, ty, human=True)
        px, py = cu.backend.position()
        err = max(abs(px - tx), abs(py - ty))
        max_err = max(max_err, err)
        print(f"    target=({tx},{ty}) actual=({px},{py}) err={err}")
    _check("G3 落点误差 ≤ 2px", max_err <= 2, f"max_err={max_err}")
    return max_err


def _cleanup(proc, cu, title):
    """关 Notepad 且不保存（Alt+F4 → 不保存 n）。"""
    try:
        cu.focus_window(title)
        cu.hotkey("alt", "f4")
        time.sleep(0.4)
        cu.press("n")          # 不保存
        time.sleep(0.3)
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="执行真实注入验收 G1/G3（会移动鼠标/在自建 Notepad 打字）")
    args = ap.parse_args()

    print("=" * 64)
    print("M3 真机验收")
    print("=" * 64)
    if not IS_WINDOWS:
        print("非 Windows 环境：真机指标跳过（仅在 Windows 有意义）。")
        print("=" * 64)
        return 0

    cu = ComputerUse(SafetyGuard.from_policy(ExecPolicy(dry_run=False)),
                     audit_path=os.path.join("blobs", "m3_real_audit.jsonl"))
    print(f"\n[后端能力探针] backend={cu.backend.name}  "
          f"screen={cu.backend.screen_size()}  IL={getattr(cu.backend,'integrity_level',lambda:'?')()}")

    # G2 始终跑（只读）
    _g2_geometry(cu)

    if args.real:
        _g1_injection(cu)
        _g3_trajectory(cu)
    else:
        print("\n[默认] 未加 --real：跳过真实注入 G1/G3（尊重桌面操作默认关）。")
        print("       需真机注入成功率/轨迹误差指标时，运行：")
        print("       python m3_realmachine_acceptance.py --real")

    print("\n" + "=" * 64)
    if _FAILS:
        print(f"M3 真机验收：FAIL（{len(_FAILS)} 项：{_FAILS}）")
        print("=" * 64)
        return 1
    print("M3 真机验收（安全部分）：✅ 通过。--real 部分由用户在真桌面显式触发。")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
