"""
desktop_play_dino.py —— 桌面 Computer Use 自操作：找 Chrome → 开网站 → 自己玩小恐龙
===================================================================================
把 M6「桌面双系统闭环」接到现成的小恐龙真机 Pilot 的**前置段**：
  ① 桌面 Computer Use（DesktopAdapter）自己找并打开 Chrome、导航到 chrome://dino
  ② 交给已真机验证的 ChromeDinoPilot（demo_dino_real）接管按键跳跃、撞死自动重开

安全（尊重硬约束）：
  - 桌面操作默认关；本脚本必须显式 --real 才碰真实桌面。
  - 不带 --real：只打印将要执行的步骤（dry plan），不碰任何窗口/键鼠、不注入。
  - --real 由你本人在真桌面显式触发；我不会在无人确认时乱动你的鼠标。

复用（不重造）：
  - 启动/导航用 DesktopAdapter（M1/M3，sendinput 原生后端）
  - 玩用 demo_dino_real.ChromeDinoPilot（真机验证：自动定位游戏区 + 发空格/下键 + 撞死自动重开）

用法：
  python desktop_play_dino.py                 # 只看计划（不碰桌面）
  python desktop_play_dino.py --real         # 真机：自己找 Chrome 开 dino 并开玩
  python desktop_play_dino.py --real --no-launch   # Chrome 已在 dino 页，直接玩
  python desktop_play_dino.py --real --frames 5000 --reaction 0.22 --show
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent.desktop_adapter import DesktopAdapter
import demo_dino_real as D


def _plan() -> list:
    return [
        "1. Win+R 打开运行框",
        "2. 输入 chrome 回车 → 启动 Chrome",
        "3. Ctrl+L 聚焦地址栏 → 输入 chrome://dino 回车 → 导航到小恐龙",
        "4. 等页面加载（约 2.5s）",
        "5. 交棒 ChromeDinoPilot：自动定位游戏区 + 发空格/下键跳跃 + 撞死自动重开",
    ]


def launch_chrome(adapter: DesktopAdapter, verbose: bool = True) -> None:
    """用桌面 Computer Use 自己启动 Chrome 并导航到小恐龙。"""
    def log(*a):
        if verbose:
            print("[launch]", *a)

    log("Win+R → 运行框")
    adapter.act("hotkey win r")
    time.sleep(0.5)
    log("输入 chrome 回车")
    adapter.act("type chrome")
    time.sleep(0.3)
    adapter.act("press enter")
    time.sleep(3.0)                      # 等 Chrome 起来
    log("Ctrl+L → 地址栏，导航 chrome://dino")
    adapter.act("hotkey ctrl l")
    time.sleep(0.4)
    adapter.act("type chrome://dino")
    time.sleep(0.3)
    adapter.act("press enter")
    time.sleep(2.5)                      # 等 dino 加载
    log("导航完成，交棒 Pilot")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="桌面 CU 自操作：找 Chrome 开网站 玩小恐龙")
    ap.add_argument("--real", action="store_true",
                    help="显式真机模式：实际启动 Chrome+导航+注入按键（默认只打印计划）")
    ap.add_argument("--no-launch", action="store_true",
                    help="跳过启动 Chrome（假设已在 chrome://dino 页面）")
    ap.add_argument("--frames", type=int, default=20000, help="Pilot 最大帧数")
    ap.add_argument("--reaction", type=float, default=0.22, help="触发阈值(秒)")
    ap.add_argument("--show", action="store_true", help="弹标注窗口(OBS/手机录屏)")
    ap.add_argument("--no-vision", action="store_true", help="关大模型定位兜底")
    ap.add_argument("--no-window", action="store_true", help="关 OS 窗口级定位")
    ap.add_argument("--no-dl", action="store_true", help="关 YOLOE 主通道(只用 CV)")
    ap.add_argument("--no-restart", action="store_true", help="撞死不自动重开")
    ap.add_argument("--region", default="", help="手动指定游戏区 x,y,w,h")
    args = ap.parse_args()

    if not args.real:
        print("== 桌面 CU 自操作小恐龙 · 计划（未 --real，不碰桌面）==")
        for s in _plan():
            print("  " + s)
        print("\n>>> 要真机执行，加 --real（会真实启动 Chrome 并接管键鼠）。")
        return

    print("== 桌面 CU 自操作小恐龙 · REAL 真机 ==")
    # 桌面操作需显式开启 + 真实注入后端
    adapter = DesktopAdapter(dry_run=False, focus_title="Chrome")
    adapter.enable()

    if not args.no_launch:
        try:
            launch_chrome(adapter, verbose=True)
        except Exception as e:
            print(f"[launch] 自动启动 Chrome 失败：{type(e).__name__}: {e}")
            print("  请手动打开 Chrome 并访问 chrome://dino，然后用 --no-launch 重跑本脚本。")
            return

    # 交棒给已真机验证的 Pilot
    region = None
    if args.region:
        region = tuple(int(x) for x in args.region.split(","))
    if region is None:
        try:
            region = D.auto_region(use_vision=(not args.no_vision),
                                    use_window=(not args.no_window))
            D.save_region(region)
            if region[2] > 1000:
                region = (region[0], region[1], 1000, region[3])
                D.save_region(region)
        except Exception as e:
            saved = D.load_region()
            if saved:
                region = saved
                print(f"[region] 自动量失败({e})，复用上次保存 {saved}")
            else:
                print(f"[region] 定位失败：{e}；请手动 --region 或先开 dino 页")
                return

    pilot = D.ChromeDinoPilot(region, sim=False, reaction_time=args.reaction,
                              prefer="sendinput", show=args.show,
                              use_dl=(not args.no_dl))
    pilot.auto_restart = (not args.no_restart)
    pilot.run(n_frames=args.frames, dt=0.05)


if __name__ == "__main__":
    main()
