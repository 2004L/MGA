# -*- coding: utf-8 -*-
"""
desktop_capability_demo.py —— 桌面 Computer Use 能力 · 老板演示一键编排
================================================================================

一个命令，现场把四块能力依次跑给老板看：
  场景 1  视觉感知（SAM3 概念分割 + mask 质心定位）—— 截一张图，标出 icon/button/window
  场景 2  三级降级链（semantic → cv 兜底）—— 同一张图，两条通路结果对比
  场景 3  意图级安全门禁（删除/支付/发送 三态拦截）—— 纯逻辑、零风险、最强卖点
  场景 4  （可选 --live）自主桌面操作（小恐龙/微信替聊）—— 默认不跑，需显式开

安全设计（现场演示不出事）：
  · 场景 1/2/3 全部用**既有截图** + 纯逻辑，绝不碰你真桌面、绝不发消息。
  · 场景 4 默认关闭；只有显式 --live 才真操作桌面，且微信替聊仍受 IntentGate 红线约束。
  · 所有打印都是"给老板看的现象 + 一句话技术注解"，可直接照着念。

用法：
  python src/desktop_capability_demo.py                 # 跑场景 1/2/3（默认，安全）
  python src/desktop_capability_demo.py --scene perceive
  python src/desktop_capability_demo.py --scene gate
  python src/desktop_capability_demo.py --live           # 额外现场跑小恐龙（真操作）
  python src/desktop_capability_demo.py --img 你的图.png  # 换一张截图

产物：
  blobs/demo_perceive.png     场景 1 的标注图（框 + 质心 + 信息面板）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ---- 路径兜底：让 perception / safety 两个子包都能 import ----
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, os.path.join(SRC, "perception"), os.path.join(SRC, "safety")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import desktop_perceive as dp
from intent_gate import IntentGate, Intent


# ---------------------------------------------------------------------------
# 画图工具
# ---------------------------------------------------------------------------
def _font(size: int):
    win_fonts = os.path.join(os.environ.get("SYSTEMROOT") or r"C:\Windows", "Fonts")
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"):
        p = os.path.join(win_fonts, name)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_annotated(rgb: np.ndarray, scene, out_path: str, title: str = ""):
    """把感知结果画到截图上：bbox 青框 + 中心黄十字 + 编号 + 信息面板。"""
    img = Image.fromarray(rgb).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(max(13, H // 64))
    r = max(10, H // 70)

    for i, e in enumerate(scene.elements[:60]):
        x1, y1, x2, y2 = [int(v) for v in e.bbox]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        d.rectangle([x1, y1, x2, y2], outline=(0, 220, 230, 230), width=2)
        d.line([cx - r, cy, cx + r, cy], fill=(255, 220, 0, 255), width=2)
        d.line([cx, cy - r, cx, cy + r], fill=(255, 220, 0, 255), width=2)
        if i < 40:
            d.text((x1, max(0, y1 - 14)), f"{i}", font=f, fill=(255, 230, 0, 255))

    # 信息面板
    lines = [title or f"active_backend = {scene.active_backend}"]
    lines.append(f"检出元素: {len(scene.elements)} 个  平均置信度: {scene.confidence:.2f}")
    # 取前几个示例
    for e in scene.elements[:6]:
        lbl = (getattr(e, "text", None) or e.label or "?")
        lines.append(f"  · {e.label:<8} conf={e.conf:.2f}  {lbl[:14]}")
    pad = 10
    lh = (f.size + 6) if hasattr(f, "size") else 20
    box_w = max(d.textlength(s, font=f) for s in lines) + pad * 2
    box_h = lh * len(lines) + pad * 2
    d.rectangle([6, 6, 6 + box_w, 6 + box_h], fill=(0, 0, 0, 175))
    for i, s in enumerate(lines):
        d.text((6 + pad, 6 + pad + i * lh), s, font=f, fill=(255, 255, 255, 255))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# 场景 1：视觉感知（SAM3）
# ---------------------------------------------------------------------------
def scene_perceive(img_path: str, goal: str):
    print("\n" + "=" * 70)
    print("场景 1 · 视觉感知（SAM3 概念分割 + mask 质心定位）")
    print("=" * 70)
    if not os.path.isfile(img_path):
        print(f"  [跳过] 缺截图: {img_path}")
        return
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"  [跳过] 读图失败: {img_path}")
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"  [输入] 截图 {os.path.basename(img_path)}  {frame.shape[1]}x{frame.shape[0]}")
    print(f"  [目标] 让模型找出画面里的 '{goal}'（开放词汇，无需预定义类别）")
    print("  [加载] 正在唤醒视觉大脑 SAM3（首次约 30–40s，演示前可空跑一次预热；之后同进程复用仅几秒）...")

    t0 = time.time()
    sc = dp.perceive_pipeline(rgb, goal=goal, use_sam3=True)
    dt = time.time() - t0
    print(f"  [结果] active_backend={sc.active_backend}  检出 {len(sc.elements)} 个元素"
          f"  耗时 {dt:.1f}s")
    out = os.path.join(ROOT, "blobs", "demo_perceive.png")
    draw_annotated(rgb, sc, out, title=f"SAM3 定位 · backend={sc.active_backend}")
    print(f"  [产物] 标注图已保存: {out}")
    print(f"  [讲解] 模型用一句话描述（'{goal}'）就抠出像素级轮廓，"
          f"质心当作点击点——比传统检测框中心更稳。")


# ---------------------------------------------------------------------------
# 场景 2：三级降级链（semantic 主通道 vs cv 兜底）
# ---------------------------------------------------------------------------
def scene_fallback(img_path: str):
    print("\n" + "=" * 70)
    print("场景 2 · 三级降级链（任一环节挂了都有兜底，绝不瞎操作）")
    print("=" * 70)
    if not os.path.isfile(img_path):
        print(f"  [跳过] 缺截图: {img_path}")
        return
    frame = cv2.imread(img_path)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 主通道：SAM3（开放词汇）
    sc1 = dp.perceive_pipeline(rgb, goal="icon", use_sam3=True)
    print(f"  L1.5 主升级通道 SAM3 : backend={sc1.active_backend:<10} "
          f"元素={len(sc1.elements)}")

    # 兜底：CV 几何（无模型也能跑，保证"至少能点"）
    sc2 = dp.perceive_pipeline(rgb, force_backend="cv")
    print(f"  L3   CV 几何兜底     : backend={sc2.active_backend:<10} "
          f"元素={len(sc2.elements)}")
    print("  [讲解] 主通道（SAM3）出问题时，自动降级到 CV 几何兜底，"
          "链路永不中断——active_backend 实时标注当前走哪条路，绝不假装主通道通。")


# ---------------------------------------------------------------------------
# 场景 3：意图级安全门禁（最强卖点，零风险）
# ---------------------------------------------------------------------------
def scene_gate():
    print("\n" + "=" * 70)
    print("场景 3 · 意图级安全门禁（高危操作自动拦截 / 人工确认）")
    print("=" * 70)
    g = IntentGate()  # 默认 SEND 也 BLOCK（未授权）
    cases = [
        ("把下载文件夹里的大文件删除", "BLOCK",       "删除：永远出自动化范围"),
        ("给客户发送这封邮件",         "BLOCK",       "发送：默认出范围，未授权拒绝"),
        ("用微信支付付这笔订单",       "NEED_CONFIRM","支付：可准备但必须人工点确认"),
        ("在表单里填好姓名和电话",     "ALLOW",       "正常任务：放行（再交物理闸）"),
        ("不发送，只是存草稿",         "ALLOW",       "否定式：'不发送'不触发拦截"),
    ]
    print("  语句                              意图       决策          说明")
    print("  " + "-" * 64)
    for text, expect, note in cases:
        d = g.gate(text)
        ok = "OK" if d.verdict == expect else "FAIL"
        print(f"  {text:<26} {d.intent.value:<9} {d.verdict:<12} {ok}")
        print(f"       ↳ {note}")
    print("  [讲解] 删除/发送 默认永不自动化；支付必须真人点确认才注入。"
          "这是硬约束，代码里物理红线，谁也绕不过。")


# ---------------------------------------------------------------------------
# 场景 4（可选 --live）：自主桌面操作
# ---------------------------------------------------------------------------
def scene_live():
    print("\n" + "=" * 70)
    print("场景 4 · 自主桌面操作（--live 显式开启，真实演示）")
    print("=" * 70)
    print("  [说明] 该能力已在 M7/M8 验证：自动开 Chrome 玩小恐龙、微信替聊。")
    print("  [约束] 微信替聊受 IntentGate 红线约束——单联系人 + 鼠标一动即取消"
          " + 删除/支付不碰。")
    try:
        from desktop_play_dino import main as dino_main
        print("  [执行] 启动小恐龙自动玩...")
        dino_main()
    except Exception as e:
        print(f"  [提示] 小恐龙演示未启动（{type(e).__name__}）："
              f"可现场手动开 chrome://dino 后由 Agent 接管。")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="桌面 Computer Use 能力 · 老板演示")
    ap.add_argument("--scene", default="all",
                    choices=["all", "perceive", "fallback", "gate", "live"])
    ap.add_argument("--live", action="store_true",
                    help="额外现场跑自主桌面操作（小恐龙）")
    ap.add_argument("--img", default=os.path.join(ROOT, "wechat_full.png"),
                    help="用于感知演示的截图（默认 wechat_full.png）")
    args = ap.parse_args()

    print("#" * 70)
    print("# 桌面 Computer Use 能力演示  ·  M1–M9 总览")
    print("# 视觉感知(SAM3) / 三级降级 / 意图门禁 / 自主操作")
    print("#" * 70)

    if args.scene in ("all", "perceive"):
        scene_perceive(args.img, goal="icon")
    if args.scene in ("all", "fallback"):
        scene_fallback(args.img)
    if args.scene in ("all", "gate"):
        scene_gate()
    if args.scene == "live" or args.live:
        scene_live()

    print("\n" + "=" * 70)
    print("演示结束。安全场景（1/2/3）全程未碰真桌面、未发任何消息。")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
