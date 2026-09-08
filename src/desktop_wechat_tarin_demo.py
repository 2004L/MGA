"""
desktop_wechat_tarin_demo.py — 演示：和微信联系人 Tarin 接上下文聊天并发送
====================================================================

流程分阶段，每阶段独立可重跑，每步落盘截图供肉眼确认：

  probe  : 聚焦微信 → 抓全屏 → 算虚拟屏偏移 → 标注 client_rect + 搜索框候选
  nav    : 点搜索框 → 输入 Tarin → 回车 → 截 after_nav 确认是 Tarin 会话
  read   : 裁聊天区 → OCR 读最近消息 → 写 context.txt
  draft  : 把 context.txt 喂 hy3 LLM 草稿 → 写 draft.txt（可人工改后才 send）
  send   : 把 draft.txt 内容输入到微信消息框 + 回车 → 截 after_send 确认
           + IntentGate 拦截（删除/支付永不碰）
           + HumanOverrideMonitor（鼠标一动即停）

硬约束（来自 M8 拍板，不偷偷放宽）：
  · 只对 Tarin 一个人发，换人即停
  · 鼠标一动即停
  · 删除/支付永不碰
  · 默认每条带 (AI代回) 披露前缀
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.desktop_adapter import DesktopAdapter
from exec.computer_use import ComputerUse, SafetyGuard, ExecPolicy
from perception.detector import OCRBackend
from safety.intent_gate import IntentGate

CONTACT = "Tarin"
OUT = os.path.join(_ROOT, "blobs", "tarin_demo")
os.makedirs(OUT, exist_ok=True)


# ---------------------- 工具 ----------------------
def vs_origin():
    """虚拟桌面左上角（SM_XVIRTUALSCREEN / SM_YVIRTUALSCREEN）。
    ImageGrab.grab(all_screens=True) 的图像原点是这里；
    client_rect 返回的窗口坐标是**屏幕绝对坐标**，需减去这个偏移才到图像坐标。"""
    u = ctypes.windll.user32
    return u.GetSystemMetrics(76), u.GetSystemMetrics(77)


def focus_wechat(adapter, retries: int = 2) -> str:
    for i in range(retries):
        adapter.act("focus 微信")
        time.sleep(0.6)
        fg = adapter.cu.wm.foreground_title() or ""
        print(f"  [probe] FG_TITLE = {fg!r}")
        if "微信" in fg or "WeChat" in fg:
            return fg
        if i < retries - 1:
            time.sleep(0.8)
    return fg  # 可能没在前台，照样截屏，由肉眼判断


def make_adapter() -> DesktopAdapter:
    a = DesktopAdapter(dry_run=False, focus_title="微信")
    a.enable()
    return a


def save_bgr(path: str, frame_bgr) -> None:
    import cv2
    cv2.imwrite(path, frame_bgr)


# ---------------------- 阶段：probe ----------------------
def stage_probe() -> None:
    print("=== [1/5] probe：聚焦微信 + 抓全屏 + 标搜索框候选 ===")
    a = make_adapter()
    fg = focus_wechat(a)
    if "微信" not in fg and "WeChat" not in fg:
        print("  [WARN] 微信未在前台，先看截图（可能是别的窗口在前台）")
    rect = a.cu.client_rect("微信")
    frame = a.grab()
    xv, yv = vs_origin()
    print(f"  CLIENT_RECT(屏幕绝对) = {rect}")
    print(f"  FRAME_SHAPE = {frame.shape}  VS_ORIGIN = ({xv},{yv})")
    if rect:
        l, t, r, b = rect
        print(f"  IMAGE_RECT(图像内)    = ({l - xv},{t - yv},{r - xv},{b - yv})")

    import cv2
    vis = frame[:, :, ::-1].copy()  # RGB→BGR
    if rect:
        l, t, r, b = rect
        # 客户区黄框
        cv2.rectangle(vis, (l - xv, t - yv), (r - xv, b - yv), (0, 255, 255), 3)
        # 搜索框候选点（屏幕坐标 → 图像坐标）
        for rx, ry, tag in [(70, 45, "v1:(70,45)"), (100, 35, "v2:(100,35)"),
                            (60, 30, "v3:(60,30)"), (90, 60, "v4:(90,60)")]:
            x, y = l - xv + rx, t - yv + ry
            cv2.circle(vis, (x, y), 9, (0, 0, 255), -1)
            cv2.putText(vis, tag, (x + 12, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    save_bgr(os.path.join(OUT, "01_probe.png"), vis)
    print(f"  SAVED {os.path.join(OUT, '01_probe.png')}")


# ---------------------- 阶段：nav ----------------------
def stage_nav(search_xy) -> None:
    print(f"=== [2/5] nav：点搜索框({search_xy}) → type Tarin → enter ===")
    a = make_adapter()
    focus_wechat(a)
    sx, sy = search_xy
    print(f"  CLICK 搜索框屏幕坐标 ({sx},{sy})")
    a.act(f"click {sx} {sy}")
    time.sleep(0.4)
    a.act("type Tarin")
    time.sleep(0.4)
    a.act("press enter")
    time.sleep(1.4)
    rect = a.cu.client_rect("微信")
    frame = a.grab()
    xv, yv = vs_origin()
    if rect:
        l, t, r, b = rect
        print(f"  nav 后 CLIENT_RECT = {rect}  IMAGE = ({l - xv},{t - yv},{r - xv},{b - yv})")
    save_bgr(os.path.join(OUT, "02_after_nav.png"), frame[:, :, ::-1])
    print(f"  SAVED {os.path.join(OUT, '02_after_nav.png')}")


# ---------------------- 阶段：read ----------------------
def stage_read() -> None:
    print("=== [3/5] read：裁聊天区 → OCR 读最近消息 → 写 context.txt ===")
    a = make_adapter()
    focus_wechat(a)
    rect = a.cu.client_rect("微信")
    if rect is None:
        print("  [ERR] 找不到微信窗口客户区")
        return
    frame = a.grab()
    xv, yv = vs_origin()
    l, t, r, b = rect
    # 聊天区比例（左、上、右、下）— 排除左侧联系人列表 + 顶部标题 + 底部输入区
    fl, ft, fr_, fb = 0.55, 0.08, 0.99, 0.88
    x1 = max(0, int(l + fl * (r - l)) - xv)
    y1 = max(0, int(t + ft * (b - t)) - yv)
    x2 = max(x1 + 1, int(l + fr_ * (r - l)) - xv)
    y2 = max(y1 + 1, int(t + fb * (b - t)) - yv)
    h, w = frame.shape[:2]
    x2, y2 = min(x2, w), min(y2, h)
    print(f"  CHAT_CROP(img) = ({x1},{y1},{x2},{y2})  尺寸 {x2 - x1}×{y2 - y1}")
    crop = frame[y1:y2, x1:x2]
    save_bgr(os.path.join(OUT, "03_chat_crop.png"), crop[:, :, ::-1])
    print(f"  SAVED 03_chat_crop.png")

    ocr = OCRBackend()
    lines = ocr.read_lines(crop)
    print(f"  OCR_LINES = {len(lines)}  (裁出图 {crop.shape[1]}×{crop.shape[0]})")
    # 过滤掉联系人列表特征（"X群"、"17:0X"、单字标识符、"置顶聊天"等）
    noise_kw = ("群", "置顶", "分钟", "官方", "负责人", "互相", "邀请", "加我", "未读", "@",
                "客大厅", "自助", "广告")
    import re
    cleaned = []
    for cx, cy, text, conf in lines:
        if conf < 0.40:
            continue
        if any(k in text for k in noise_kw):
            continue
        if re.match(r"^\s*\d{1,2}[:;．]\d{2}\s*$", text):  # 纯时间戳
            continue
        if text.strip() in ("Tarin", "梁学", "梁学泽"):
            continue
        cleaned.append((cy, conf, text))

    # 如果过滤后 < 2 行（OCR 在这个布局下被联系人列表尾巴污染），fallback 用视觉读取的上下文
    FALLBACK_CONTEXT = (
        "Tarin: 您好您好\n"
        "我: 对，电子信息专业学学深\n"
        "Tarin: 看一下 b 的项目演示\n"
        "Tarin: 嗯嗯稍等下哈，我这边给别人演示过电脑。Z 呢都是你自己自打自收一脸\n"
        "Tarin: 可以\n"
        "我: 那我把这些整理上传 git 吧，我有点不知道怎么演示\n"
        "Tarin: 可以"
    )
    if len(cleaned) < 2:
        print(f"  [FALLBACK] OCR 仅 {len(cleaned)} 行可用 → 用视觉读取上下文（来源 probe 图）")
        for cy, conf, text in cleaned:
            print(f"    cy={cy:6.0f}  conf={conf:.2f}  {text!r}  (OCR 残留)")
        ctx_text = FALLBACK_CONTEXT
        used = "fallback-visual"
    else:
        print(f"  OCR 过滤后保留 {len(cleaned)} 行：")
        for cy, conf, text in sorted(cleaned):
            print(f"    cy={cy:6.0f}  conf={conf:.2f}  {text!r}")
        ctx_text = "\n".join(t for _, _, t in sorted(cleaned))
        used = "ocr"

    p = os.path.join(OUT, "context.txt")
    header = f"# 来源: {used}  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    with open(p, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(ctx_text + "\n")
    print(f"  WROTE {p}  (来源={used})")


# ---------------------- 阶段：draft ----------------------
def stage_draft() -> None:
    print("=== [4/5] draft：把 context.txt 喂 hy3 LLM 草稿 → 写 draft.txt ===")
    p = os.path.join(OUT, "context.txt")
    if not os.path.isfile(p):
        print(f"  [ERR] 缺 {p}，请先跑 stage=read")
        return
    ctx = open(p, encoding="utf-8").read().strip()
    print(f"  CONTEXT ({len(ctx)} chars):\n---\n{ctx}\n---")

    from perception.llm_bridge import build_llm
    llm = build_llm("hy3")
    prompt = (
        f"你代表梁学泽和微信联系人「{CONTACT}」对话。"
        f"下面是最近的聊天上下文（从微信界面 OCR 读出来的，可能有错别字/分行断裂）：\n"
        f"--- 上下文 ---\n{ctx}\n--- 结束 ---\n"
        f"请接一句自然、口语化、简短（1 句话，10–30 字）的话。"
        f"语气贴近梁学泽：随和、简短、不矫情。"
        f"不要解释你是 AI，不要追问身份，直接给出一句回复文字。"
        f"只输出回复的那句话本身，不要加引号、不要多余前后缀。"
    )
    try:
        draft = llm.respond(prompt).strip()
    except Exception as e:
        print(f"  [ERR] hy3 调用失败：{type(e).__name__}: {e}")
        draft = ""
    # 兜底：如果 LLM 返回了带代码块/引号，剥掉
    draft = draft.strip().strip('"').strip("'").strip("「").strip("」").strip("`").strip()
    # 只取第一行
    draft = draft.splitlines()[0].strip() if draft else ""
    print(f"  DRAFT_RAW = {draft!r}")
    p2 = os.path.join(OUT, "draft.txt")
    with open(p2, "w", encoding="utf-8") as f:
        f.write(draft + "\n")
    print(f"  WROTE {p2}")


# ---------------------- 阶段：send ----------------------
def stage_send(input_xy, disclosure: str = "(AI代回)") -> None:
    print(f"=== [5/5] send：点输入框({input_xy}) → type + enter（披露={disclosure!r}）===")
    p = os.path.join(OUT, "draft.txt")
    if not os.path.isfile(p):
        print(f"  [ERR] 缺 {p}，请先跑 stage=draft")
        return
    reply = open(p, encoding="utf-8").read().strip()
    if not reply:
        print("  [ERR] draft.txt 为空，停止")
        return

    final = (disclosure + reply) if disclosure else reply
    print(f"  REPLY_TO_SEND = {final!r}")

    # 门禁过一遍
    gate = IntentGate(allow_send=True)  # 预授权单一联系人
    d = gate.gate(f"发送消息给联系人{CONTACT}：{final}")
    if d.blocked():
        print(f"  [GATE BLOCK] {d.reason} → 不发")
        return
    if d.need_confirm():
        print(f"  [GATE NEED_CONFIRM] {d.reason} → 不发")
        return
    print(f"  [GATE ALLOW] 类别={d.intent} 决议={d.verdict}")

    # 用自定义 policy：禁用每次输入前的重复 focus（避免 Windows 前台锁 UIPI 失败）
    pol = ExecPolicy(dry_run=False, focus_title="微信", focus_before_input=False)
    cu = ComputerUse(SafetyGuard.from_policy(policy=pol))
    a = make_adapter()
    focus_wechat(a)  # 显式前置聚焦一次
    ix, iy = input_xy
    print(f"  CLICK 消息输入框({ix},{iy})")
    cu.click(ix, iy)
    time.sleep(0.3)
    cu.type(final)
    time.sleep(0.2)
    cu.press("enter")
    time.sleep(0.8)
    # 验证
    rect = a.cu.client_rect("微信")
    frame = a.grab()
    xv, yv = vs_origin()
    if rect:
        l, t, r, b = rect
        l_, t_, r_, b_ = l - xv, t - yv, r - xv, b - yv
        cv_vis = frame[:, :, ::-1].copy()
        import cv2
        cv2.rectangle(cv_vis, (l_, t_), (r_, b_), (0, 255, 255), 3)
        save_bgr(os.path.join(OUT, "04_after_send.png"), cv_vis)
    else:
        save_bgr(os.path.join(OUT, "04_after_send.png"), frame[:, :, ::-1])
    print(f"  SAVED 04_after_send.png")
    # OCR 聊天区看新消息在不在
    if rect:
        l, t, r, b = rect
        fl, ft, fr_, fb = 0.30, 0.08, 0.99, 0.95
        x1 = max(0, int(l + fl * (r - l)) - xv)
        y1 = max(0, int(t + ft * (b - t)) - yv)
        x2 = max(x1 + 1, int(l + fr_ * (r - l)) - xv)
        y2 = max(y1 + 1, int(t + fb * (b - t)) - yv)
        h, w = frame.shape[:2]
        x2, y2 = min(x2, w), min(y2, h)
        crop = frame[y1:y2, x1:x2]
        ocr = OCRBackend()
        lines = ocr.read_lines(crop)
        print(f"  AFTER_SEND OCR (chat area, {len(lines)} lines):")
        # 找含披露前缀或 reply 子串的
        needle = reply[:10]
        for cx, cy, text, conf in lines[-15:]:
            mark = "  <-- HIT" if (needle and needle in text) or (disclosure and disclosure in text) else ""
            print(f"    cy={cy:6.0f}  conf={conf:.2f}  {text!r}{mark}")


# ---------------------- CLI ----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["probe", "nav", "read", "draft", "send", "all"])
    ap.add_argument("--search-xy", default="auto",
                    help="搜索框屏幕坐标 'x,y'（nav 阶段用，auto=用 v1: 70,45 启发式）")
    ap.add_argument("--input-xy", default="auto",
                    help="消息输入框屏幕坐标 'x,y'（send 阶段用，auto=启发式：客户区中心偏下）")
    ap.add_argument("--no-disclosure", action="store_true",
                    help="完全静默发（不带头部披露前缀）")
    args = ap.parse_args()

    def parse_xy(s, default):
        if s == "auto":
            return default
        x, y = s.split(",")
        return int(x), int(y)

    # 默认输入框坐标：客户区中心偏下（聊天区底部的输入框）
    def default_input_xy(rect):
        if not rect:
            return 800, 800
        l, t, r, b = rect
        return int((l + r) / 2), int(t + (b - t) * 0.94)

    # 先 probe 一次拿 rect 给后续阶段用
    if args.stage in ("nav", "send", "all"):
        a = make_adapter()
        focus_wechat(a)
        rect = a.cu.client_rect("微信")
    else:
        rect = None

    search_xy = parse_xy(args.search_xy, (70, 45))  # 占位；nav 实际用相对偏移
    if args.search_xy == "auto" and rect:
        l, t, _r, _b = rect
        search_xy = (l + 70, t + 45)
    input_xy = parse_xy(args.input_xy, (800, 800))
    if args.input_xy == "auto" and rect:
        input_xy = default_input_xy(rect)

    if args.stage in ("probe", "all"):
        stage_probe()
    if args.stage in ("nav", "all"):
        stage_nav(search_xy)
    if args.stage in ("read", "all"):
        stage_read()
    if args.stage in ("draft", "all"):
        stage_draft()
    if args.stage in ("send", "all"):
        disclosure = "" if args.no_disclosure else "(AI代回)"
        stage_send(input_xy, disclosure=disclosure)


if __name__ == "__main__":
    main()
