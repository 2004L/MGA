"""
desktop_wechat_scout.py —— 只读「挑人 + 起草」scout（绝不含发送路径）
============================================================================
用途：去你微信【只读】挑一个最近联系人，起草 3 句话，停下来给你过目。
      —— 本脚本永远不发送任何消息（没有 act 发送调用），发不发由你确认后另一步决定。

安全边界：
  · 只读：仅聚焦/打开微信 + 截屏 + OCR 列最近会话；不点进聊天、不打字、不回车。
  · 不发送：本文件无 send/type-enter 到输入框的逻辑。
  · 披露：起草内容默认带 (AI代回) 前缀说明（关掉需你显式要求）。

用法：
  python desktop_wechat_scout.py            # 真机只读挑人 + 起草，打印预览
  python desktop_wechat_scout.py --no-draft # 只列候选联系人，不调 LLM 起草
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.desktop_adapter import DesktopAdapter
from perception.detector import OCRBackend

# 看起来像「服务号/订阅号/系统通知/纯数字」的，挑人时降权跳过（不发给它们）
_NOISE_HINTS = ("微信团队", "订阅号", "服务通知", "文件传输", "交易", "支付", "通知")


def is_noise(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return True
    if any(h in n for h in _NOISE_HINTS):
        return True
    # 纯数字 / 很短的串当作噪声
    if n.isdigit():
        return True
    return False


def open_and_focus(adapter: "DesktopAdapter") -> bool:
    """只读：聚焦已开的微信；没开则启动（真机动作，但只是开 App，不发送）。"""
    rect = adapter.cu.client_rect("微信")
    if rect is None:
        print("[scout] 未检测到微信窗口，尝试启动（Win+R → wechat）…")
        adapter.act("hotkey win r")
        time.sleep(0.5)
        adapter.act("type wechat")
        time.sleep(0.3)
        adapter.act("press enter")
        time.sleep(3.0)
        rect = adapter.cu.client_rect("微信")
        if rect is None:
            print("[scout] 启动后仍未检测到微信窗口，可能未安装/未登录。停止。")
            return False
    else:
        adapter.act("focus 微信")
        time.sleep(0.6)
    return True


def list_recent_contacts(adapter: "DesktopAdapter", ocr: "OCRBackend") -> list:
    """截屏 → 裁左侧会话列表 → OCR 出候选名字（按 y 排序）。"""
    frame = adapter.grab()
    h_img, w_img = frame.shape[:2]
    # screenshot() 截的是微信窗口本身（原点 0,0），client_rect 是虚拟桌面坐标
    # （副屏可能为负），用它裁剪会越界。故裁剪一律用【图像相对坐标】。
    x1, y1 = 0, int(0.06 * h_img)
    x2, y2 = int(0.32 * w_img), int(0.96 * h_img)
    crop = frame[y1:y2, x1:x2]
    lines = ocr.read_lines(crop)   # (cx, cy, text, conf) 已按 y 排序
    # 只取像名字的短行（去过长预览），并去重
    seen = set()
    out = []
    for (cx, cy, text, conf) in lines:
        txt = (text or "").strip()
        if not txt or txt in seen:
            continue
        if len(txt) > 12:        # 太长的多半是消息预览，跳过
            continue
        seen.add(txt)
        out.append((txt, round(float(conf), 2), int(cy)))
    return out


def draft_three_sentences(contact: str, backend: str = "hy3",
                          disclosure: str = "(AI代回)") -> list:
    """起草 3 句（不读对方历史，保护隐私）。优先 hy3，失败回退本地模板。"""
    try:
        from perception.llm_bridge import build_llm
        llm = build_llm(backend)
        prompt = (
            f"你是梁学泽的 AI 助手林梦梦，现在要替梁学泽主动给微信好友「{contact}」"
            f"发 3 句轻松自然的闲聊开场白（像朋友间随便聊）。\n"
            f"要求：中文、口语化、每句独立成行、不要问号结尾的审问、不要提你是 AI、"
            f"不要涉及钱/转账/支付/删除等高危内容。只输出 3 行正文，不要序号、不要解释。"
        )
        raw = llm.respond(prompt)
        parts = [p.strip() for p in raw.replace("\r", "\n").split("\n") if p.strip()]
        if len(parts) >= 3:
            parts = parts[:3]
        else:
            parts = (parts + ["最近咋样呀？", "好久没聊了哈哈", "有空出来约一波？"])[:3]
    except Exception as e:
        print(f"[scout] LLM 起草失败（{type(e).__name__}），用本地模板：{e}")
        parts = ["在忙吗？突然想到你了哈哈", "最近咋样，好久没聊了", "有空出来约一波？"]
    # 披露前缀（仅首句加，避免每条都挂前缀显得怪）
    if disclosure:
        parts[0] = f"{disclosure}{parts[0]}"
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description="只读挑人 + 起草（不发送）")
    ap.add_argument("--llm-backend", default="hy3", help="起草用 LLM（默认 hy3）")
    ap.add_argument("--no-disclosure", action="store_true", help="起草不带 (AI代回) 前缀")
    ap.add_argument("--no-draft", action="store_true", help="只列候选，不调 LLM 起草")
    args = ap.parse_args()

    disclosure = "" if args.no_disclosure else "(AI代回)"
    adapter = DesktopAdapter(dry_run=False, focus_title="微信")
    adapter.enable()                       # 仅用于 focus/启动；本脚本不调用发送
    ocr = OCRBackend(scale=0.6)

    if not open_and_focus(adapter):
        return

    print("\n[scout] 截屏并 OCR 最近会话列表…")
    contacts = list_recent_contacts(adapter, ocr)
    if not contacts:
        print("[scout] 没从左侧列表 OCR 到候选名字（微信布局/缩放可能需微调）。")
        print(">>> 你可以手动告诉我发给谁，我直接起草。")
        return

    print(f"[scout] 候选（按屏幕从上到下，已跳过服务号/纯数字）：")
    candidates = [(n, c, y) for (n, c, y) in contacts if not is_noise(n)]
    for i, (n, c, _y) in enumerate(candidates, 1):
        print(f"  {i}. {n}  (ocr_conf={c})")

    # 随机挑一个（排除噪声）
    if not candidates:
        print("[scout] 过滤后无合适候选，停止。")
        return
    pick = random.choice(candidates)
    picked_name = pick[0]
    print(f"\n[scout] 我随机挑了：{picked_name}")

    if args.no_draft:
        print("[scout] （按 --no-draft，未起草）")
        return

    sentences = draft_three_sentences(picked_name, args.llm_backend, disclosure)
    print("\n================ 预览（未发送） ================")
    print(f"发给：{picked_name}")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")
    print("================================================")
    print(">>> 以上只是草稿，没有发任何消息。你确认后回复「发」我才真发；"
          "也可说「换X」换人，或「改：…」改内容。")


if __name__ == "__main__":
    main()
