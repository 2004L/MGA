"""
demo_locator.py —— ③ 大脑自训练成果的可视化 demo（一条命令，当场看到效果）
========================================================================
做的事：截你当前屏幕 → 检测 UI 元素 → 用**刚训练好的 LoRA 大脑**定位目标
       → 把「训练前预测」「训练后预测」「真实答案」画在同一张图上，并算误差。

为什么需要它：loss 下降只是数字，不等于模型真的会定位。模型完全可以靠
「背住训练集里那几个坐标」把 loss 刷到 0.02，却在没见过的屏幕上乱点。
这个 demo 就是照妖镜——把预测画到真实截图上跟真值元素比：
  蓝圈 = 训练前(base MiniMind)预测
  红圈 = 训练后(+LoRA)预测
  绿圈 = 真实元素中心（标准答案）
三者重合度一眼可见，且给出「预测是否落在目标元素框内」的硬判定。

用法：
  # 自动从当前屏幕挑一个目标来定位（最简单）
  PYTHONPATH=src python -m demo_locator

  # 指定目标（名字要和屏幕上的文字对得上，否则无真值可比）
  PYTHONPATH=src python -m demo_locator --goal 搜索

  # 对已有截图跑，不重新截屏
  PYTHONPATH=src python -m demo_locator --shot blobs/shots/xxx.png --goal 搜索

输出：blobs/demo_<时间戳>.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# src 既是脚本所在目录也是包根（perception/、exec/ 都在其下）
SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
ROOT = os.path.dirname(SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image, ImageDraw, ImageFont

from perception.detector import (OCRBackend, ScreenParserBackend,
                                 TriLevelLocator, UIALocator)
from perception.llm_bridge import (DEFAULT_MAX_ELEMENTS, _extract_json,
                                   build_prompt, format_scene)
from exec.computer_use import ComputerUse, SafetyGuard

DEFAULT_BASE = os.path.join("blobs", "weights", "minimind2-small")
DEFAULT_ADAPTER = os.path.join("blobs", "weights", "mga-lora")

# 可点控件优先（与 main.pick_auto_goal 同策略）：按钮类名比整段正文更像目标
CLICKABLE_LABELS = ("Button", "Icon", "CheckBox", "RadioButton",
                    "MenuItem", "Tab", "Input", "Link", "Utility Button")


def goal_candidates(els):
    """列出屏幕上「可作目标」的带文字元素，返回 [(label, text, n_cjk), ...]。

    过滤很关键：OCR 会把图标边框、噪点误认成字（实测产出过 "on"/"L7"/"94"/
    "STS" 这类）。拿噪声当题目有两个后果——① 真值退化（匹配不到真实元素，
    demo 判定失去意义）；② 更糟的是训练数据若大量是这种噪声，模型学到的就是
    噪声映射，换一屏立刻现原形。故纯英文/数字的短串一律不要。
    """
    cands = []
    for e in els:
        t = (getattr(e, "text", None) or "").strip()
        if len(t) < 2 or len(t) > 12:
            continue
        if "?" in t or "？" in t:
            continue
        n_cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
        n_ok = sum(1 for ch in t if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if n_ok < 2:
            continue
        if n_cjk == 0 and len(t) < 4:      # 纯 ASCII 短串 ≈ OCR 噪声
            continue
        cands.append((getattr(e, "label", "") or "", t, n_cjk))
    return cands


def pick_auto_goal(els, cursor: int = 0):
    """从当前屏幕**实际存在**的带文字元素里挑一个当目标。

    为什么不写死目标：你的桌面每时每刻都不一样，写死的「搜索」在你切到
    别的软件后根本不存在 → 没有真值可比 → demo 变成空跑。从屏幕实际内容
    出题，(题目, 答案) 在任何页面都天然成立。
    """
    cands = goal_candidates(els)
    if not cands:
        return None
    # 去歧义：排除屏上出现多次的同名目标（如「精选」同时是顶部标签+底部导航，
    # 真值与模型可能选不同实例 → 虚假大误差）。优先选唯一名元素。
    from collections import Counter
    cnt = Counter(t for _, t, _ in cands)
    pool = [c for c in cands if cnt[c[1]] == 1] or cands
    # 含汉字优先 → 可点控件优先 → 短名优先（短名更像按钮标题而非整段正文）
    pool.sort(key=lambda x: (x[2] == 0, x[0] not in CLICKABLE_LABELS, len(x[1])))
    return pool[cursor % len(pool)][1]


def find_ground_truth(els, goal):
    """按 goal 在检测结果里找最匹配的元素（与 OracleLLM 同逻辑，即训练数据的
    「标准答案」来源）。用它当真值，demo 的判定才和训练目标口径一致。"""
    kw = goal.replace("点击", "").replace("按钮", "").strip() or goal
    best, best_score = None, -1
    for e in els:
        label = (getattr(e, "text", None) or getattr(e, "label", None) or "")
        score = 0
        if kw and kw in label:
            score = 3
        elif kw and kw.lower() in label.lower():
            score = 2
        if score == 0 and label and goal and label in goal:
            score = 1
        if score > best_score:
            best_score, best = score, e
    return best


def coords_of(response: str):
    """从模型响应里取坐标。**故意不做 target 回退**——回退会用检测结果帮模型
    补坐标，把「模型没输出坐标」粉饰成「模型点对了」，demo 就失去鉴别力。"""
    data = _extract_json(response)
    c = data.get("coordinates")
    if isinstance(c, (list, tuple)) and len(c) == 2:
        try:
            return (int(c[0]), int(c[1]))
        except Exception:
            return None
    return None


def in_bbox(pt, bbox, tol: int = 8) -> bool:
    if pt is None or not bbox or len(bbox) != 4:
        return False
    x, y = pt
    x1, y1, x2, y2 = bbox
    return (x1 - tol) <= x <= (x2 + tol) and (y1 - tol) <= y <= (y2 + tol)


def dist(a, b) -> float:
    if not a or not b:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def load_brain(base_dir: str, adapter_dir: str):
    """加载 base MiniMind + LoRA 适配器。返回一个可「临时摘掉适配器」的模型，
    这样同一份权重能同时产出训练前/训练后的预测，对比才公平。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, trust_remote_code=True, torch_dtype=torch.float32)
    has_lora = False
    if adapter_dir and os.path.isdir(adapter_dir):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        has_lora = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[5-device] 使用 {device}")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model, has_lora


def generate(tok, model, prompt: str, max_new: int = 48,
             disable_adapter: bool = False) -> str:
    """disable_adapter=True 时临时摘掉 LoRA，等价于「训练前的 base 模型」。"""
    import torch
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        next(model.parameters()).device)
    pad = tok.pad_token_id or tok.eos_token_id
    if disable_adapter and hasattr(model, "disable_adapter"):
        with torch.no_grad(), model.disable_adapter():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=pad)
    else:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=pad)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def _font(size: int):
    """加载中文字体用于绘制信息面板。

    坑：默认 PIL 位图字体只支持 ASCII，信息面板里的中文会全成方块。
    用 SYSTEMROOT 而不是硬编码 C:\\Windows，否则在非默认系统盘/容器里就跪。
    """
    win_fonts = os.path.join(os.environ.get("SYSTEMROOT") or r"C:\Windows", "Fonts")
    # 优先用微软雅黑；都没有再退化到 arial（中文仍会丢，但至少英文能看）
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf", "arial.ttf"):
        p = os.path.join(win_fonts, name)
        if not os.path.exists(p):
            continue
        try:
            return ImageFont.truetype(p, size)
        except Exception as ex:
            print(f"  [font] 加载 {p} 失败: {type(ex).__name__}: {ex}")
    return ImageFont.load_default()


def draw_result(shot_path: str, goal: str, pred_base, pred_lora, truth_el,
                resp_base: str, resp_lora: str, out_path: str):
    """三色同图：蓝=训练前，红=训练后，绿=真值。"""
    img = Image.open(shot_path).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(max(14, H // 60))

    def circle(pt, color, r, w):
        if not pt:
            return
        x, y = pt
        d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=w)
        d.line([x - r - 8, y, x + r + 8, y], fill=color, width=max(1, w // 3))
        d.line([x, y - r - 8, x, y + r + 8], fill=color, width=max(1, w // 3))

    tb = truth_el.bbox if truth_el is not None else None
    if tb and len(tb) == 4:                      # 真值元素框
        d.rectangle(list(tb), outline=(0, 220, 0, 220), width=3)
    circle(pred_base, (60, 130, 255, 255), max(16, H // 60), 4)   # 蓝：训练前
    circle(pred_lora, (255, 60, 60, 255), max(20, H // 50), 5)    # 红：训练后
    if truth_el is not None:
        circle(tuple(int(v) for v in truth_el.center()), (0, 230, 0, 255),
               max(12, H // 75), 4)                                # 绿：真值

    # 信息面板
    lines = [
        f"目标 goal: {goal}",
        f"蓝圈 = 训练前(base)  -> {pred_base}",
        f"红圈 = 训练后(+LoRA) -> {pred_lora}",
        f"绿圈/绿框 = 真实元素中心 -> "
        + (f"{tuple(int(v) for v in truth_el.center())}" if truth_el else "无"),
    ]
    if truth_el is not None:
        lines.append(
            f"误差: 训练前 {dist(pred_base, truth_el.center()):.0f}px  |  "
            f"训练后 {dist(pred_lora, truth_el.center()):.0f}px")
        lines.append(
            f"命中(预测落在元素框内): 训练前 "
            f"{'是' if in_bbox(pred_base, tb) else '否'}  |  训练后 "
            f"{'是' if in_bbox(pred_lora, tb) else '否'}")
    lines.append(f"训练前原始输出: {resp_base.strip()[:60]}")
    lines.append(f"训练后原始输出: {resp_lora.strip()[:60]}")

    pad = 10
    lh = f.size + 6 if hasattr(f, "size") else 22
    box_w = max(d.textlength(s, font=f) for s in lines) + pad * 2
    box_h = lh * len(lines) + pad * 2
    d.rectangle([6, 6, 6 + box_w, 6 + box_h], fill=(0, 0, 0, 170))
    for i, s in enumerate(lines):
        d.text((6 + pad, 6 + pad + i * lh), s, font=f,
               fill=(255, 255, 255, 255))
    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="", help="要定位的目标名（默认从屏幕自动挑）")
    ap.add_argument("--shot", default="", help="指定已有截图；不传则实时截屏")
    ap.add_argument("--max-elements", type=int, default=DEFAULT_MAX_ELEMENTS,
                    help=f"进 prompt 的元素数（必须与训练时同源，默认 "
                         f"{DEFAULT_MAX_ELEMENTS}）")
    ap.add_argument("--ocr-scale", type=float, default=0.5,
                    help="OCR 缩放，0.5 把全图 OCR 从 ~18s 降到 ~5s")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--out", default="", help="输出图路径（默认 blobs/demo_<ts>.png）")
    ap.add_argument("--no-lora", action="store_true", help="不挂 LoRA（只跑 base）")
    ap.add_argument("--list", action="store_true",
                    help="只列出屏幕上可作目标的带文字元素，不跑模型（用来挑 --goal）")
    args = ap.parse_args()

    print("=== MGA ③ 大脑定位 demo ===\n")

    # 1) 截图
    t0 = time.time()
    if args.shot:
        shot = args.shot
        print(f"[1] 使用已有截图: {shot}")
    else:
        cu = ComputerUse(SafetyGuard(dry_run=True, require_confirm=False,
                                     allowed_region=(0, 0, 1920, 1080)))
        res = cu.screenshot()
        # 坑：ComputerUse.screenshot() 走的是 pyautogui 语义——传了 path 会存盘，
        # 但**返回值仍是 PIL.Image 而不是路径**。检测与绘图统一吃路径，
        # 故这里自己落盘，免得后面把 Image 当路径用（os.path.exists 直接失败）。
        if isinstance(res, Image.Image):
            os.makedirs(os.path.join("blobs", "shots"), exist_ok=True)
            shot = os.path.join("blobs", "shots", f"demo_{int(time.time())}.png")
            res.save(shot)
        else:
            shot = str(res)
        print(f"[1] 已截取当前屏幕: {shot}")
    if not os.path.exists(str(shot)):
        print(f"!! 截图不存在: {shot}")
        return 1

    # 2) 检测元素（ScreenParser + OCR 补文字，与采集时同一套）
    locator = TriLevelLocator(visual=ScreenParserBackend(),
                              ocr=OCRBackend(scale=args.ocr_scale),
                              uia=UIALocator())
    els = locator.detect(shot)
    with_text = sum(1 for e in els if getattr(e, "text", None))
    print(f"[2] 检测到 {len(els)} 个元素（{with_text} 个带 OCR 文字）"
          f"  耗时 {time.time()-t0:.1f}s")
    if not els:
        print("!! 没检测到任何元素，无法 demo。")
        return 1

    if args.list:
        cands = goal_candidates(els)
        cands.sort(key=lambda x: (x[2] == 0, x[0] not in CLICKABLE_LABELS, len(x[1])))
        if not cands:
            print("!! 没有可作目标的带文字元素（多半是 OCR 未装，或全是噪声短串）。")
            return 1
        print(f"\n可作目标的带文字元素（共 {len(cands)} 个，按推荐度排序）：")
        for i, (lab, t, n_cjk) in enumerate(cands[:30], 1):
            print(f"  {i:2d}. [{lab}] {t!r}")
        print("\n挑一个，用 --goal 'xxx' 再跑。")
        return 0

    # 3) 目标
    goal = args.goal or pick_auto_goal(els)
    if not goal:
        print("!! 屏幕上没找到可作目标的带文字元素，用 --goal 手动指定。")
        return 1
    print(f"[3] 目标 goal = {goal!r}")

    # 4) 构造 prompt（与训练/采集同源，这是能否复现训练效果的关键）
    scene = format_scene(els, goal, max_elements=args.max_elements)
    prompt = build_prompt(scene, goal)
    print(f"[4] prompt 已构造（同源校验：max_elements={args.max_elements}）")

    # 5) 加载大脑
    tok, model, has_lora = load_brain(
        args.base, None if args.no_lora else args.adapter)
    print(f"[5] 大脑已加载 base={args.base}  "
          f"LoRA={'on' if has_lora else 'off'}")

    # 6) 两次推理：训练前 vs 训练后
    resp_lora = generate(tok, model, prompt)
    resp_base = (generate(tok, model, prompt, disable_adapter=True)
                 if has_lora else "(未挂 LoRA，跳过对比)")
    pred_lora = coords_of(resp_lora)
    pred_base = coords_of(resp_base) if has_lora else None
    print(f"[6] 训练前输出: {resp_base.strip()[:120]}")
    print(f"    训练后输出: {resp_lora.strip()[:120]}")

    # 7) 真值与判定
    truth_el = find_ground_truth(els, goal)
    out = args.out or os.path.join("blobs", f"demo_{int(time.time())}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    draw_result(shot, goal, pred_base, pred_lora, truth_el,
                resp_base, resp_lora, out)

    print("\n=== 结果 ===")
    if truth_el is None:
        print("（该 goal 在检测结果里没有真值元素，仅输出模型预测，无法判定命中）")
    else:
        tc = tuple(int(v) for v in truth_el.center())
        lbl = getattr(truth_el, "text", None) or truth_el.label
        print(f"真值元素: {lbl}  中心={tc}  bbox={truth_el.bbox}")
        if has_lora:
            print(f"训练前预测 {pred_base}  误差 {dist(pred_base, tc):.0f}px  "
                  f"命中={'是' if in_bbox(pred_base, truth_el.bbox) else '否'}")
        print(f"训练后预测 {pred_lora}  误差 {dist(pred_lora, tc):.0f}px  "
              f"命中={'是' if in_bbox(pred_lora, truth_el.bbox) else '否'}")
    print(f"\n可视化已保存: {out}")
    print("（蓝=训练前，红=训练后，绿=真实答案；三者重合即模型真的学会了定位）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
