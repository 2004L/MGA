"""
uia_probe.py —— UIA 可行性探针（架构真相核验用）
=================================================
只回答一个问题：Windows UI Automation 到底能不能拿到**真实的、带准确文字的控件**，
而不是只有几个顶层窗口？这是决定 UIA 能否替代 OCR 当训练数据主信号源的前提。

背景：detector.UIALocator 原来只遍历 root.GetChildren()（=顶层窗口），
且 uiautomation 从未安装 → 这一级一直静默返回 []，从没贡献过元素。
所以训练数据的文字全是 OCR 噪声（'on'/'STS'/'L7'）。

用法：
  PYTHONPATH=src python -m uia_probe            # 默认 25s 预算
  PYTHONPATH=src python -m uia_probe --budget 40 --show 40
"""
import argparse
import sys
import time

try:
    import uiautomation as auto
except ImportError as ex:
    print("!! uiautomation 未安装:", ex)
    print("    pip install uiautomation comtypes")
    sys.exit(1)

MAX_DEPTH = 14
MAX_NODES = 6000
_deadline = 0.0


def walk(ctrl, depth, out, stats):
    if depth > MAX_DEPTH or len(out) >= MAX_NODES or time.time() > _deadline:
        return
    try:
        children = ctrl.GetChildren()
    except Exception:
        return
    for c in children:
        if time.time() > _deadline or len(out) >= MAX_NODES:
            return
        stats["visited"] += 1
        try:
            name = (c.Name or "").strip()
            rect = c.BoundingRectangle
            ctype = c.ControlTypeName or ""
        except Exception:
            continue
        if name and rect is not None:
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            # 过滤零面积控件与整屏容器（它们不是"可点击目标"）
            if 4 <= w <= 3000 and 4 <= h <= 2000:
                out.append((ctype, name,
                            (rect.left, rect.top, rect.right, rect.bottom)))
        walk(c, depth + 1, out, stats)


def main():
    global _deadline
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=25.0, help="遍历时间预算(秒)")
    ap.add_argument("--show", type=int, default=25, help="打印前 N 个控件")
    args = ap.parse_args()

    t0 = time.time()
    _deadline = t0 + args.budget

    try:
        root = auto.GetRootControl()
        tops = root.GetChildren()
    except Exception as ex:
        print(f"!! 拿不到根控件: {type(ex).__name__}: {ex}")
        return 1
    print(f"顶层窗口 {len(tops)} 个，开始递归遍历（预算 {args.budget:.0f}s）...")

    out, stats = [], {"visited": 0}
    for w in tops:
        if time.time() > _deadline or len(out) >= MAX_NODES:
            break
        walk(w, 0, out, stats)

    dt = time.time() - t0
    print(f"\n遍历节点 {stats['visited']} 个 → **有效带名控件 {len(out)} 个**"
          f"，耗时 {dt:.1f}s")
    if not out:
        print("!! 一个控件都没拿到，UIA 在当前环境不可用。")
        return 1

    print(f"\n前 {min(args.show, len(out))} 个真实控件：")
    print(f"  {'ControlType':<18}{'Name':<40}center")
    for ctype, name, r in out[:args.show]:
        cx, cy = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
        print(f"  [{ctype:<16}] {name[:36]!r:<40}({cx},{cy})")

    # 与 OCR 对比的关键指标：名字质量
    junk = sum(1 for _, n, _ in out if len(n.strip()) < 2)
    cjk = sum(1 for _, n, _ in out if any("\u4e00" <= ch <= "\u9fff" for ch in n))
    print(f"\n名字质量：中文名 {cjk} 个，长度<2 的疑似噪声 {junk} 个"
          f"（OCR 方案里噪声占比极高，这是本质差别）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
