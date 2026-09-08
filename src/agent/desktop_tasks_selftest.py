"""
desktop_tasks_selftest.py —— M4 任务模板沙箱自检（不碰桌面）
============================================================================
验证：① 三类模板编译出正确步骤  ② 绝不生成 delete/send 原语
③ 边界：含 删除/发送 的 spec 编译失败(BLOCK) ④ 含 支付 的步骤标 confirm。
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SAFETY = os.path.join(_ROOT, "safety")
for _p in (_ROOT, _HERE, _SAFETY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from desktop_tasks import (FormFillTemplate, CrossSystemTransferTemplate,
                            FileOrganizeTemplate, build_task)

_FAILS = []


def _check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)


# 模板只应生成这些动作原语（绝不含 delete/send 类）
_ALLOWED_ACTIONS = ("focus", "click", "double_click", "right_click", "move",
                    "type", "drag", "scroll", "press", "hotkey", "list_windows",
                    "screenshot")


def _only_safe_actions(steps):
    """步骤的动作命令是否全在白名单（模板从不生成 delete/send 原语）。"""
    return all(s.action.split()[0] in _ALLOWED_ACTIONS for s in steps)


def main():
    print("=" * 60)
    print("M4 任务模板自检")
    print("=" * 60)

    # ① 填表
    ff = FormFillTemplate().build({
        "window_title": "报销单",
        "fields": [{"label": "姓名", "value": "张三", "x": 200, "y": 120},
                   {"label": "金额", "value": "88.5"}]})
    # 步骤 = 初始聚焦 + 字段1(点击+输入) + 字段2(聚焦+输入) = 5
    _check("填表 步数=5（聚焦+2字段×2步）", len(ff) == 5, f"len={len(ff)}")
    _check("填表 无危险原语", _only_safe_actions(ff))
    _check("填表 首步聚焦", ff[0].action.startswith("focus"))

    # ② 跨系统搬运
    ct = CrossSystemTransferTemplate().build({
        "source_window": "Excel", "target_window": "邮件", "what": "表格"})
    _check("搬运 含复制", any("ctrl c" in s.action for s in ct))
    _check("搬运 含粘贴", any("ctrl v" in s.action for s in ct))
    _check("搬运 无发送原语", _only_safe_actions(ct))

    # ③ 文件整理
    fo = FileOrganizeTemplate().build({
        "folder": "下载", "rules": [{"pattern": "*.pdf", "dest": "文档/PDF"}]})
    _check("整理 动作均为移动(非删除)",
           all("ctrl v" in s.action or s.action.startswith(("focus", "list", "type"))
               for s in fo))
    _check("整理 无删除原语", _only_safe_actions(fo))

    # ④ 边界：含删除意图 → 编译失败
    try:
        FormFillTemplate().build({
            "window_title": "工具",
            "fields": [{"label": "备注", "value": "删除旧数据后写入"}]})
        _check("边界 删除意图被拦截", False, "未抛错=漏拦")
    except ValueError as e:
        _check("边界 删除意图被拦截", True, f"已拒：{str(e)[:30]}...")

    # ⑤ 边界：含支付意图 → 该步标 confirm
    pay = FormFillTemplate().build({
        "window_title": "订单",
        "fields": [{"label": "支付金额", "value": "99"}]})
    _check("边界 支付步标 confirm",
           any(s.confirm for s in pay), "支付步未标需确认")

    # ⑥ build_task 工厂
    try:
        build_task("form_fill", {"window_title": "x", "fields": []})
        _check("build_task 工厂可用", True)
    except Exception as e:
        _check("build_task 工厂可用", False, f"{type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    if _FAILS:
        print(f"M4 自检：FAIL（{_FAILS}）")
        print("=" * 60)
        return 1
    print("M4 任务模板自检：✅ 全过（三类 Plan 正确 / 无危险原语 / 边界生效）")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
