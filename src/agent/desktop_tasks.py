"""
desktop_tasks.py —— M4 三类非高危桌面任务模板
============================================================================

复用：感知(desktop_perceive) + 执行(ComputerUse) + 意图门禁(IntentGate)。
不重造执行层；模板只负责把「一类任务」编译成**可执行的步骤序列(Plan)**，
供 S2 编排层 / DesktopAdapter.act() 消费。

三类（均落在用户拍板的「非高危重复任务」范围内）：
  · FormFillTemplate      填表：聚焦窗口 → 逐字段点击/输入
  · CrossSystemTransfer  跨系统搬运：复制(ctrl+c) → 粘贴(ctrl+v)，**不做发送**
  · FileOrganizeTemplate 文件整理：感知文件 → 移动到目标目录（**不做删除**）

边界硬约束（架构真相，模板编译期即生效，不是运行时才拦）：
  · 每步经 IntentGate 过语义闸；含 删除/发送 → 整份 Plan 编译失败(BLOCK)，
    连准备步骤都不产出；含 支付 → 该步标 confirm=True，需真人 checkpoint。
  · 模板自身也只生成 read/write 动作，绝不生成 delete/send 原语。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SAFETY = os.path.join(_ROOT, "safety")
for _p in (_ROOT, _HERE, _SAFETY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from safety.intent_gate import IntentGate


@dataclass
class Step:
    """一个可执行步骤（喂 DesktopAdapter.act 的紧凑命令 + 元信息）。"""
    action: str                 # 紧凑命令，如 "click 530 320" / "type 你好"
    note: str                   # 人类可读 + 喂 IntentGate 的语义文本
    risk: str = "write"         # read / write
    confirm: bool = False       # True=需人工确认（支付等）


class TaskTemplate:
    """任务模板基类：build(spec) -> List[Step]，编译期过意图闸。"""

    name = "base"

    def __init__(self):
        self.gate = IntentGate()

    def build(self, spec: dict) -> List[Step]:
        raise NotImplementedError

    def _screen(self, steps: List[Step]) -> List[Step]:
        """过意图闸：BLOCK 整份拒绝；NEED_CONFIRM 标 confirm。"""
        out: List[Step] = []
        for s in steps:
            d = self.gate.gate(s.note)
            if d.blocked():
                raise ValueError(
                    f"[{self.name}] 任务含高危意图被拦截，整份 Plan 不产出："
                    f"{s.note}（{d.reason}）")
            out.append(Step(s.action, s.note, s.risk, confirm=d.need_confirm()))
        return out

    def summarize(self, steps: List[Step]) -> str:
        lines = [f"# {self.name} 任务 Plan（{len(steps)} 步）"]
        for i, s in enumerate(steps, 1):
            tag = " [需确认]" if s.confirm else ""
            lines.append(f"  {i:2d}. {s.action:24s} | {s.note}{tag}")
        return "\n".join(lines)


class FormFillTemplate(TaskTemplate):
    """填表：聚焦目标窗口 → 逐字段点击(可选坐标) + 输入值。"""

    name = "填表"

    def build(self, spec: dict) -> List[Step]:
        title = spec["window_title"]
        fields = spec["fields"]          # [{label, value, x?, y?}]
        steps = [Step(f"focus {title}", f"聚焦窗口 {title}", "write")]
        for f in fields:
            if "x" in f and "y" in f:
                steps.append(Step(f"click {f['x']} {f['y']}",
                                  f"点击字段「{f['label']}」", "write"))
            else:
                steps.append(Step(f"focus {title}",
                                  f"聚焦后定位字段「{f['label']}」", "write"))
            steps.append(Step(f"type {f['value']}",
                              f"填入「{f['label']}」={f['value']}", "write"))
        return self._screen(steps)


class CrossSystemTransferTemplate(TaskTemplate):
    """跨系统搬运：来源复制(ctrl+c) → 目标粘贴(ctrl+v)，不做发送。"""

    name = "跨系统搬运"

    def build(self, spec: dict) -> List[Step]:
        src = spec["source_window"]
        tgt = spec["target_window"]
        what = spec.get("what", "选中内容")
        steps = [
            Step(f"focus {src}", f"聚焦来源窗口 {src}", "write"),
            Step("hotkey ctrl a", f"全选 {what}", "write"),
            Step("hotkey ctrl c", f"复制 {what}（跨系统搬运，不发送）", "write"),
            Step(f"focus {tgt}", f"聚焦目标窗口 {tgt}", "write"),
            Step("hotkey ctrl v", f"粘贴到 {tgt}", "write"),
        ]
        return self._screen(steps)


class FileOrganizeTemplate(TaskTemplate):
    """文件整理：感知文件 → 移动到目标目录（移动，非删除）。"""

    name = "文件整理"

    def build(self, spec: dict) -> List[Step]:
        folder = spec["folder"]
        rules = spec["rules"]            # [{pattern, dest}]
        steps = [
            Step(f"focus {folder}", f"聚焦文件夹 {folder}", "read"),
            Step("list_windows", f"列出 {folder} 文件（感知）", "read"),
        ]
        for r in rules:
            steps.append(Step(f"focus {folder}", f"聚焦 {folder}", "write"))
            steps.append(Step(f"type {r['pattern']}",
                              f"选中匹配「{r['pattern']}」的文件", "write"))
            steps.append(Step(f"focus {r['dest']}",
                              f"聚焦目标目录 {r['dest']}", "write"))
            steps.append(Step("hotkey ctrl v",
                              f"移动到 {r['dest']}（整理，非删除）", "write"))
        return self._screen(steps)


# 注册表（S2 按名取模板）
TEMPLATES = {
    "form_fill": FormFillTemplate,
    "cross_transfer": CrossSystemTransferTemplate,
    "file_organize": FileOrganizeTemplate,
}


def build_task(kind: str, spec: dict) -> List[Step]:
    if kind not in TEMPLATES:
        raise ValueError(f"未知任务类型: {kind}（可选 {list(TEMPLATES)}）")
    return TEMPLATES[kind]().build(spec)


if __name__ == "__main__":
    # 沙箱自检
    ff = FormFillTemplate().build({
        "window_title": "报销单",
        "fields": [{"label": "姓名", "value": "张三", "x": 200, "y": 120},
                   {"label": "金额", "value": "88.5"}]})
    print(FormFillTemplate().summarize(ff))
    ct = CrossSystemTransferTemplate().build({
        "source_window": "Excel", "target_window": "邮件", "what": "表格"})
    print("\n" + CrossSystemTransferTemplate().summarize(ct))
    fo = FileOrganizeTemplate().build({
        "folder": "下载", "rules": [{"pattern": "*.pdf", "dest": "文档/PDF"}]})
    print("\n" + FileOrganizeTemplate().summarize(fo))
    print("\nM4 模板自检：三类 Plan 编译成功，无 delete/send 原语")
