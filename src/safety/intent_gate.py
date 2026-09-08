"""
intent_gate.py —— 意图级安全门禁（本期新增，架在 ComputerUse.SafetyGuard 之前）
================================================================================

为什么需要这一层：
  ComputerUse.SafetyGuard 只到「动作级」闸（region / 危险文本 / 急停），
  它不知道"这个人想干嘛"。而用户硬约束是**语义级**的：
    · 删除/发送 → 出自动化范围（连准备都不做）
    · 支付     → 可准备但必须人工确认点
  所以 S2 规划出 PlanStep 时，先过 IntentGate 做语义分类，再决定是否放行给
  ComputerUse。这是从外到内的第 1 道闸（IntentGate → SafetyGuard → 急停）。

决策三态：
  BLOCK        : delete（永远出范围，直接拒绝，连准备都不做）
  NEED_CONFIRM : payment（生成待确认摘要，等真人 checkpoint 才放行）
  ALLOW        : send（默认出范围 BLOCK；仅当显式 allow_send=True 且范围受控时 ALLOW）
                 / 其余正常任务（再交 SafetyGuard 物理闸）

红线放松说明（架构真相，不偷偷改）：
  · DELETE / 支付 的物理红线**永不放松**（删除永远 BLOCK、支付永远 NEED_CONFIRM）。
  · SEND 仅在调用方**显式**传 allow_send=True 时放行——而这必须配合"范围受控"
    （单一预授权联系人、换人重问、真人鼠标活动即取消）才有意义。allow_send
    默认 False，即 SEND 默认仍 BLOCK。这是用户 2026-09-08 拍板的有意识放宽，
    不是默认行为。放行的 send 在 reason 里标 authorized-send 便于审计。

架构真相：本层只做"分类 + 拦截/确认"，不执行任何键鼠。确认动作由外部
UI/编排层回调 confirm() 完成，避免 Agent 自己"点了支付"。
"""

from __future__ import annotations

import os
import sys

# 路径兜底（同 desktop_perceive）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    NORMAL = "normal"
    DELETE = "delete"          # 删除/清空/rm/trash —— 出自动化范围
    SEND = "send"              # 发送消息/邮件/群发 —— 出自动化范围
    PAYMENT = "payment"        # 支付/付款/转账 —— 需人工确认点


# 语义关键词（小写匹配；可随业务扩充）
# 注意：删除/发送 出自动化范围，关键词宁可多覆盖（保守），漏判比误拦更危险。
_DELETE_KW = ("删除", "清空", "彻底删除", "discard", "delete", "rm -rf",
              "rm ", "trash", "回收站", "抹除", "移除", "注销", "销毁",
              "删号", "抹掉", "清掉")
_SEND_KW = ("发送", "群发", "发出", "submit form", "send", "发送消息",
            "发送邮件", "send email", "post", "发布", "发给", "转发",
            "转寄", "寄给", "推送", "分享", "共享")
_PAYMENT_KW = ("支付", "付款", "转账", "下单付款", "结算", "pay", "payment",
               "transfer", "微信支付", "支付宝", "checkout", "购买",
               "提交订单", "下单", "付费", "扣款", "续费")

# 否定标记（出现在关键词前 2 字内 → 该意图取消，处理「不发送/非删除/禁止…」）
_NEG = ("不", "别", "勿", "未", "无", "禁", "没", "非", "勿", "暂不")


@dataclass
class GateDecision:
    intent: Intent
    verdict: str                # BLOCK / NEED_CONFIRM / ALLOW
    reason: str
    summary: Optional[str] = None   # NEED_CONFIRM 时给真人的确认摘要
    pending_id: Optional[str] = None

    def blocked(self) -> bool:
        return self.verdict == "BLOCK"

    def need_confirm(self) -> bool:
        return self.verdict == "NEED_CONFIRM"


# 简易自增 id（确认回调用）
_counter = 0


def _next_id() -> str:
    global _counter
    _counter += 1
    return f"pending_{_counter}"


class IntentGate:
    """意图级安全门禁：分类 → 拦截/确认/放行。

    allow_send=False（默认）：SEND 与 DELETE 一样出范围，直接 BLOCK。
    allow_send=True（需调用方显式打开）：SEND 放行为 ALLOW，但 reason 标
    authorized-send 便于审计。DELETE / PAYMENT 无论 allow_send 如何都保持
    BLOCK / NEED_CONFIRM（红线永不放松）。
    """

    def __init__(self, allow_send: bool = False):
        self.allow_send = bool(allow_send)
        self._pending: dict = {}   # pending_id -> GateDecision（待真人确认）

    # ---- 分类：语义关键词 → Intent ----
    @staticmethod
    def _negated(t: str, kw: str) -> bool:
        """关键词前 2 字内是否带否定标记（不/别/勿/非/禁止…）。"""
        i = t.find(kw)
        if i < 0:
            return False
        window = t[max(0, i - 2):i]
        return any(n in window for n in _NEG)

    def classify(self, text: str) -> Intent:
        t = (text or "").lower()
        # 优先级：payment > send > delete（避免"删除并发送"这类复合句漏判高危）
        # 否定式（"不发送"/"非删除"/"禁止清空"）不触发对应意图。
        if any((k in t and not self._negated(t, k)) for k in _PAYMENT_KW):
            return Intent.PAYMENT
        if any((k in t and not self._negated(t, k)) for k in _SEND_KW):
            return Intent.SEND
        if any((k in t and not self._negated(t, k)) for k in _DELETE_KW):
            return Intent.DELETE
        return Intent.NORMAL

    # ---- 决策：分类 → 三态 ----
    def gate(self, step_text: str, context: str = "", allow_send: bool = None) -> GateDecision:
        intent = self.classify(step_text)
        eff_allow_send = self.allow_send if allow_send is None else bool(allow_send)
        if intent == Intent.DELETE:
            return GateDecision(
                intent=intent, verdict="BLOCK",
                reason="delete 物理红线：永远出自动化范围，连准备都不做，直接拒绝",
                summary=step_text)
        if intent == Intent.SEND:
            if eff_allow_send:
                return GateDecision(
                    intent=intent, verdict="ALLOW",
                    reason="authorized-send：SEND 在显式授权范围内放行（范围受控+审计）",
                    summary=step_text)
            return GateDecision(
                intent=intent, verdict="BLOCK",
                reason="send 出自动化范围：未授权，直接拒绝，连准备都不做",
                summary=step_text)
        if intent == Intent.PAYMENT:
            pid = _next_id()
            d = GateDecision(
                intent=intent, verdict="NEED_CONFIRM",
                reason="支付：可准备但必须人工确认点才注入",
                summary=f"[待确认] {step_text}\n上下文: {context}".strip(),
                pending_id=pid)
            self._pending[pid] = d
            return d
        return GateDecision(intent=Intent.NORMAL, verdict="ALLOW",
                            reason="非高危意图，放行（再交 SafetyGuard 物理闸）",
                            summary=step_text)

    # ---- 人工确认回调（外部 UI/编排层在真人点确认后调用）----
    def confirm(self, pending_id: str) -> bool:
        d = self._pending.pop(pending_id, None)
        return d is not None   # True=已确认放行；False=无此待确认（或已超时）

    def pending_count(self) -> int:
        return len(self._pending)


if __name__ == "__main__":
    # 沙箱自检：三类意图 + 正常，断言三态正确
    g = IntentGate()
    cases = [
        ("把下载文件夹里的大文件删除", "BLOCK"),
        ("给客户发送这封邮件", "BLOCK"),
        ("用微信支付付这笔订单", "NEED_CONFIRM"),
        ("在表单里填好姓名和电话", "ALLOW"),
        ("把桌面报告和表格整理到归档目录", "ALLOW"),
    ]
    ok = True
    for text, expect in cases:
        d = g.gate(text)
        status = "OK" if d.verdict == expect else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] {text!r:30s} → {d.intent.value:8s} {d.verdict}")
    print("意图门禁自检:", "全部通过" if ok else "存在失败")
