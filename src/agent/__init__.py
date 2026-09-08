"""
agent/ —— 基于 computer-use 范式的「分层决策智能体」
=====================================================
目标（两阶段）：
    Phase 1（已完成可跑）：chrome://dino 小恐龙 —— 简单、可观测、动作集小，用于打通闭环
    Phase 2（扩展位已留）：GTA5 —— 复杂、部分可观测、连续控制 + 长任务

双系统架构（模拟真实大脑）：
    ┌ 感知层  ── 三级降级：ScreenParser(UI) + YOLOE(开放词汇) + CV 几何兜底 ──┐
    │                                                                        │
    │  System1 小模型（快·经验性）      ←→      经验记忆（embedding 检索库）   │
    │   · 物理基线：MotionModel 纯数学 ETA                                     │
    │   · 习得残差：LearnedResidual —— System2 的纠正回流训练它（知识蒸馏）     │
    │   · 经验检索：命中历史经验则直接快决策（像"本能"）                        │
    │   · 逐帧运行、µs~ms 级、零 LLM                                           │
    │                                                                        │
    │  System2 LLM 大脑（慢·思考）                                             │
    │   · 仅在「新情形 / 低置信」关键帧唤醒                                     │
    │   · 快速推理 + 情景规划                                                  │
    │   · 把经验与记忆**持久化写入存储**（memory_system，sqlite）               │
    │   · 纠正回流训练 System1（像皮层固化 → 基底节习惯）                       │
    │                                                                        │
    └ 执行层 ── MGA ⑥ exec：WindowManager 聚焦 + SendInput 原生注入 ─────────┘

LLM 主导宏观决策：System2 是 LLM；System1 是小模型（习得残差 + 经验检索），
两者协同形成「感知 → 决策 → 执行 → 校验 → 学习」的完整闭环。

模块：
    types.py         公共数据结构（Scene / Decision / AgentStats）
    memory.py        经验记忆持久化（包 memory_system.MemorySystem）
    system1.py       System1 小模型（物理基线 + 习得残差 + 经验检索）
    system2.py       System2 LLM 大脑（推理 + 情景规划 + 写记忆）
    executor.py      执行层（聚焦 + 按键注入 + 回执自检）
    game_adapter.py  游戏适配器（DinoAdapter 已实现 / GTA5Adapter 为 Phase2 占位）
    orchestrator.py  编排器（S1 逐帧 → 升级 S2 → 执行 → 校验 → 学习）

设计原则（与全系统一致）：
  - 零硬依赖：任何外部模块缺失即降级，绝不崩（满血运行 = 有则用、无则退）。
  - 架构真相：S2 是否真调 LLM 会在 stats 里如实统计（n_llm / n_heuristic）。
"""

__all__ = [
    "types", "memory", "system1", "system2",
    "executor", "game_adapter", "orchestrator",
]

PHASE = {
    1: "chrome://dino 小恐龙（简单：完全可观测 / 离散动作 / 稠密奖励）",
    2: "GTA5（复杂：部分可观测 / 连续控制 / 稀疏奖励 / 长任务规划）",
}
