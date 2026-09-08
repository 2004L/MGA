"""
demo_dino_agent.py —— Phase 1 入口：用「分层决策智能体」通关 chrome://dino
========================================================================
架构（computer-use 范式 + 双系统，模拟真实大脑）：

    感知三级降级(ScreenParser+YOLOE+CV)
        → System1 小模型逐帧快决策（物理基线 + 习得残差 + 经验检索）
        → 新情形升级 System2（LLM 快速推理 + 情景规划 + 经验写库 + 训练 S1）
        → 执行层（聚焦 + SendInput 注入 + 回执自检）
        → 死亡检测 → 自动重开（闭环永不断）

用法：
    # 真机（自动定位游戏区，CV 感知，LLM 用 mock 后端）
    python -m demo_dino_agent --frames 3000

    # 想用真 LLM 决策（需配置 .env 里的 API）：
    python -m demo_dino_agent --frames 3000 --llm-backend api

    # 沙箱自测（不碰真实屏幕/键盘）
    python -m demo_dino_agent --sim --frames 300

    # 指定区域 + 强制 CV + 打开标注窗口
    python -m demo_dino_agent --region 0,556,1000,180 --no-dl --show

观察重点（验证双系统是否真在工作）：
    · S1 占比应 >95%（它逐帧跑，是主力）
    · S2 只在"新情形"被唤醒；唤醒次数 = 学到的经验条数
    · 经验命中(exp_hits) 随运行增长 → 说明 System2 的经验已固化成 System1 的本能
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from agent.game_adapter import DinoAdapter
from agent.memory import ExperienceMemory
from agent.orchestrator import AgentOrchestrator
from agent.system1 import System1
from agent.system2 import System2
from agent.executor import Executor


def _parse_region(s: str):
    if not s:
        return None
    try:
        return tuple(int(v) for v in s.split(","))
    except Exception:
        print(f"  [参数] --region 格式应为 x,y,w,h，忽略：{s}")
        return None


def main():
    ap = argparse.ArgumentParser(
        description="分层决策智能体 Phase1：小恐龙（双系统 + 经验记忆）")
    ap.add_argument("--frames", type=int, default=3000, help="总帧数")
    ap.add_argument("--region", type=str, default="", help="游戏区 x,y,w,h（空=自动定位）")
    ap.add_argument("--no-dl", action="store_true", help="强制 CV 兜底（关闭 YOLOE 主通道）")
    ap.add_argument("--sim", action="store_true", help="合成截图自测（不碰真机）")
    ap.add_argument("--llm-backend", default="reasoner",
                    choices=["mock", "oracle", "reasoner", "api"],
                    help="System2 大脑后端：reasoner(本地推理替身,离线演示)/mock(占位)/"
                         "oracle(规则真身)/api(真·大模型,需配置密钥)")
    ap.add_argument("--llm-model", default=None, help="LLM 模型名（api 后端用）")
    ap.add_argument("--memory-db", default="blobs/agent_memory.db", help="经验记忆库路径")
    ap.add_argument("--reset-memory", action="store_true",
                    help="启动前清空经验库（演示 System2 从零学习 → 经验固化成 System1 本能）")
    ap.add_argument("--reaction", type=float, default=0.22, help="System1 物理基线触发阈值(秒)")
    ap.add_argument("--airtime", type=float, default=0.6, help="滞空估计(秒)，空中抑制窗")
    ap.add_argument("--s2-horizon", type=float, default=1.2,
                    help="升级 System2 的时间窗口：ETA 小于该值才来得及问")
    ap.add_argument("--s2-override", action="store_true",
                    help="允许 System2 直接覆盖当帧动作（默认关，安全优先）")
    ap.add_argument("--no-restart", action="store_true", help="撞死后不自动重开")
    ap.add_argument("--show", action="store_true", help="显示标注窗口（需 cv2）")
    args = ap.parse_args()

    print("=" * 72)
    print("  分层决策智能体 · Phase 1：chrome://dino")
    print("  感知三级降级 → System1 小模型(逐帧) ⇄ System2 LLM 大脑(关键帧) → 执行")
    print("  经验记忆：System2 写 / System1 读 / 失败降权（反经验主义）")
    print("=" * 72)

    # ---- 组装智能体 ----
    if args.reset_memory and os.path.exists(args.memory_db):
        try:
            os.remove(args.memory_db)
            print(f"  [记忆] 已清空经验库：{args.memory_db}（System2 将从零学习）")
        except Exception as e:
            print(f"  [记忆] 清空失败：{e}")
    memory = ExperienceMemory(args.memory_db, verbose=True)
    s1 = System1(dino_x=44.0, reaction=args.reaction, airtime=args.airtime,
                 memory=memory, s2_horizon=args.s2_horizon)
    s2 = System2(memory=memory, backend=args.llm_backend, model=args.llm_model,
                 verbose=True)
    executor = Executor(dry=args.sim, focus_title="Chrome")
    adapter = DinoAdapter(region=_parse_region(args.region),
                          use_dl=(not args.no_dl), sim=args.sim,
                          executor=executor)
    if adapter.region is None:
        adapter.locate()
    if adapter.region is None:
        print("  [致命] 找不到游戏区，请先打开 chrome://dino 或用 --region 指定")
        sys.exit(1)
    print(f"  游戏区={adapter.region}  sim={args.sim}  "
          f"感知={'CV兜底' if args.no_dl else 'YOLOE主通道+CV兜底'}  "
          f"LLM后端={args.llm_backend}")

    orch = AgentOrchestrator(adapter, s1, s2, executor, memory=memory,
                             auto_restart=(not args.no_restart),
                             s2_override=args.s2_override, verbose=True)
    if args.show:
        orch._show = True

    # ---- 跑 ----
    try:
        stats = orch.run(args.frames)
    except KeyboardInterrupt:
        stats = orch.stats
        stats.s2_llm = getattr(s2, "n_llm", 0)
        stats.s2_heuristic = getattr(s2, "n_heuristic", 0)
        stats.s2_reasoner = getattr(s2, "n_reasoner", 0)
        stats.suppress = s1.n_suppress
        print("\n  [中断] 已停止")

    # ---- 报告 ----
    print()
    print("=" * 72)
    print("  闭环报告")
    print("=" * 72)
    print(f"  {stats.report()}")
    print(f"  System1 习得残差样本数={s1.n_samples}  "
          f"(System2 的纠正已回流训练 S1 的次数)")
    try:
        ms = memory.stats()
        print(f"  经验库：{ms}")
    except Exception:
        pass
    if getattr(s2, "n_llm", 0):
        print(f"  ✅ 本局 System2 真正调用了大模型：{s2.n_llm} 次决策 + "
              f"{s2.n_plans} 次宏观调控 + {s2.n_correct} 次及时纠错")
    elif getattr(s2, "n_reasoner", 0):
        print(f"  ℹ️ 本局 System2 用本地 reasoner 替身跑通完整推理/宏观调控/纠错闭环"
              f"（{s2.n_reasoner} 次决策 + {s2.n_plans} 次宏观调控 + {s2.n_correct} 次纠错）；"
              f"接真大模型用 --llm-backend api")
    else:
        print("  ⚠️ 架构真相：本局 System2 走启发式兜底（无 reasoner/LLM）；"
              f"用 --llm-backend reasoner 或 api 启用推理")
    print("=" * 72)


if __name__ == "__main__":
    main()
