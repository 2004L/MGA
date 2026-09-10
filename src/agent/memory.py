"""
memory.py —— 经验与记忆持久化层（System2 写 / System1 读）
==========================================================
对应架构图里的「经验记忆」：
    System2 大脑思考后 → write() 把经验落库（持久化存储，跨会话不丢）
    System1 小模型每帧 → retrieve() 检索相似历史经验，直接快决策（像"本能"）
    结果不及预期        → challenge(success=False) 把该条经验降权（反经验主义）

底层直接复用 src/memory_system.py（stdlib + sqlite，零框架污染）：
    · 评分 = 频次强度 × 时间衰减 × 重要性（类艾宾浩斯遗忘曲线）
    · validity / contested：失败纠正会降权，避免"经验主义"把错误习惯固化
    · embedding 可插拔（默认 HashEmbedder，生产换真实句向量即可）

零依赖兜底：memory_system 不可用时退化为 _NullMemory（全 no-op），闭环照跑不崩。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class _NullMemory:
    """记忆系统不可用时的空实现（保证闭环不崩）。"""

    def write(self, key: str, summary: str, tags: Optional[List[str]] = None,
              ref: str = "") -> str:
        return ""

    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        return []

    def challenge(self, memory_id: str, success: bool) -> None:
        return None

    def recent(self, limit: int = 1) -> List[Dict[str, Any]]:
        return []

    def stats(self) -> Dict[str, Any]:
        return {"available": False}


class ExperienceMemory:
    """经验记忆：包一层 MemorySystem，提供面向「情形键」的写/读/纠正接口。"""

    def __init__(self, db_path: str = "blobs/agent_memory.db", verbose: bool = False):
        self.db_path = db_path
        self.verbose = verbose
        self.available = False
        self._ms = None
        self._last_id: Optional[str] = None
        self._recent: List[Dict[str, Any]] = []   # 最近写入的经验（纠错时回看，避免依赖底层列举）
        try:
            d = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(d, exist_ok=True)
            from memory_system import MemorySystem
            self._ms = MemorySystem(db_path)
            self.available = True
            if verbose:
                print(f"  [记忆] 经验库已挂载：{db_path}")
        except Exception as e:
            print(f"  [记忆] MemorySystem 不可用，降级为无记忆模式：{type(e).__name__}: {e}")
            self._ms = _NullMemory()

    # ---- System2 写：把这次思考的结论固化成经验 --------------------------
    def write(self, key: str, summary: str, tags: Optional[List[str]] = None,
              ref: str = "") -> str:
        """System2 写入一条经验。key=情形键(用于检索)，summary=决策与理由。"""
        try:
            mid = self._ms.write_memory(
                title=key, summary=summary,
                file_pointer=ref or f"agent://experience/{key}",
                tags=list(tags or ["game", "dino"]),
            )
            self._last_id = mid
            if mid:
                self._recent.append({"id": mid, "summary": summary, "key": key})
                if len(self._recent) > 30:
                    self._recent = self._recent[-30:]
            return mid or ""
        except Exception as e:
            if self.verbose:
                print(f"  [记忆] 写入失败：{type(e).__name__}: {e}")
            return ""

    # ---- System1 读：检索历史经验，命中即快决策 ---------------------------
    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        try:
            return list(self._ms.search_memory(query, top_k=top_k) or [])
        except Exception as e:
            if self.verbose:
                print(f"  [记忆] 检索失败：{type(e).__name__}: {e}")
            return []

    # ---- 结果反馈：失败则把该经验降权（反经验主义）------------------------
    def challenge(self, memory_id: str, success: bool) -> None:
        if not memory_id:
            return
        try:
            self._ms.challenge_memory(memory_id, success)
        except Exception as e:
            # 不再静默：挑战(降权)失败 = 错误经验不会被降权，会一直被复用，
            # 表现为"老犯同一个错"却看不出记忆没更新
            print(f"  [memory] ⚠️ 经验降权失败(id={memory_id})：{type(e).__name__}: {e}")

    def stats(self) -> Dict[str, Any]:
        try:
            s = self._ms.stats() or {}
            s["available"] = True
            return s
        except Exception:
            return {"available": False}

    @property
    def last_id(self) -> Optional[str]:
        return self._last_id

    # ---- 纠错用：回看最近写入的经验（如撞死时 System2 要诊断"哪条经验带偏了"）----
    def recent(self, limit: int = 1) -> List[Dict[str, Any]]:
        if limit <= 0:
            return list(self._recent)
        return list(self._recent[-limit:])
