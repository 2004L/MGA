"""
memory_system.py
================
MGA ④ 知识层 —— 经验记忆系统（零依赖实现，仅用 Python 标准库）

设计原则（学泽定）：
- 不引入任何第三方框架，全部 stdlib。
- embedding 通过可插拔 Embedder 注入；默认提供零依赖的 HashEmbedder 供演示。
- 评分 = 频次强度 × 时间衰减 × 重要性（类艾宾浩斯遗忘曲线）。
- 支持：频率衰减、归档回捞、经验主义(validity/contested)、token 经济联动。

对应架构：
    ④ 知识层
    └── 经验记忆（本文件）
        ├── 频率衰减   = 经验主义（常用留，闲置淡）
        ├── validity   = 避免经验主义（失败降权，接 System2 纠正）
        ├── 归档回捞   = 长期记忆冷存储
        └── token 联动 = 记忆省钱 = 生存（接 economy/token.py）

用法：
    ms = MemorySystem("memory.db")
    ms.write_memory("Boss战闪避", "贴脸绕后", "obsidian/boss.md", tags=["游戏"])
    hits = ms.search_memory("怎么打Boss")
    ms.challenge_memory(hits[0]["memory_id"], success=False)  # System2 纠正回调
生产环境把 Embedder 换成真实句子编码器即可，其余代码不动。
"""
from __future__ import annotations

import sqlite3
import json
import hashlib
import math
import os
import time
import uuid
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# 1. Embedder：可插拔语义向量生成器（默认零依赖实现）
# ---------------------------------------------------------------------------
class Embedder:
    """语义向量生成接口。子类实现 encode() 即可接入任意模型。"""
    def encode(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """零依赖占位实现：基于词哈希生成确定性向量。

    仅用于无模型环境下的检索演示——相同/相近文本得到相近向量。
    生产环境替换为真实句子编码器（如本地小模型 / 轻量 VLM 文本塔）即可，
    不修改 MemorySystem 任何代码。
    """
    DIM = 256

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for tok in _tokenize(text):
            h = hashlib.md5(tok.encode("utf-8")).digest()
            for i in range(self.DIM):
                vec[i] += (h[i % len(h)] - 128) / 128.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _tokenize(text: str) -> list[str]:
    """简单归一化：小写、按非字母数字切分。"""
    toks, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            toks.append("".join(cur))
            cur = []
    if cur:
        toks.append("".join(cur))
    return toks or [text.lower()]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


# ---------------------------------------------------------------------------
# 2. Token 经济联动钩子（零依赖占位，避免直接 import economy —— 保持模块解耦）
# ---------------------------------------------------------------------------
class _NullToken:
    """未注入 TokenEconomy 时用的空实现，保证 memory_system 可独立运行。"""
    def reward(self, amount: float, reason: str = "") -> None:
        pass


# ---------------------------------------------------------------------------
# 3. 经验记忆系统
# ---------------------------------------------------------------------------
class MemorySystem:
    ARCHIVE_THRESHOLD = 0.05  # score 低于此值软删入归档

    def __init__(self, db_path: str = "memory.db",
                 embedder: Optional[Embedder] = None,
                 token=None,
                 decay_rate: float = 0.1,
                 archive_threshold: float = ARCHIVE_THRESHOLD):
        self.db_path = db_path
        self.embedder = embedder or HashEmbedder()
        self.token = token or _NullToken()
        self.default_decay = decay_rate
        self.archive_threshold = archive_threshold
        self._init_db()

    # -- 建表 ----------------------------------------------------------------
    def _init_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")  # 高频读写避免锁表
        conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            memory_id      TEXT PRIMARY KEY,
            title          TEXT,
            summary        TEXT,
            file_pointer   TEXT,
            embedding      BLOB,
            frequency      INTEGER DEFAULT 0,
            last_accessed  REAL,
            decay_rate     REAL DEFAULT 0.1,
            importance     REAL DEFAULT 0.5,
            tags           TEXT,             -- JSON 数组
            source_session TEXT,
            created_at     REAL,
            archived       INTEGER DEFAULT 0,
            contested      INTEGER DEFAULT 0 -- 被 System2 纠正/质疑次数（避免经验主义）
        )
        """)
        conn.commit()
        conn.close()

    # -- 评分 ----------------------------------------------------------------
    def calculate_score(self, row: dict) -> float:
        """strength × decay × importance（类艾宾浩斯遗忘曲线）。

        row 需含：frequency, last_accessed, decay_rate, importance, contested
        contested 越多，有效重要性打折——失败经验贬值但保留记录（可复盘）。
        """
        strength = math.log(row["frequency"] + 1)
        now = time.time()
        days_ago = (now - row["last_accessed"]) / 86400.0
        decay = math.exp(-row["decay_rate"] * days_ago)
        eff_importance = row["importance"] / (1.0 + row["contested"] * 0.5)
        return strength * decay * eff_importance

    # -- 写入 ----------------------------------------------------------------
    def write_memory(self, title: str, summary: str, file_pointer: str,
                     tags: Optional[list[str]] = None,
                     importance: float = 0.5,
                     source_session: str = "",
                     permanent: bool = False) -> str:
        """写入一条记忆。标题重复视为复习：frequency+1, last_accessed 刷新。

        permanent=True 的记忆 decay_rate=0，永不衰减（如核心安全规则）。
        """
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        hit = conn.execute(
            "SELECT memory_id, frequency FROM memory WHERE title=? AND archived=0",
            (title,)).fetchone()
        emb = json.dumps(self.embedder.encode(title + " " + summary)).encode("utf-8")
        if hit:  # 复习已有记忆
            mid, freq = hit
            conn.execute(
                "UPDATE memory SET summary=?, file_pointer=?, embedding=?, "
                "frequency=?, last_accessed=?, importance=?, tags=? WHERE memory_id=?",
                (summary, file_pointer, emb, freq + 1, now, importance,
                 json.dumps(tags or []), mid))
            conn.commit(); conn.close()
            return mid
        # 新记忆
        mid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO memory (memory_id, title, summary, file_pointer, embedding, "
            "frequency, last_accessed, decay_rate, importance, tags, source_session, "
            "created_at, archived) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, title, summary, file_pointer, emb, 1, now,
             self.default_decay, importance, json.dumps(tags or []),
             source_session, now, 0))
        if permanent:
            conn.execute("UPDATE memory SET decay_rate=0 WHERE memory_id=?", (mid,))
        conn.commit(); conn.close()
        return mid

    def write_memory_with_id(self, memory_id: str, title: str, summary: str,
                             file_pointer: str, tags: Optional[list[str]] = None,
                             importance: float = 0.5, source_session: str = "",
                             permanent: bool = False) -> str:
        """写入一条记忆并固定其 memory_id（用于踩坑归档等需稳定引用的场景）。

        与 write_memory 的区别：id 由调用方指定，不自动生成 uuid；
        title 重复同样视为复习（frequency+1）。全程单连接，避免外部改 id 时的锁竞争。
        """
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        hit = conn.execute(
            "SELECT memory_id, frequency FROM memory WHERE title=? AND archived=0",
            (title,)).fetchone()
        emb = json.dumps(self.embedder.encode(title + " " + summary)).encode("utf-8")
        if hit:  # 复习已有记忆（保留调用方指定的 id）
            conn.execute(
                "UPDATE memory SET summary=?, file_pointer=?, embedding=?, "
                "frequency=?, last_accessed=?, importance=?, tags=?, memory_id=? "
                "WHERE memory_id=?",
                (summary, file_pointer, emb, hit[1] + 1, now, importance,
                 json.dumps(tags or []), memory_id, hit[0]))
            if permanent:
                conn.execute("UPDATE memory SET decay_rate=0 WHERE memory_id=?", (memory_id,))
            conn.commit(); conn.close()
            return memory_id
        conn.execute(
            "INSERT INTO memory (memory_id, title, summary, file_pointer, embedding, "
            "frequency, last_accessed, decay_rate, importance, tags, source_session, "
            "created_at, archived) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (memory_id, title, summary, file_pointer, emb, 1, now,
             self.default_decay, importance, json.dumps(tags or []),
             source_session, now, 0))
        if permanent:
            conn.execute("UPDATE memory SET decay_rate=0 WHERE memory_id=?", (memory_id,))
        conn.commit(); conn.close()
        return memory_id

    # -- 检索 ----------------------------------------------------------------
    def search_memory(self, query: str, top_k: int = 10,
                      include_archived: bool = False) -> list:
        """语义检索：余弦相似度排序，命中即复习（frequency+1, last_accessed 刷新）。"""
        q_emb = self.embedder.encode(query)
        conn = sqlite3.connect(self.db_path)
        where = "1=1" if include_archived else "archived=0"
        rows = conn.execute(
            "SELECT memory_id, title, summary, embedding, frequency, last_accessed, "
            "decay_rate, importance, tags, contested FROM memory WHERE " + where
        ).fetchall()
        scored = []
        now = time.time()
        for r in rows:
            emb = json.loads(r[3].decode("utf-8"))
            scored.append((_cosine(q_emb, emb), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, r in scored[:top_k]:
            mid = r[0]
            conn.execute(
                "UPDATE memory SET frequency=frequency+1, last_accessed=? WHERE memory_id=?",
                (now, mid))
            results.append({
                "memory_id": mid, "title": r[1], "summary": r[2],
                "similarity": round(sim, 4), "tags": json.loads(r[8]),
                "contested": r[9],
            })
        conn.commit(); conn.close()
        return results

    # -- 衰减更新 / 归档 -----------------------------------------------------
    def update_all_scores(self, archive: bool = True) -> dict:
        """定时衰减（建议由 System1 低频触发 / OS cron 每天一次）。

        score 低于阈值 → 软删（archived=1）。
        permanent（decay_rate=0）跳过，永不归档。
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT memory_id, frequency, last_accessed, decay_rate, importance, "
            "contested, archived FROM memory").fetchall()
        archived_n = 0
        for r in rows:
            mid, freq, last, dr, imp, con, arc = r
            if arc == 0 and dr > 0:
                row = {"frequency": freq, "last_accessed": last, "decay_rate": dr,
                       "importance": imp, "contested": con}
                if self.calculate_score(row) < self.archive_threshold:
                    if archive:
                        conn.execute("UPDATE memory SET archived=1 WHERE memory_id=?", (mid,))
                        archived_n += 1
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM memory WHERE archived=0").fetchone()[0]
        conn.close()
        return {"archived": archived_n, "active": total}

    # -- 回捞 ----------------------------------------------------------------
    def recall_from_archive(self, query: str, top_k: int = 5) -> list:
        """在归档存储中搜索匹配项，找到则重建索引（archived=0）重新激活。"""
        q_emb = self.embedder.encode(query)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT memory_id, title, summary, embedding FROM memory WHERE archived=1"
        ).fetchall()
        scored = []
        now = time.time()
        for r in rows:
            emb = json.loads(r[3].decode("utf-8"))
            scored.append((_cosine(q_emb, emb), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        reactivated = []
        for sim, r in scored[:top_k]:
            if sim < 0.3:  # 弱匹配不回捞，避免噪声
                continue
            mid = r[0]
            conn.execute(
                "UPDATE memory SET archived=0, last_accessed=?, frequency=frequency+1 "
                "WHERE memory_id=?", (now, mid))
            reactivated.append({"memory_id": mid, "title": r[1],
                                "similarity": round(sim, 4)})
        conn.commit(); conn.close()
        return reactivated

    # -- 经验主义纠正钩子（接 System2 关键帧纠正） --------------------------
    def challenge_memory(self, memory_id: str, success: bool) -> None:
        """System2 在关键帧纠正后回调：

        success=False → 该经验被质疑：importance 打折、contested+1
                        （避免经验主义中毒，保留记录供复盘）
        success=True  → 复用成功：importance 微涨，并反哺 token 经济
                        （记忆=省一次 System2 调用=省钱=活得更久）
        """
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT importance, contested FROM memory WHERE memory_id=?",
            (memory_id,)).fetchone()
        if not row:
            conn.close(); return
        imp, con = row
        if success:
            conn.execute("UPDATE memory SET importance=? WHERE memory_id=?",
                         (min(1.0, imp + 0.05), memory_id))
            self.token.reward(5.0, reason="memory_reuse_success")
        else:
            conn.execute(
                "UPDATE memory SET importance=?, contested=? WHERE memory_id=?",
                (imp * 0.5, con + 1, memory_id))
        conn.commit(); conn.close()

    # -- 统计 ----------------------------------------------------------------
    def stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        active = conn.execute("SELECT COUNT(*) FROM memory WHERE archived=0").fetchone()[0]
        archived = conn.execute("SELECT COUNT(*) FROM memory WHERE archived=1").fetchone()[0]
        contested = conn.execute(
            "SELECT COUNT(*) FROM memory WHERE contested>0 AND archived=0").fetchone()[0]
        conn.close()
        return {"active": active, "archived": archived, "contested": contested}


# ---------------------------------------------------------------------------
# 4. 自检（python memory_system.py 直接运行）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ms = MemorySystem("demo_memory.db", decay_rate=0.1)
    ms.write_memory("Boss战闪避策略", "黑神话二郎神第二阶段贴脸绕后",
                    "obsidian/boss.md", tags=["游戏", "黑神话"], importance=0.8)
    ms.write_memory("UE5循线校准", "速度初值偏差用0.1s双采样反推修正",
                    "obsidian/calib.md", tags=["预判帧", "ue5"], importance=0.6)

    print("检索 '怎么打Boss':", ms.search_memory("怎么打Boss战"))
    print("统计:", ms.stats())

    # 模拟一条失败经验被 System2 质疑
    mid = ms.search_memory("Boss")[0]["memory_id"]
    ms.challenge_memory(mid, success=False)
    print("质疑后统计:", ms.stats())

    # 衰减归档演示（permanent 记忆不会被归档）
    ms.write_memory("核心安全规则", "动作输出前必须过急停校验",
                    "obsidian/safety.md", importance=1.0, permanent=True)
    print("衰减更新:", ms.update_all_scores())
    print("回捞测试:", ms.recall_from_archive("Boss 二阶段"))
