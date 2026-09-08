"""
src/knowledge/graphrag.py —— MGA ④ 知识层：外部知识 GraphRAG（numpy+networkx 自研）

与 memory_system.py（M5 经验记忆）互补：
    M5 管「我做过什么」（episodic 经验），本层管「世界是什么」（semantic/facts）。
    两者都喂 LLM 上下文，但检索机制不同：M5 按频率衰减+语义相似，
    本层按「实体多跳 + 语义」——这正是普通 RAG 多跳推理失效的解法（报告 1.1）。

设计原则（与系统一致）：
    - 零重依赖：numpy + networkx + stdlib 即可跑（自检用 MockExtractor/MockEmbedder）。
    - 不集成 tiny-graphrag/nano-graphrag：只借其「双索引（图+向量）」与 NetworkXStorage 思路，自研。
    - 触发式：Local Search 只在关键帧（System2 唤醒）时调用，平日零开销。

参考的 GraphRAG 7 步索引（本版精简）：
    源文档 → 分块 → 实体/关系抽取 → 摘要 → 社区检测(Louvain) → 嵌入 → 索引
    保留灵魂：先按实体从图捞一跳/两跳关系，再喂 LLM（不是纯向量召回）；
    社区检测支撑 Global Search（社区主题摘要 → 全局问答）。

核心接口（从 nano-graphrag 抽形状，内部全自研）：
    insert_document(text)   分块→抽实体/关系→加节点边→嵌入→存向量
    upsert_elements(els)    M3 检测元素→实体节点（elements_to_entities）
    local_search(q)         embed(q)→top-k实体→扩hops邻居→组上下文
    save/load               图持久化（pickle），重启不丢
"""

from __future__ import annotations

import re
import uuid
import json
import pickle
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities


# ---------------------------------------------------------------------------
# 1. 可插拔原语（零依赖自检用 Mock，真部署注入真 LLM/嵌入）
# ---------------------------------------------------------------------------
class Embedder:
    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError


class MockEmbedder(Embedder):
    """零依赖嵌入：字符+词哈希成 dim 维向量，L2 归一化。
    共享词/字 → 余弦相似度高，足够演示 Local Search 召回。真部署换真编码器。"""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        for ch in text:
            vec[hash(ch) % self.dim] += 1.0
        for tok in re.findall(r"\w+", text):
            vec[hash(tok) % self.dim] += 2.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class Extractor:
    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        """返回 (实体A, 关系, 实体B) 三元组。"""
        raise NotImplementedError


class MockExtractor(Extractor):
    """零依赖抽取：按标点拆子句为实体，相邻用 related_to 连成链。
    演示图结构（多跳）价值；真部署换 LLM 抽取（接口不变）。"""

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        parts = [p.strip() for p in re.split(r"[。；;]", text) if p.strip()]
        nodes = [p[:24] for p in parts]
        triples = []
        for i in range(len(nodes) - 1):
            triples.append((nodes[i], "related_to", nodes[i + 1]))
        if not triples and nodes:
            triples.append((nodes[0], "is_a", "concept"))
        return triples


# ---------------------------------------------------------------------------
# 真 LLM 抽取器（接口与 MockExtractor 完全一致，仅实现换成 LLM）
# ---------------------------------------------------------------------------
def _coerce_json(text: str):
    """把 LLM 回复抠成 list 或 dict，兼容「裸数组」与「含数组的对象」。"""
    if not text:
        return []
    m = re.search(r"\[.*\]", text, re.DOTALL)     # 优先裸数组
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return v
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)     # 退化为对象
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return []


class LLMExtractor(Extractor):
    """真 LLM 实体/关系抽取：文本 → 让 LLM 返回 JSON 三元组 → 解析为 (A,关系,B)。

    接口与 MockExtractor 完全一致（extract(text)->triples），KnowledgeGraph 无需改动。
    llm_client 需提供 respond(prompt: str) -> str（如 llm_bridge.UltralyticsLLM / APILLM）。
    依赖懒加载：只在调用时连 LLM，缺失即报错（上层可降级回 MockExtractor）。
    """

    PROMPT = (
        "从下面文本中抽取实体及其关系。仅输出一个 JSON 数组，每项为"
        '{"subject":"实体A","relation":"关系","object":"实体B"}。'
        "若文本只是单一概念无明确关系，也至少返回一个自描述三元组。"
        "不要解释、不要代码块。\n\n文本：{text}"
    )

    def __init__(self, llm_client):
        self.llm = llm_client       # duck-typed: 需有 respond(prompt)->str
        self._warned = False        # 失败只在首次提示，后续静默——关键帧每帧调一次，刷屏会淹没采集日志

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        prompt = self.PROMPT.replace("{text}", text)   # 避免 .format 误解析 JSON 花括号
        try:
            reply = self.llm.respond(prompt)
        except Exception as e:
            if not self._warned:
                print(f"[LLMExtractor] LLM 调用失败，后续静默回退空三元组: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                self._warned = True
            return []
        data = _coerce_json(reply)
        items = data if isinstance(data, list) else data.get("triples", data.get("relations", []))
        triples: List[Tuple[str, str, str]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            s = it.get("subject") or it.get("entity") or it.get("head")
            r = it.get("relation") or it.get("predicate") or "related_to"
            o = it.get("object") or it.get("target") or it.get("tail")
            if s and o:
                triples.append((str(s), str(r), str(o)))
        return triples


# ---------------------------------------------------------------------------
# 2. 核心：KnowledgeGraph（双索引 = networkx 图 + numpy 向量）
# ---------------------------------------------------------------------------
class KnowledgeGraph:
    def __init__(self, embedder: Optional[Embedder] = None,
                 extractor: Optional[Extractor] = None, dim: int = 64):
        self.G = nx.DiGraph()          # 节点=实体, 边=关系(带 relations 列表)
        self.vecs = {}                 # node_id -> np.ndarray (D维)
        self.embedder = embedder or MockEmbedder(dim)
        self.extractor = extractor or MockExtractor()
        self.dim = dim
        self.communities = []                 # louvain 社区列表
        self.community_summaries = {}         # 社区 id -> LLM 主题摘要

    # ---- 底层节点/边（增量 Upsert：同名实体合并，图不膨胀）----
    def _merge_node(self, name: str, descriptions: Optional[list] = None,
                    source_id=None, etype: str = "entity"):
        """参考 nano-graphrag _merge_nodes_then_upsert：
        同名节点合并 description 列表累加、source_id 去重合并，绝不新建重复节点。"""
        descriptions = descriptions or []
        if self.G.has_node(name):
            n = self.G.nodes[name]
            n["descriptions"] = list(set(n.get("descriptions", []) + descriptions))
            if source_id:
                n["source_ids"] = list(set(n.get("source_ids", []) + [source_id]))
        else:
            self.G.add_node(name, descriptions=list(set(descriptions)),
                            source_ids=[source_id] if source_id else [], etype=etype)
        # 合并后重算向量（累积知识要反映进检索）
        self.vecs[name] = self.embedder.encode(
            " ".join(self.G.nodes[name]["descriptions"]) or name)

    def _add_node(self, name: str, summary: str = "", source_id=None, etype: str = "entity"):
        """薄包装：单条摘要也走合并逻辑，接口兼容旧调用。"""
        self._merge_node(name, descriptions=[summary] if summary else [],
                         source_id=source_id, etype=etype)

    def _add_edge(self, a: str, b: str, rel: str):
        if self.G.has_edge(a, b):
            e = self.G[a][b]
            e["relations"] = list(set(e.get("relations", []) + [rel]))
        else:
            self.G.add_edge(a, b, relation=rel, relations=[rel])

    # ---- Step2 接口：插入文档（7 步精简版 + 增量 Upsert）----
    def insert_document(self, text: str, doc_id: Optional[str] = None) -> str:
        triples = self.extractor.extract(text)
        # 增量 Upsert：先按实体名聚合同名出现，再合并节点（description 累加、source_id 合并）
        agg = defaultdict(list)
        for a, rel, b in triples:
            agg[a].append((rel, b, "out"))
            agg[b].append((rel, a, "in"))
        for name, occ in agg.items():
            descs = []
            for rel, other, direction in occ:
                if direction == "out":
                    descs.append(f"{name} {rel} {other}")
                else:
                    descs.append(f"{other} {rel} {name}")
            self._merge_node(name, descriptions=descs, source_id=doc_id, etype="entity")
        for a, rel, b in triples:
            self._add_edge(a, b, rel)
        # doc 节点挂到它提及的实体上（mentions 边），避免 doc 成为游离单点社区
        did = doc_id or f"doc_{uuid.uuid4().hex[:6]}"
        for name in agg.keys():
            self._add_edge(did, name, "mentions")
        self._add_node(did, text[:200], source_id=doc_id, etype="document")  # 原文也可检索
        return did

    # ---- 对接 M3：视觉元素 → 实体节点（同样走合并，每帧不膨胀）----
    def upsert_elements(self, elements) -> str:
        """elements: List[Element]（detector.py）。每个元素→实体节点（按名合并），
        全部挂到「当前场景」节点下。这就是 3DGraphLLM + Chat-Scene 的代码落地，
        且因合并，每帧重复出现的「确定」不会新建重复节点。"""
        scene = "当前场景"
        self._merge_node(scene, descriptions=["实时视觉检测到的 UI 元素集合"], etype="scene")
        for e in elements:
            name = (getattr(e, "text", None) or e.label or "element")
            self._merge_node(name, descriptions=[f"{e.label} 位于 {e.center()}"], etype="ui_element")
            self._add_edge(name, scene, "appears_in")
            self._add_edge(scene, name, "contains")
        return scene

    # ---- Step3 接口：Local Search（关键帧时调用）----
    def _retrieve_seeds(self, q: str, top_k: int = 3) -> List[str]:
        """语义召回 top-k 种子实体（供 Local / Global Search 共用）。"""
        if not self.G.nodes:
            return []
        qv = self.embedder.encode(q)
        scored = []
        for nid, vec in self.vecs.items():
            sim = float(np.dot(qv, vec) / (np.linalg.norm(qv) * np.linalg.norm(vec) + 1e-9))
            scored.append((sim, nid))
        scored.sort(reverse=True)
        return [nid for _, nid in scored[:top_k]]

    def local_search(self, q: str, top_k: int = 3, hops: int = 1) -> str:
        seeds = self._retrieve_seeds(q, top_k)
        # 图多跳扩邻居（双向），拿到关系上下文
        ctx_nodes = set(seeds)
        for s in seeds:
            for nb in nx.ego_graph(self.G, s, radius=hops, undirected=True):
                ctx_nodes.add(nb)
        lines = []
        for nid in ctx_nodes:
            descs = self.G.nodes[nid].get("descriptions", [])
            summ = " | ".join(descs) if descs else ""
            lines.append(f"- {nid}: {summ}")
        return "\n".join(lines)

    # ---- 社区检测 + Global Search（networkx Louvain，零额外依赖）----
    def detect_communities(self, seed: int = 42) -> list:
        """对每个连通分量分别做 Louvain 社区划分（含孤立单点也自成一社区），
        社区标签写回节点。单层即可（Leiden 更准但需 igraph，自研阶段不需要）。"""
        und = self.G.to_undirected()
        if und.number_of_nodes() == 0:
            return []
        comms = []
        cid = 0
        for comp in nx.connected_components(und):
            sub = und.subgraph(comp).copy()
            if sub.number_of_nodes() == 1:
                sub_comms = [frozenset(sub.nodes())]     # 单点组件自成一类
            else:
                sub_comms = louvain_communities(sub, seed=seed)
            for c in sub_comms:
                for n in c:
                    if self.G.has_node(n):
                        self.G.nodes[n]["community"] = cid
                comms.append(c)
                cid += 1
        self.communities = comms
        return comms

    def summarize_communities(self, llm_func, seed: int = 42) -> dict:
        """每个社区用 LLM 生成多跳主题摘要（Global Search 的底层支撑）。
        llm_func(prompt: str) -> str（如 UltralyticsLLM.respond / APILLM.respond）。"""
        comms = self.communities or self.detect_communities(seed)
        summaries = {}
        for i, c in enumerate(comms):
            ctx = []
            for node in c:
                for nbr in self.G.successors(node):
                    if nbr in c:
                        rels = self.G[node][nbr].get("relations",
                                                    [self.G[node][nbr].get("relation", "关联")])
                        for r in rels:
                            ctx.append(f"{node} --{r}--> {nbr}")
            prompt = ("以下是一组紧密关联的实体和关系，请提炼主题摘要：\n"
                      + "\n".join(ctx))
            try:
                summaries[i] = llm_func(prompt)
            except Exception as e:
                summaries[i] = f"[摘要失败:{e}] " + "; ".join(ctx[:3])
        self.community_summaries = summaries
        return summaries

    def global_search(self, q: str, top_k: int = 2) -> str:
        """Global Search：Local Search 找种子实体 → 取其所属社区摘要作上下文。"""
        if not self.communities:
            self.detect_communities()
        seeds = self._retrieve_seeds(q, top_k)
        seen, out = set(), []
        for sn in seeds:
            ci = self.G.nodes.get(sn, {}).get("community")
            if ci is not None and ci not in seen:
                seen.add(ci)
                out.append(f"[社区{ci}] {self.community_summaries.get(ci, '')}")
        return "\n".join(out) if out else self.local_search(q, top_k, hops=0)

    # ---- 持久化（任意属性可序列化，用 pickle 最稳）----
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.G, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.G = pickle.load(f)
        for nid in self.G.nodes:               # 向量不在图里，按 description 重算
            descs = self.G.nodes[nid].get("descriptions", [])
            self.vecs[nid] = self.embedder.encode(" ".join(descs) if descs else nid)


# ---------------------------------------------------------------------------
# 3. 自检（python knowledge/graphrag.py）：用 M5 经验数据 + M3 元素演示闭环
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs("blobs", exist_ok=True)
    from memory_system import MemorySystem
    from perception.detector import Element

    print("=== ④ 知识层 GraphRAG 自检（numpy+networkx 自研，零重依赖）===")
    kg = KnowledgeGraph()

    # --- 用 M5 经验记忆数据灌入外部知识层 ---
    mem = MemorySystem("blobs/m5_demo.db")     # 真文件库（:memory: 在 MemorySystem 下不跨连接共享）
    mem.write_memory(title="关键帧纠正", summary="物体偏移时锁定确认按钮并点击成功，残差回流训练System1",
                     file_pointer="blobs/m_correction.md", tags=["correction"])
    mem.write_memory(title="ScreenParser定位", summary="YOLO检测UI元素输出bbox，LLM按语义名点击重试按钮",
                     file_pointer="blobs/m_locator.md", tags=["locator"])
    mem.write_memory(title="token经济", summary="余额低时优先走System1省算力，余额高才唤醒System2",
                     file_pointer="blobs/m_economy.md", tags=["economy"])
    for m in mem.search_memory("关键帧", top_k=100):
        kg.insert_document(m["summary"], doc_id=m["memory_id"])
    print(f"M5 经验已转为知识图：节点数={kg.G.number_of_nodes()} 边数={kg.G.number_of_edges()}")

    # --- 对接 M3：视觉检测元素 → 实体节点 ---
    els = [Element(id="e1", label="Button", bbox=(500, 300, 560, 340), conf=0.95, text="确定"),
           Element(id="e2", label="Button", bbox=(620, 360, 700, 400), conf=0.90, text="重试")]
    kg.upsert_elements(els)
    print(f"M3 元素已 upsert：节点数={kg.G.number_of_nodes()}")

    # --- Step3：关键帧时 Local Search，拿背景知识喂 LLM ---
    print("\n=== Local Search（关键帧触发时调用，结果注入 LLM 上下文）===")
    ctx = kg.local_search("点击确定按钮时如何纠正偏移", top_k=2, hops=1)
    print(ctx)

    # --- 持久化验证 ---
    kg.save("blobs/knowledge.pkl")
    kg2 = KnowledgeGraph()
    kg2.load("blobs/knowledge.pkl")
    print(f"\n持久化往返 OK：重载节点数={kg2.G.number_of_nodes()}")

    # 统一的脚本化 LLM client（按文本返回不同三元组 / 社区模板摘要，免联网免密钥）
    class FakeLLMClient:
        def respond(self, prompt: str) -> str:
            if "抽取实体" in prompt:                      # LLMExtractor 抽取请求
                if "YOLO检测" in prompt:
                    return ('[{"subject":"YOLO","relation":"detects","object":"UI元素"},'
                            '{"subject":"LLM","relation":"clicks","object":"重试按钮"}]')
                if "ScreenParser" in prompt:
                    return ('[{"subject":"ScreenParser","relation":"locates","object":"元素坐标"},'
                            '{"subject":"OCR","relation":"adds","object":"文本标签"}]')
                if "token" in prompt or "余额" in prompt:
                    return ('[{"subject":"token经济","relation":"constrains","object":"System1"},'
                            '{"subject":"余额高","relation":"wakes","object":"System2"}]')
                if "System1残差" in prompt:
                    return ('[{"subject":"System1残差","relation":"trained_by","object":"RLS"},'
                            '{"subject":"预测帧","relation":"predicts","object":"物理"}]')
                if "确认按钮触发" in prompt:
                    return '[{"subject":"确认按钮","relation":"triggers","object":"关键帧纠正"}]'
                if "确认按钮点击" in prompt:
                    return '[{"subject":"确认按钮","relation":"opens","object":"确认弹窗"}]'
                return '[{"subject":"概念","relation":"is_a","object":"concept"}]'
            return "主题：视觉感知与执行链路。"            # 社区摘要请求

    # --- 增量 Upsert 验证（核心 feature 1）：同名实体合并，图不膨胀 ---
    print("\n=== 增量 Upsert（同名实体合并，禁止图膨胀）===")
    kg_u = KnowledgeGraph()
    core = "确认按钮触发关键帧纠正"
    kg_u.insert_document(core, doc_id="d1")
    ent1 = sum(1 for n in kg_u.G.nodes if kg_u.G.nodes[n].get("etype") == "entity")
    tot1 = kg_u.G.number_of_nodes()
    # 同核心句再 upsert（模拟重复帧/重复来源）→ 实体应合并，仅新增 1 个 doc 节点
    kg_u.insert_document(core, doc_id="d2")
    ent2 = sum(1 for n in kg_u.G.nodes if kg_u.G.nodes[n].get("etype") == "entity")
    tot2 = kg_u.G.number_of_nodes()
    # 异质文档才真正扩实体节点
    kg_u.insert_document("token经济约束System2唤醒频次", doc_id="d3")
    ent3 = sum(1 for n in kg_u.G.nodes if kg_u.G.nodes[n].get("etype") == "entity")
    print(f"entity节点: d1={ent1} 同核d2={ent2} 异质d3={ent3}")
    print(f"total节点 : d1={tot1} d2={tot2} d3={kg_u.G.number_of_nodes()}")
    assert ent2 == ent1, "❌ 同名实体重复，图膨胀！"
    assert ent3 > ent2, "❌ 异名实体未扩容"
    assert tot2 == tot1 + 1, "❌ 重复 upsert 应仅新增 1 个 doc 节点（实体零新增）"
    print("✅ 增量 Upsert 通过：entity 零重复、仅按文档新增 doc 节点")

    # description 跨文档累加（用真 LLM 抽取路径，同名实体跨篇聚合不同描述）
    kg_acc = KnowledgeGraph(extractor=LLMExtractor(FakeLLMClient()))
    kg_acc.insert_document("确认按钮触发关键帧纠正", doc_id="a1")
    kg_acc.insert_document("确认按钮点击后弹出确认弹窗", doc_id="a2")
    acc_desc = kg_acc.G.nodes["确认按钮"]["descriptions"]
    print(f"'确认按钮' 跨文档描述条数={len(acc_desc)}: {acc_desc}")
    assert len(acc_desc) >= 2, "❌ 同名实体 description 未跨文档累加"
    print("✅ 跨文档同名实体 description 累加通过")

    # --- 社区检测 + Global Search 验证（核心 feature 2）---
    print("\n=== 社区检测（networkx Louvain）+ Global Search ===")
    kg_c = KnowledgeGraph(extractor=LLMExtractor(FakeLLMClient()))
    # 造多个天然隔离的主题簇，并用 MGA系统 桥接形成主成分 → 验证能切分多社区
    kg_c.insert_document("YOLO检测UI元素。LLM按语义名点击重试按钮", doc_id="c1")
    kg_c.insert_document("ScreenParser定位元素坐标。OCR补充文本标签", doc_id="c2")
    kg_c.insert_document("token经济余额低走System1。余额高唤醒System2", doc_id="c3")
    kg_c.insert_document("System1残差用RLS回流。预测帧做物理预测", doc_id="c4")
    kg_c.insert_document("MGA系统集成YOLO与token经济", doc_id="c5")
    comms = kg_c.detect_communities(seed=42)
    print(f"检测到社区数={len(comms)}")
    for i, c in enumerate(comms):
        print(f"  社区{i}（{len(c)}节点）: {sorted(c)[:6]}{'...' if len(c) > 6 else ''}")
    summaries = kg_c.summarize_communities(FakeLLMClient().respond, seed=42)
    print(f"社区摘要生成数={len(summaries)}")
    gctx = kg_c.global_search("YOLO如何检测元素", top_k=2)
    print("Global Search 结果（社区摘要作上下文）：\n", gctx)
    assert len(comms) >= 1, "❌ 社区检测未产出"
    assert len(summaries) == len(comms), "❌ 社区摘要数与社区数不一致"
    print("✅ 社区检测 + Global Search 通过")

    # --- 真 LLM 抽取路径演示（接口与 MockExtractor 一致）---
    print("\n=== LLMExtractor 真抽取路径（接口与 MockExtractor 一致）===")
    kg_llm = KnowledgeGraph(extractor=LLMExtractor(FakeLLMClient()))
    kg_llm.insert_document("确认按钮触发关键帧纠正，纠正样本训练System1残差模型")
    print("抽取得到的实体节点：", list(kg_llm.G.nodes))
    ctx = kg_llm.local_search("确认按钮如何训练System1", top_k=2, hops=1)
    print("LLM 抽取路径 Local Search：\n", ctx)

    print("\n闭环验证：视觉检测(M3) → 实体节点(增量Upsert) → Local Search/Global Search "
          "→ 上下文可注入 LLM 决策。")
