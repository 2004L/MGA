"""
trajectory.py —— MGA 真实反馈数据收集（为 ③ MiniMind 自训练供料）
============================================================
闭环每跑通一次「看 → 理解 → 说 → 动 → 结果」，就把这条轨迹写进 JSONL。

字段契约见 **data/trajectory_schema.json**（v1），训练脚本只认那份定义。
核心约束（防止之后重新清洗数据）：
    1. screenshot_ref 必须是**可回查的文件路径**，截图先落盘再存路径。
       绝不存图像数组内容（v0 bug：numpy 被 str() 后单条数 MB），
       也绝不存对象内存地址（v0 bug：PIL 被 str() 只剩 <PIL...at 0x...>）。
    2. 必须存 response（大脑完整原始响应）—— 它是 SFT 的监督目标 y，
       缺了就只能做特征对齐，无法训练生成。
    3. 必须存 prompt，且与推理时 llm_bridge.build_prompt 逐字同源，
       否则训练/推理分布不一致，微调直接失效。
    4. 每条带 schema 版本 v，便于识别遗留数据与做迁移。

依赖：stdlib；numpy 与 PIL 均按需懒加载（缺失也不影响基础写盘）。
"""
from __future__ import annotations

import json
import os
import time

SCHEMA_VERSION = 1
SCHEMA_FILE = os.path.join("data", "trajectory_schema.json")

# 按 data/trajectory_schema.json -> validation.required_keys
REQUIRED_KEYS = [
    "v", "t", "ts", "screenshot_ref", "llm_feats_shape", "llm_feats",
    "scene", "goal", "prompt", "response", "action_coords", "outcome",
]


class TrajectoryCollector:
    def __init__(self, path: str = "blobs/trajectories.jsonl",
                 shot_dir: str = "blobs/shots"):
        self.path = path
        self.shot_dir = shot_dir
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        os.makedirs(shot_dir, exist_ok=True)
        self.n = 0
        self._shot_n = 0

    # -- 截图引用落盘 ------------------------------------------------------
    def _materialize(self, ref) -> str:
        """统一把截图变成**可回查的文件路径**，杜绝内容/地址混入 JSONL。"""
        if ref is None:
            return ""        # 截图引用为空时落空字符串（headless/合成截图场景）
        # 1) 已是路径字符串
        if isinstance(ref, str):
            return ref if len(ref) <= 512 else f"<toolong:{len(ref)}>"
        # 2) PIL Image / 任何带 save 的对象（pyautogui 截图返回 PIL）
        if hasattr(ref, "save"):
            try:
                self._shot_n += 1
                path = os.path.join(self.shot_dir,
                                    f"mga_{int(time.time())}_{self._shot_n}.png")
                ref.save(path)
                return path
            except Exception:
                pass  # 落盘失败继续往下找别的表示，不中断闭环
        # 3) numpy 数组（无 pyautogui 时的合成兜底截图）
        #    注意：hasattr 只做类型判断，取值必须用 np.asarray，
        #    否则拿到的是 __array__ 绑定方法而非数组本身。
        import numpy as np
        if hasattr(ref, "__array__"):
            try:
                from PIL import Image
                a = np.asarray(ref)
                self._shot_n += 1
                path = os.path.join(self.shot_dir,
                                    f"mga_{int(time.time())}_{self._shot_n}.png")
                Image.fromarray(a).save(path)
                return path
            except Exception:
                pass
            # 没有 PIL：退化为内容哈希 + 形状，至少可去重/可追溯，不塞整张图
            try:
                import hashlib
                a = np.asarray(ref)
                h = hashlib.sha1(a.tobytes()).hexdigest()[:12]
                return f"array:sha1:{h}:shape{a.shape}"
            except Exception:
                pass
        # 4) 其余（Mock 的占位 dict 等）：截断，防止体积失控
        s = str(ref)
        return s if len(s) <= 512 else s[:512] + f"...<truncated:{len(s)}>"

    # -- 写一条轨迹 --------------------------------------------------------
    def record(self, *, t, screenshot_ref, llm_feats, scene,
               response_target, action_coords, outcome,
               goal: str = "", prompt: str = "", response: str = "",
               action: str = "", backend: dict = None,
               executed: bool = False, max_elements: int = 0) -> None:
        """写一条 v1 轨迹。llm_feats 为类数组（numpy）；outcome 为 bool。

        outcome 语义（与 executed 配套，见 data/trajectory_schema.json）：
        - executed=True  → outcome 表示动作真实执行是否成功
        - executed=False → 仅规划未真点，outcome 表示规划是否命中真实目标
        旧版规划一律记 False，会让 schema 的 sample_weight_or_filter=outcome
        过滤掉全部采集样本，故必须靠 executed 区分。
        executed 为可选字段（默认 False），不放进 REQUIRED_KEYS，保证老 v1 记录仍可校验。
        """
        import numpy as np
        feats = (np.asarray(llm_feats, dtype=float)
                 if llm_feats is not None else np.zeros((0,), dtype=float))
        row = {
            "v": SCHEMA_VERSION,
            "t": float(t),
            "ts": time.time(),
            "screenshot_ref": self._materialize(screenshot_ref),
            "llm_feats_shape": list(feats.shape),
            "llm_feats": feats.tolist(),   # 完整特征（n_patches*out_dim 不大）
            "scene": [str(s) for s in scene],
            "goal": str(goal),
            "prompt": str(prompt),
            "response": str(response),          # SFT 监督目标 y
            "response_target": str(response_target),
            "action": str(action),
            "action_coords": [int(c) for c in action_coords],
            "outcome": bool(outcome),           # 执行成功 / 规划有效（看 executed）
            "executed": bool(executed),         # True=真点过；False=仅规划
            # 进 prompt 的元素数上限。必须落盘：训练侧若与推理侧上限不同，
            # prompt 分布会静默错配（这类 bug 训完才发现，等于白洗一遍数据），
            # 训练脚本据此校验一致性。
            "max_elements": int(max_elements),
            "backend": dict(backend or {}),     # 数据来源溯源
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.n += 1

    def stats(self) -> dict:
        return {"records": self.n, "path": self.path}


def validate(path: str = "blobs/trajectories.jsonl", verbose: bool = True) -> dict:
    """按 data/trajectory_schema.json 的 validation 规则逐行校验。
    返回 {total, v0, v1, problems:[(行号, 原因)]}。"""
    stats = {"total": 0, "v0": 0, "v1": 0, "problems": []}
    if not os.path.exists(path):
        if verbose:
            print(f"[validate] 文件不存在：{path}")
        return stats
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                stats["problems"].append((i, f"JSON 解析失败: {e}"))
                continue
            stats["total"] += 1
            if "v" in row:
                stats["v1"] += 1
            else:
                stats["v0"] += 1
                stats["problems"].append((i, "v0 遗留记录（缺 response/prompt/goal，不可用于生成训练）"))
            miss = [k for k in REQUIRED_KEYS if k not in row]
            if miss:
                stats["problems"].append((i, f"缺字段 {miss}"))
            shp, feats = row.get("llm_feats_shape"), row.get("llm_feats")
            if shp and feats and (len(feats) != shp[0] or len(feats[0]) != shp[1]):
                stats["problems"].append((i, f"llm_feats 形状与 llm_feats_shape 不符 {shp}"))
            ref = row.get("screenshot_ref", "")
            # 可回查引用判定：要么是真实存在的文件，要么是 numpy 内容哈希引用；
            # Mock 占位 dict 的 str()（如 "{t: 1.5, pos: ...}"）既非文件也非哈希，必须标脏。
            good_ref = (isinstance(ref, str) and len(ref) <= 512 and not ref.startswith("<")
                        and (ref.startswith("array:sha1:") or os.path.isfile(ref)))
            if not good_ref:
                stats["problems"].append((i, f"screenshot_ref 不可回查（非文件/非哈希/超长）: {str(ref)[:60]}"))
            if not isinstance(row.get("outcome"), bool):
                stats["problems"].append((i, "outcome 非严格布尔"))
    if verbose:
        print(f"[validate] {path}: 共 {stats['total']} 条 "
              f"(v1={stats['v1']}, v0={stats['v0']})，问题 {len(stats['problems'])} 处")
        for i, why in stats["problems"][:10]:
            print(f"    行 {i}: {why}")
    return stats


if __name__ == "__main__":
    # 自检：numpy 数组截图 / PIL 类对象截图 都必须落成**真实存在的文件路径**，
    # 且写出的记录能通过 schema 校验。
    import tempfile
    import numpy as np

    tmp = tempfile.mkdtemp()
    tc = TrajectoryCollector(path=os.path.join(tmp, "t.jsonl"),
                             shot_dir=os.path.join(tmp, "shots"))

    class _FakePIL:                       # 模拟 pyautogui 返回的 PIL Image
        def __init__(self, arr): self.arr = arr
        def save(self, path):
            from PIL import Image
            Image.fromarray(self.arr).save(path)

    img = (np.arange(64 * 64 * 3) % 255).reshape(64, 64, 3).astype("uint8")
    feats = np.zeros((16, 64))

    tc.record(t=1.0, screenshot_ref=img, llm_feats=feats, scene=["1. Button [确定]"],
              response_target="确定", action_coords=(10, 20), outcome=True,
              goal="点击确定按钮", prompt="目标：点击确定按钮\n\n1. Button [确定]",
              response='{"action":"click","target":"确定","coordinates":[10,20]}',
              action="click", backend={"visual": "screenparser", "llm": "mock"})
    tc.record(t=2.0, screenshot_ref=_FakePIL(img), llm_feats=feats, scene=["1. Button [确定]"],
              response_target="确定", action_coords=(11, 21), outcome=False,
              goal="点击确定按钮", prompt="目标：点击确定按钮\n\n1. Button [确定]",
              response='{"action":"click","target":"确定","coordinates":[11,21]}',
              action="click", backend={"visual": "yolo", "llm": "mock"})

    st = validate(tc.path, verbose=False)
    assert st["total"] == 2 and not st["problems"], st["problems"]

    try:
        from PIL import Image  # noqa: F401
        has_pil = True
    except ImportError:
        has_pil = False

    with open(tc.path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        ref = r["screenshot_ref"]
        assert len(ref) < 512 and not ref.startswith("<"), f"引用可疑: {ref[:60]}"
        if has_pil:
            # 有 PIL：截图必须真正落盘成可回查文件
            assert os.path.isfile(ref), f"截图未落盘: {ref}"
        else:
            # 无 PIL：数组截图退化为内容哈希引用（可去重可追溯），路径类引用仍须存在
            assert ref.startswith("array:sha1:") or os.path.isfile(ref), ref
        assert r["v"] == SCHEMA_VERSION and r["response"] and r["prompt"] and r["goal"]
    tag = "截图落盘为可回查路径" if has_pil else "截图退化为哈希引用（未装 PIL，装后即落盘）"
    print(f"✅ trajectory 自检通过：{tag} / v1 全字段齐备 / 通过 schema 校验")
