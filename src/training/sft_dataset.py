"""轨迹 → SFT 监督对。

字段契约唯一来源：data/trajectory_schema.json（minimind_training_mapping）
  X_text = prompt（与推理时 build_prompt 输出逐字一致，禁止在训练侧重新拼装）
  y      = response（完整原始响应，逐 token 监督）
  过滤   = outcome + executed（见 _usable）
"""
import json
import os
from typing import Dict, List, Optional

DEFAULT_PATH = os.path.join("blobs", "trajectories.jsonl")


def _plan_valid(row: Dict) -> bool:
    """仅规划样本（executed=false）的「规划有效性」：目标非空 + 坐标不是 (0,0) 占位。"""
    target = str(row.get("response_target", "")).strip()
    coords = row.get("action_coords") or [0, 0]
    return bool(target) and list(coords) != [0, 0]


def _usable(row: Dict) -> bool:
    """按 schema 的 sample_weight_or_filter 规则判断是否可用作正样本。

    - executed=true  : outcome 表示执行是否成功，只有 true 可用
    - executed=false : outcome 表示规划是否有效（新语义）；
                       老 v1 记录无 executed 字段且 outcome 恒 false，
                       此时按 _plan_valid 从已存字段判定，避免误杀全部采集样本。
    """
    outcome = bool(row.get("outcome", False))
    executed = bool(row.get("executed", False))   # 老记录缺失 → False
    if executed:
        return outcome
    return outcome or _plan_valid(row)


def load_sft_samples(path: str = DEFAULT_PATH,
                     dedup: bool = True,
                     keep_visual: bool = False,
                     verbose: bool = True) -> List[Dict]:
    """读取轨迹文件，返回可直接训练的文本监督对。

    返回元素：{prompt, response, goal, backend, action_coords,
              (可选) llm_feats, llm_feats_shape, screenshot_ref}
    """
    if not os.path.isfile(path):
        if verbose:
            print(f"[sft] 轨迹文件不存在：{path}")
        return []

    rows, dropped_bad, dropped_unusable = [], 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                dropped_bad += 1
                continue
            if int(row.get("v", 0)) != 1:
                dropped_bad += 1
                continue
            if not _usable(row):
                dropped_unusable += 1
                continue
            rows.append(row)

    if dedup:
        seen, uniq = set(), []
        for r in rows:
            key = (r.get("prompt", ""), r.get("response", ""))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        rows = uniq

    samples = []
    for r in rows:
        s = {
            "prompt": r.get("prompt", ""),
            "response": r.get("response", ""),
            "goal": r.get("goal", ""),
            "backend": r.get("backend", {}),
            "action_coords": r.get("action_coords", [0, 0]),
            "executed": bool(r.get("executed", False)),
            "max_elements": int(r.get("max_elements", 0)),
        }
        if keep_visual:
            s["llm_feats"] = r.get("llm_feats")
            s["llm_feats_shape"] = r.get("llm_feats_shape")
            s["screenshot_ref"] = r.get("screenshot_ref")
        samples.append(s)

    if verbose:
        n_raw = sum(1 for _ in open(path, encoding="utf-8") if _.strip())
        print(f"[sft] {path}: 原始 {n_raw} 条 → 可用 {len(samples)} 条"
              f"（丢弃：不可用 {dropped_unusable}，坏行/非 v1 {dropped_bad}"
              f"{'，去重后' if dedup else ''}）")
    return samples


def check_prompt_consistency(samples: List[Dict],
                             expect_max_elements: int,
                             verbose: bool = True) -> bool:
    """训练/推理 prompt 同源校验。

    prompt 里塞多少 UI 元素（format_scene 的 max_elements）直接决定 prompt 长度与
    信息量。训练数据用 A、推理用 B，模型等于在考没复习过的题型，且损失曲线看起来
    还挺正常——训完才发现就晚了。这里在开训前就拦下。
    """
    vals = sorted({s.get("max_elements", 0) for s in samples})
    ok = True
    if vals == [0]:
        # 该字段引入前的老数据：无法校验，只提醒，不拦（拦了就没法用已有数据起步）
        if verbose:
            print(f"[consistency] 数据中无 max_elements（老数据，采于该字段引入前），"
                  f"无法校验同源。当前推理配置 {expect_max_elements}，"
                  f"重采后即可自动校验。")
        return True
    if len(vals) > 1:
        ok = False
        if verbose:
            print(f"!! 数据集里混了多种 max_elements={vals}，"
                  f"prompt 分布不一致，需按值分组后分别训练。")
    elif vals and vals[0] != expect_max_elements:
        ok = False
        if verbose:
            print(f"!! 数据 max_elements={vals[0]}，但当前推理配置为 "
                  f"{expect_max_elements}：训练与推理 prompt 不同源。"
                  f"要么用 --max-elements {vals[0]} 重采，要么把推理侧改成 {vals[0]}。")
    if ok and verbose:
        print(f"[consistency] prompt 同源 OK：max_elements={vals or ['(缺失)']} "
              f"== 推理配置 {expect_max_elements}")
    return ok


def stats(samples: List[Dict]) -> Dict:
    """看数据多样性，别拿一堆近重复样本去训练还以为在学东西。"""
    goals = {}
    for s in samples:
        goals.setdefault(s.get("goal", ""), 0)
        goals[s.get("goal", "")] += 1
    return {
        "n": len(samples),
        "unique_prompts": len({s["prompt"] for s in samples}),
        "unique_responses": len({s["response"] for s in samples}),
        "unique_goals": len(goals),
        "goal_dist": goals,
        "backends": list({(s.get("backend") or {}).get("visual") for s in samples}),
    }


if __name__ == "__main__":
    ss = load_sft_samples()
    st = stats(ss)
    print(json.dumps(st, ensure_ascii=False, indent=2))
    if ss:
        print("\n--- 样本 0 prompt（前 200 字）---")
        print(ss[0]["prompt"][:200])
        print("--- 样本 0 response ---")
        print(ss[0]["response"])
        warn = []
        if st["n"] < 100:
            warn.append(f"仅 {st['n']} 条，不足以微调出可用策略。")
        # 真正致命的是「响应多样性」：静止屏会让 prompt 因检测抖动而各不相同，
        # 但答案永远只有几个坐标 → 模型只会背答案，不会泛化。
        if st["n"] and st["unique_responses"] / st["n"] < 0.05:
            warn.append(
                f"{st['n']} 条样本只对应 {st['unique_responses']} 种 response"
                f"（多样性 {st['unique_responses']/st['n']:.1%}）：屏幕是静止的，"
                f"prompt 的微小差异只来自检测抖动，模型会背住这几个坐标而非学会定位。"
                f" remedy：① 换更多 goal ② 让屏幕真正变化（真实操作时采）③ 换不同 App/页面采。")
        if st["unique_goals"] < 5:
            warn.append(f"仅 {st['unique_goals']} 个 goal，泛化面很窄。")
        for w in warn:
            print("!! " + w)
        if not warn:
            print("OK 数据多样性可初步支撑 SFT。")
