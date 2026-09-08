"""③ 大脑自训练：MiniMind + LoRA SFT，消费 blobs/trajectories.jsonl（v1 契约）。

数据契约（唯一来源 data/trajectory_schema.json -> minimind_training_mapping）：
  X_text = trajectory.prompt   必须与推理期 build_prompt(scene, goal) 输出逐字一致
  y      = trajectory.response 完整原始响应，逐 token 监督
  过滤   = outcome + executed（见 sft_dataset._usable）

关键：completion-only loss。prompt（整屏 UI 元素列表）不参与损失，
      否则模型会去背「屏幕描述文本」而不是学「给定目标该点哪」。

用法：
  python -m training.train_minimind --report-lengths        # 只看 token 长度分布，不训练
  python -m training.train_minimind --epochs 3 --limit 64   # 小样本冒烟
  python -m training.train_minimind --epochs 3              # 全量
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from training.sft_dataset import (check_prompt_consistency, load_sft_samples,
                                  stats)
from perception.llm_bridge import DEFAULT_MAX_ELEMENTS

DEFAULT_MODEL = os.path.join("blobs", "weights", "minimind2-small")
DEFAULT_OUT = os.path.join("blobs", "weights", "mga-lora")
# MiniMind2-Small 自定义架构，trust_remote_code 加载
FALLBACK_CODE_REPO = "jingyaogong/MiniMind2-V"   # 该仓库带 model_minimind.py


def load_model_and_tokenizer(model_dir: str = DEFAULT_MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _try(d):
        tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            d, trust_remote_code=True, torch_dtype=torch.float32)
        return tok, model

    try:
        return _try(model_dir)
    except Exception as e:
        # 常见坑：权重仓库只放 config/safetensors，没放自定义模型代码
        if "model_minimind" not in str(e) and "trust_remote_code" not in str(e):
            raise
        from huggingface_hub import hf_hub_download
        print(f"[model] 权重目录缺自定义代码，从 {FALLBACK_CODE_REPO} 补 model_minimind.py")
        src = hf_hub_download(FALLBACK_CODE_REPO, "model_minimind.py",
                              local_dir=model_dir)
        print(f"[model] 已补：{src}")
        return _try(model_dir)


class SFTDataset(Dataset):
    """prompt+response 拼接；labels 在 prompt 段置 -100（只监督回答）。"""

    def __init__(self, samples, tok, max_len: int = 512):
        self.tok, self.max_len = tok, max_len
        self.items, self.truncated = [], 0
        eos = tok.eos_token_id if tok.eos_token_id is not None else 2
        for s in samples:
            p_ids = tok(s["prompt"], add_special_tokens=False)["input_ids"]
            r_ids = tok(s["response"], add_special_tokens=False)["input_ids"] + [eos]
            # 超长时优先截 prompt 的「中部元素列表」，保留目标行与回答
            if len(p_ids) + len(r_ids) > max_len:
                self.truncated += 1
                keep_r = min(len(r_ids), max_len // 4)
                keep_p = max_len - keep_r
                p_ids = p_ids[:keep_p]
                r_ids = r_ids[:keep_r]
            ids = p_ids + r_ids
            labels = [-100] * len(p_ids) + list(r_ids)
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id: int):
    n = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = n - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)          # pad 不参与损失
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def apply_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model
    targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = sorted(targets & names) or None      # 架构不含这些名字则交 peft 自动推断
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     bias="none", task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def sample_gen(model, tok, prompt: str, max_new: int = 48) -> str:
    model.eval()
    d = next(model.parameters()).device          # 跟随模型所在设备（GPU/CPU）
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(d)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def train(args):
    samples = load_sft_samples(args.data, dedup=not args.no_dedup)
    st = stats(samples)
    print("[data]", json.dumps(st, ensure_ascii=False))
    if not samples:
        print("!! 没有可用样本，先跑真实模式采集。")
        return 1
    # 开训前拦下「训练/推理 prompt 不同源」：这类错配损失曲线看不出来，训完才发现。
    if not check_prompt_consistency(samples, args.max_elements or DEFAULT_MAX_ELEMENTS):
        if not args.force:
            print("!! 已中止。确认无碍可加 --force 强行训练。")
            return 2
        print("!! --force：忽略 prompt 同源校验继续（风险自负）")
    if args.limit:
        samples = samples[:args.limit]

    tok, model = load_model_and_tokenizer(args.model)

    if args.report_lengths:
        lens = [len(tok(s["prompt"], add_special_tokens=False)["input_ids"])
                for s in samples]
        lens.sort()
        n = len(lens)
        print(f"[len] prompt tokens: min={lens[0]} p50={lens[n//2]} "
              f"p90={lens[int(n*0.9)]} max={lens[-1]}  (max_len={args.max_len})")
        over = sum(1 for L in lens if L > args.max_len)
        print(f"[len] 超过 max_len 的样本：{over}/{n} "
              f"({over/n:.0%})——若占比高，必须在 format_scene 侧截断元素数，"
              f"训练与推理同源，不能只在训练侧截。")
        return 0

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = SFTDataset(samples, tok, max_len=args.max_len)
    print(f"[data] 截断样本 {ds.truncated}/{len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    if args.lora:
        model = apply_lora(model, r=args.r, alpha=args.alpha)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[device] 使用 {device}  (cuda_available={torch.cuda.is_available()})")

    probe = samples[0]["prompt"]
    print("\n[before] " + sample_gen(model, tok, probe)[:200])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for ids, labels, attn in dl:
            ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=attn)
            logits = out.logits[:, :-1].float()
            tgt = labels[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   tgt.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep+1}/{args.epochs}  loss={tot/max(nb,1):.4f}  "
              f"({time.time()-t0:.0f}s)")

    print("\n[after ] " + sample_gen(model, tok, probe)[:200])

    if args.lora:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        print(f"[save] LoRA 适配器 -> {args.out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("blobs", "trajectories.jsonl"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（冒烟）")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--max-elements", type=int, default=0,
                    help="推理侧元素数上限，用于与数据里的值做同源校验"
                         f"（默认取 llm_bridge.DEFAULT_MAX_ELEMENTS={DEFAULT_MAX_ELEMENTS}）")
    ap.add_argument("--force", action="store_true",
                    help="忽略 prompt 同源校验强行训练")
    ap.add_argument("--no-lora", dest="lora", action="store_false",
                    help="全参数微调（CPU 上很慢）")
    ap.add_argument("--report-lengths", action="store_true",
                    help="只统计 token 长度分布，不训练")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
