"""LoRA fine-tuning on VQA-RAD.

    python -m medvqa.train --epochs 2 --out runs/lora          # GPU (Colab T4: ~15 min)
    python -m medvqa.train --limit 8 --epochs 1 --device cpu   # quick check

A plain PyTorch loop instead of the HF Trainer: every step is visible.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from .data import load_split
from .model import DEFAULT_MODEL, add_lora, collate, count_parameters, load


def train(args) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, processor = load(args.model, args.device, args.dtype, args.image_size)
    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    trainable, total = count_parameters(model)
    print(f"trainable parameters: {trainable:,} of {total:,} ({trainable / total:.2%})")

    data = load_split("train", args.limit)
    steps_per_epoch = math.ceil(len(data) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs // args.grad_accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(0.05 * total_steps))
    # linear warm-up for the first 5% of steps, then linear decay to 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # float16 gradients can underflow to zero; the scaler multiplies the loss
    # before backward and divides the gradients again before the update.
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device.type == "cuda"))
    model.train()
    history, step, t0 = [], 0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(data)
        for b in range(steps_per_epoch):
            batch = collate(processor, data[b * args.batch_size:(b + 1) * args.batch_size]).to(device)
            if "pixel_values" in batch:
                batch["pixel_values"] = batch["pixel_values"].to(dtype)
            loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            if (b + 1) % args.grad_accum == 0 or b + 1 == steps_per_epoch:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            history.append({"epoch": epoch, "batch": b, "loss": loss.item() * args.grad_accum})
            if b % args.log_every == 0:
                print(f"epoch {epoch} batch {b}/{steps_per_epoch} loss {history[-1]['loss']:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")          # LoRA weights only: a few MB
    processor.save_pretrained(out / "adapter")
    summary = {"model": args.model, "trainable_params": trainable, "total_params": total,
               "train_samples": len(data), "epochs": args.epochs, "steps": step,
               "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"],
               "minutes": round((time.time() - t0) / 60, 1)}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "loss.json").write_text(json.dumps(history))
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA fine-tuning of a small VLM on VQA-RAD")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="use only the first N training pairs")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out", default="runs/lora")
    return p


def main(argv=None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
