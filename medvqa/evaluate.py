"""Evaluate a model (base, fine-tuned, or compressed) on the VQA-RAD test set.

    python -m medvqa.evaluate                               # zero-shot base model
    python -m medvqa.evaluate --adapter runs/lora/adapter   # fine-tuned
    python -m medvqa.evaluate --adapter runs/lora/adapter --int8 --device cpu   # edge version
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .compress import model_size_mb, quantize_int8
from .data import load_split
from .metrics import score
from .model import DEFAULT_MODEL, answer, load


def load_variant(model_name: str, adapter: str | None, int8: bool, device: str, dtype: str):
    model, processor = load(model_name, "cpu" if int8 else device, "float32" if int8 else dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()  # fold LoRA into the weights
    if int8:
        model = quantize_int8(model)
    model.eval()
    return model, processor


def evaluate(model, processor, samples, label: str, max_new_tokens: int = 16, show: int = 3) -> dict:
    preds, times = [], []
    for i, s in enumerate(samples):
        text, dt = answer(model, processor, s, max_new_tokens)
        preds.append(text)
        times.append(dt)
        if i < show:
            print(f"  Q: {s.question}\n  truth: {s.answer} | model: {text}")
    res = score(preds, [s.answer for s in samples])
    res.update({"variant": label, "median_latency_s": statistics.median(times),
                "size_mb": model_size_mb(model), "device": str(next(model.parameters()).device)})
    res["examples"] = [{"q": s.question, "truth": s.answer, "pred": p} for s, p in zip(samples[:20], preds[:20])]
    return res


def to_markdown(rows: list[dict]) -> str:
    pct = lambda x: "n/a" if x != x else f"{x:.1%}"  # noqa: E731
    lines = ["| Variant | Closed acc. | Open exact | Open recall | Size (MB) | Latency / answer | Device |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {pct(r['closed_accuracy'])} | {pct(r['open_exact'])} | "
                     f"{pct(r['open_recall'])} | {r['size_mb']:.0f} | {r['median_latency_s']:.2f} s | {r['device']} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate on VQA-RAD test")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--int8", action="store_true", help="CPU dynamic int8 quantisation")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None, help="append the result to this JSON list")
    args = p.parse_args(argv)

    label = args.label or ("fine-tuned" if args.adapter else "zero-shot") + (" int8" if args.int8 else "")
    model, processor = load_variant(args.model, args.adapter, args.int8, args.device, args.dtype)
    res = evaluate(model, processor, load_split("test", args.limit), label)
    print(to_markdown([res]))
    if args.out:
        path = Path(args.out)
        rows = json.loads(path.read_text()) if path.exists() else []
        rows.append(res)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
