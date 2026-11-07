"""Loading the vision-language model, adding LoRA, batching and answering.

Default model: SmolVLM-256M-Instruct (Hugging Face, Apache 2.0): a SigLIP
vision encoder (~93M) + SmolLM2 text decoder, about 0.26B parameters. Each
512x512 image becomes 64 visual tokens, which keeps it fast enough for a CPU.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from .data import Sample, mask_labels, messages

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(model_name: str = DEFAULT_MODEL, device: str = "auto", dtype: str = "auto", image_size: int = 512):
    """Return (model, processor). Works across transformers versions, whose
    auto-class for image-text models has been renamed over time."""
    import transformers as tf

    processor = tf.AutoProcessor.from_pretrained(model_name, size={"longest_edge": image_size},
                                                 do_image_splitting=False)
    model, last_err = None, None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForMultimodalLM"):
        cls = getattr(tf, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_name)
            break
        except (ValueError, KeyError) as exc:  # class does not cover this architecture
            last_err = exc
    if model is None:
        raise RuntimeError(f"Could not load {model_name}: {last_err}")

    dev = pick_device(device)
    if dtype == "auto":
        if dev.type != "cuda":
            dtype = "float32"
        else:
            # bf16 only where the hardware has it (Ampere, compute capability 8.0+).
            # Older GPUs such as the T4 report bf16 "supported" through slow emulation,
            # so they get float16. PEFT keeps the LoRA weights in float32 either way.
            dtype = "bfloat16" if torch.cuda.get_device_capability(dev)[0] >= 8 else "float16"
    model.to(device=dev, dtype=getattr(torch, dtype))
    return model, processor


def lora_targets(model: nn.Module) -> list[str]:
    """Attention projections of the TEXT decoder only.

    The vision encoder keeps its pre-trained weights: it already sees images
    well, and freezing it saves memory. Module names are looked up instead of
    hard-coded, so this survives renames between library versions.
    """
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.split(".")[-1] in ATTN_PROJ and "vision" not in n]
    if not names:
        raise RuntimeError("No text attention projections found for LoRA")
    return names


def add_lora(model: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=lora_targets(model),
                     bias="none")
    return get_peft_model(model, cfg)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ------------------------------------------------------------- batching ---

def prompt_text(processor, question: str) -> str:
    return processor.apply_chat_template(messages(question), add_generation_prompt=True)


def full_text(processor, question: str, answer: str) -> str:
    return processor.apply_chat_template(messages(question, answer), add_generation_prompt=False)


def collate(processor, samples: list[Sample]) -> dict:
    """Tokenise a training batch and build answer-only labels."""
    tok = processor.tokenizer
    tok.padding_side = "right"
    fulls = [full_text(processor, s.question, s.answer) for s in samples]
    batch = processor(text=fulls, images=[[s.image] for s in samples], padding=True, return_tensors="pt")
    labels = []
    for i, s in enumerate(samples):
        p = processor(text=[prompt_text(processor, s.question)], images=[[s.image]], return_tensors="pt")
        prompt_len = p["input_ids"].shape[1]
        labels.append(mask_labels(batch["input_ids"][i].tolist(), prompt_len, tok.pad_token_id))
    batch["labels"] = torch.tensor(labels)
    return batch


@torch.no_grad()
def answer(model, processor, sample: Sample, max_new_tokens: int = 16) -> tuple[str, float]:
    """Greedy answer for one sample, and the time it took (seconds)."""
    device = next(model.parameters()).device
    inputs = processor(text=[prompt_text(processor, sample.question)], images=[[sample.image]],
                       return_tensors="pt").to(device)
    dtype = next(model.parameters()).dtype
    if "pixel_values" in inputs and dtype.is_floating_point:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.perf_counter() - t0
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip(), dt
