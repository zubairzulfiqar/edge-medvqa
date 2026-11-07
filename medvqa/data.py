"""VQA-RAD loading and chat-format prompts.

VQA-RAD (Lau et al., Scientific Data 2018): radiology images (CT, MRI,
X-ray) with clinician-written questions. The Hugging Face copy used here
(flaviagiammarino/vqa-rad, CC0) has 1,793 training and 451 test pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

DATASET = "flaviagiammarino/vqa-rad"
INSTRUCTION = "Answer the question about this radiology image with a single word or a short phrase."


@dataclass
class Sample:
    image: object        # PIL.Image (RGB)
    question: str
    answer: str


def load_split(split: str = "train", limit: int | None = None) -> list[Sample]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [Sample(r["image"].convert("RGB"), r["question"], str(r["answer"])) for r in ds]


def messages(question: str, answer: str | None = None) -> list[dict]:
    """Chat messages in the processor's multimodal format.

    Without an answer: the prompt the model sees at inference.
    With an answer: the full conversation used for training.
    """
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{INSTRUCTION}\nQuestion: {question}"},
    ]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def mask_labels(full_ids: list[int], prompt_len: int, pad_id: int | None) -> list[int]:
    """Training targets: -100 (ignored by the loss) on the prompt and padding,
    so the model is only trained to produce the answer, not to repeat the
    question or the image tokens."""
    labels = list(full_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    if pad_id is not None:
        labels = [-100 if t == pad_id else t for t in labels]
    return labels
