"""VQA-RAD metrics (pure Python).

VQA-RAD has two kinds of questions:

* closed - yes/no questions ("Is there a pneumothorax?")
* open   - free answers ("What organ is enlarged?" -> "spleen")

Following common practice for this benchmark we report:
* closed accuracy  - exact match after normalisation
* open exact match - exact match after normalisation
* open recall      - share of ground-truth words that appear in the
                     prediction (gives partial credit to "left lung" vs "lung")
"""
from __future__ import annotations

import re
import string

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def normalize(text: str) -> str:
    """lower-case, strip punctuation and articles, collapse spaces."""
    words = text.lower().translate(_PUNCT).split()
    return " ".join(w for w in words if w not in _ARTICLES)


def is_closed(answer: str) -> bool:
    return normalize(answer) in {"yes", "no"}


def first_answer(prediction: str) -> str:
    """Models sometimes add a sentence; keep the first line / clause."""
    pred = prediction.strip().split("\n")[0]
    pred = re.split(r"[.;]", pred)[0]
    norm = normalize(pred)
    # "Yes, there is ..." -> "yes"
    first = norm.split(" ")[0] if norm else ""
    return first if first in {"yes", "no"} else norm


def recall(prediction: str, truth: str) -> float:
    gt = normalize(truth).split()
    if not gt:
        return 0.0
    pred = set(normalize(prediction).split())
    return sum(w in pred for w in gt) / len(gt)


def score(predictions: list[str], answers: list[str]) -> dict:
    closed_hits, open_exact, open_rec = [], [], []
    for p, a in zip(predictions, answers):
        p_norm, a_norm = first_answer(p), normalize(a)
        if is_closed(a):
            closed_hits.append(p_norm == a_norm)
        else:
            open_exact.append(p_norm == a_norm)
            open_rec.append(recall(p, a))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {
        "n": len(answers), "n_closed": len(closed_hits), "n_open": len(open_exact),
        "closed_accuracy": mean(closed_hits),
        "open_exact": mean(open_exact),
        "open_recall": mean(open_rec),
        "overall_exact": mean(closed_hits + open_exact),
    }
