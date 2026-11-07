"""End-to-end test with the real SmolVLM-256M and real VQA-RAD samples, on CPU.

Downloads ~0.5 GB, so it only runs when RUN_MODEL_TESTS=1 (CI sets it).
"""
import json
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")


@pytest.fixture(scope="module")
def base():
    from medvqa.model import load
    return load(device="cpu", dtype="float32")


def test_collate_masks_everything_but_the_answer(base):
    from medvqa.data import load_split
    from medvqa.model import collate

    _, processor = base
    samples = load_split("train", limit=2)
    batch = collate(processor, samples)
    for i, s in enumerate(samples):
        kept = [t for t in batch["labels"][i].tolist() if t != -100]
        text = processor.decode(kept, skip_special_tokens=True).strip()
        assert text.lower().startswith(s.answer.lower()[:20]), (text, s.answer)
        assert len(kept) < 0.5 * len(batch["labels"][i])     # image + prompt tokens are masked


def test_lora_targets_only_the_text_decoder(base):
    from medvqa.model import lora_targets

    model, _ = base
    names = lora_targets(model)
    assert names and all("vision" not in n for n in names)
    assert {n.split(".")[-1] for n in names} == {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_train_evaluate_and_compress(tmp_path):
    from medvqa.data import load_split
    from medvqa.evaluate import evaluate, load_variant, to_markdown
    from medvqa.train import build_parser, train

    summary = train(build_parser().parse_args([
        "--limit", "8", "--epochs", "1", "--batch-size", "2", "--device", "cpu", "--dtype", "float32",
        "--log-every", "1", "--out", str(tmp_path / "run")]))
    assert summary["steps"] == 4 and summary["trainable_params"] < 0.05 * summary["total_params"]
    assert (tmp_path / "run" / "adapter" / "adapter_config.json").exists()

    test = load_split("test", limit=3)
    rows = []
    for int8 in (False, True):
        model, processor = load_variant("HuggingFaceTB/SmolVLM-256M-Instruct", str(tmp_path / "run" / "adapter"),
                                        int8, "cpu", "float32")
        rows.append(evaluate(model, processor, test, "int8" if int8 else "fp32", show=0))
    fp32, int8 = rows
    for e in fp32["examples"]:
        print(f"EXAMPLE | {e['q']} | truth: {e['truth']} | fine-tuned (4 steps): {e['pred']}")
    assert all(isinstance(e["pred"], str) for r in rows for e in r["examples"])
    assert int8["size_mb"] < 0.6 * fp32["size_mb"]
    print(to_markdown(rows))
    print(json.dumps([{k: r[k] for k in ("variant", "size_mb", "median_latency_s")} for r in rows]))
