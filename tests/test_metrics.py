import math

from medvqa.data import mask_labels, messages
from medvqa.metrics import first_answer, is_closed, normalize, recall, score


def test_normalize():
    assert normalize("The Left  Lung.") == "left lung"
    assert normalize("Yes!") == "yes"


def test_first_answer_handles_chatty_models():
    assert first_answer("Yes, there is a fracture.") == "yes"
    assert first_answer("No. The heart is normal") == "no"
    assert first_answer("Right kidney\nIt appears enlarged") == "right kidney"


def test_recall_gives_partial_credit():
    assert recall("lung", "left lung") == 0.5
    assert recall("the left lung", "left lung") == 1.0
    assert recall("", "liver") == 0.0


def test_score_splits_closed_and_open():
    res = score(["yes", "No, it is not", "spleen", "left kidney"],
                ["yes", "yes", "spleen", "right kidney"])
    assert res["n_closed"] == 2 and res["closed_accuracy"] == 0.5
    assert res["n_open"] == 2 and res["open_exact"] == 0.5 and res["open_recall"] == 0.75
    assert res["overall_exact"] == 0.5
    assert is_closed("No") and not is_closed("no fracture")


def test_score_with_only_closed_questions_reports_nan_for_open():
    res = score(["yes"], ["yes"])
    assert res["closed_accuracy"] == 1.0 and math.isnan(res["open_exact"])


def test_mask_labels_keeps_only_the_answer():
    full = [1, 2, 3, 4, 50, 51, 9, 0, 0]          # prompt (4 tokens) + answer + end + padding
    assert mask_labels(full, prompt_len=4, pad_id=0) == [-100] * 4 + [50, 51, 9, -100, -100]


def test_messages_format():
    m = messages("Is there a fracture?")
    assert m[0]["role"] == "user" and m[0]["content"][0] == {"type": "image"}
    assert "Is there a fracture?" in m[0]["content"][1]["text"]
    assert messages("q", "yes")[-1] == {"role": "assistant", "content": [{"type": "text", "text": "yes"}]}
