import pytest

from owl.scorers import binary_label, exact_match, strip_think
from owl.types import ModelOutput, Sample

score = binary_label("1")


def run(text: str, target: str = "1", probability: float | None = None):
    return score(Sample(id="0", input="x", target=target), ModelOutput(text=text, probability=probability))


@pytest.mark.parametrize(
    "text,pred",
    [
        ("1", "1"),
        ("0", "0"),
        ("  1\n", "1"),
        ("**0**", "0"),
        ("The answer is 1.", "1"),
        ("0 or 1", "1"),  # last digit wins
        ("10", "0"),
        ("<think>maybe 0, maybe 0</think>1", "1"),
        ("<think>\nline one\nline 0\n</think>\n\n0", "0"),
        ("<think>a</think><think>b 0</think>1", "1"),
    ],
)
def test_prediction_parsing(text, pred):
    assert run(text).pred == pred


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I am not sure.",
        "yes",
        "<think>the answer is 1</think>",  # digits only inside a think block
        "<think>1",  # think block that never closed
        "2 3 4",
    ],
)
def test_missing_digit_is_unparseable_and_incorrect(text):
    s = run(text)
    assert s.pred is None
    assert s.unparseable is True
    assert s.correct is False
    assert s.value == 0.0


def test_correctness_against_target():
    assert run("1", target="1").correct
    assert not run("1", target="0").correct
    assert run("0", target="0").correct


def test_parseable_output_is_not_flagged():
    assert run("1").unparseable is False


def test_probability_is_carried_into_metadata():
    s = run("1", probability=0.9709)
    assert s.metadata["probability"] == 0.9709
    assert "0.9709" in s.explanation
    assert "probability" not in run("1").metadata


def test_binary_label_rejects_bad_positive_class():
    with pytest.raises(ValueError):
        binary_label("2")


def test_strip_think():
    assert strip_think("<think>x</think>y") == "y"
    assert strip_think("a<think>never closed") == "a"


def test_exact_match():
    em = exact_match()
    sample = Sample(id="0", input="q", target="Paris")
    assert em(sample, ModelOutput(text=" paris\n")).correct
    assert not em(sample, ModelOutput(text="London")).correct
    assert not exact_match(case_sensitive=True)(sample, ModelOutput(text="paris")).correct
    assert em(sample, ModelOutput(text="<think>hmm</think>Paris")).correct
