import pytest

from app.explain import Explanation, WordAttribution, merge_wordpieces


def test_wordpieces_are_joined_and_summed():
    tokens = ["[CLS]", "verif", "##y", "your", "pass", "##word", "[SEP]"]
    scores = [0.5, 0.2, 0.1, -0.05, 0.3, 0.1, 0.9]
    words = merge_wordpieces(tokens, scores)
    assert [w.word for w in words] == ["verify", "your", "password"]
    assert [w.weight for w in words] == pytest.approx([0.3, -0.05, 0.4])


def test_leading_continuation_piece_starts_a_word():
    assert merge_wordpieces(["##ing", "now"], [0.1, 0.2])[0].word == "ing"


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        merge_wordpieces(["a"], [0.1, 0.2])


def test_top_sorts_by_absolute_effect():
    explanation = Explanation(
        "integrated_gradients",
        0.9,
        [WordAttribution("hi", 0.01), WordAttribution("meeting", -0.3), WordAttribution("win", 0.2)],
        0.002,
    )
    assert [w.word for w in explanation.top(2)] == ["meeting", "win"]
    data = explanation.to_dict(top_k=1)
    assert data["words"] == [{"word": "meeting", "weight": -0.3}]
    assert data["convergence_delta"] == 0.002
