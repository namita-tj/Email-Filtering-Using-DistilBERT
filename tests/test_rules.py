from app.rules import classify_with_rules


def test_two_patterns_mark_spam():
    result = classify_with_rules("Congratulations, you have won! Act now to claim it.")
    assert result.label == "spam"
    assert set(result.matched) == {"prize or lottery", "too-good offer"}
    assert "prize or lottery" in result.reason


def test_single_pattern_is_not_enough():
    result = classify_with_rules("The lottery results are in the newsletter.")
    assert result.label == "ham"
    assert result.matched == ("prize or lottery",)


def test_normal_email_is_ham():
    result = classify_with_rules("Can we move the meeting to Thursday?")
    assert result.label == "ham"
    assert result.reason == "No spam patterns matched"
