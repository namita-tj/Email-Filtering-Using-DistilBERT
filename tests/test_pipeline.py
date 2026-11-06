import pytest

from app.pipeline import SpamFilter

PHISHING = "You have won a prize! Verify your account and pay with gift cards."


@pytest.mark.parametrize(
    ("probability", "label"),
    [(0.95, "spam"), (0.8, "spam"), (0.05, "ham"), (0.3, "ham")],
)
def test_confident_model_decides_alone(fake_model, fake_llm, probability, label):
    llm = fake_llm()
    result = SpamFilter(fake_model(probability), llm, low=0.3, high=0.8).classify("Hi", "text")
    assert result.label == label
    assert result.source == "model"
    assert result.spam_probability == probability
    assert llm.calls == 0


def test_uncertain_email_goes_to_llm(fake_model, fake_llm):
    llm = fake_llm(label="spam", reason="asks for gift cards")
    result = SpamFilter(fake_model(0.5), llm).classify("Hello", "Please buy gift cards")
    assert (result.label, result.source, result.reason) == ("spam", "llm", "asks for gift cards")
    assert llm.calls == 1


def test_llm_failure_falls_back_to_rules(fake_model, fake_llm):
    result = SpamFilter(fake_model(0.5), fake_llm(fail=True)).classify("Winner", PHISHING)
    assert result.source == "rules"
    assert result.label == "spam"


def test_no_llm_configured_uses_rules(fake_model):
    result = SpamFilter(fake_model(0.5), llm=None).classify("Lunch", "See you at noon?")
    assert (result.label, result.source) == ("ham", "rules")


def test_model_sees_cleaned_subject_and_body(fake_model):
    model = fake_model(0.1)
    SpamFilter(model).classify("Re: <b>Invoice</b>", "Visit http://x.io now!!")
    assert model.calls == ["Re Invoice Visit now"]


@pytest.mark.parametrize(("low", "high"), [(0.9, 0.1), (-0.1, 0.5), (0.2, 1.5)])
def test_invalid_thresholds_rejected(fake_model, low, high):
    with pytest.raises(ValueError):
        SpamFilter(fake_model(0.5), low=low, high=high)
