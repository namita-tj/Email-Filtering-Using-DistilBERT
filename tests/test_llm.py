from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm import LLMClassifier, LLMError, build_llm_classifier, parse_response


def fake_client(content=None, error=None):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def test_parse_valid_response():
    result = parse_response('{"label": "SPAM", "reason": "Fake invoice"}')
    assert (result.label, result.reason) == ("spam", "Fake invoice")


@pytest.mark.parametrize("content", [None, "not json", '{"label": "maybe"}', "{}"])
def test_parse_rejects_bad_responses(content):
    with pytest.raises(LLMError):
        parse_response(content)


def test_classifier_sends_json_request_and_caches():
    client, calls = fake_client('{"label": "ham", "reason": "Team update"}')
    llm = LLMClassifier(client, model="gpt-4o-mini")

    first = llm.classify("Standup", "Notes from today")
    second = llm.classify("Standup", "Notes from today")

    assert first == second
    assert first.label == "ham"
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["temperature"] == 0


def test_long_emails_are_truncated():
    client, calls = fake_client('{"label": "ham", "reason": "ok"}')
    LLMClassifier(client, model="m").classify("s", "x" * 10_000)
    user_message = calls[0]["messages"][1]["content"]
    assert len(user_message) < 4_100


def test_api_errors_become_llm_error():
    client, _ = fake_client(error=TimeoutError("slow"))
    with pytest.raises(LLMError, match="TimeoutError"):
        LLMClassifier(client, model="m").classify("s", "b")


def test_failures_are_not_cached():
    client, calls = fake_client(error=TimeoutError())
    llm = LLMClassifier(client, model="m")
    for _ in range(2):
        with pytest.raises(LLMError):
            llm.classify("s", "b")
    assert len(calls) == 2


def _settings(monkeypatch, **env):
    for key in ("LLM_PROVIDER", "OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings.from_env()


def test_no_provider_means_no_llm(monkeypatch):
    settings = _settings(monkeypatch)
    assert settings.llm_provider == "none"
    assert build_llm_classifier(settings) is None


def test_azure_detected_and_built(monkeypatch):
    settings = _settings(
        monkeypatch,
        AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
        AZURE_OPENAI_API_KEY="test-key",
        AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini",
    )
    assert settings.llm_provider == "azure"
    assert isinstance(build_llm_classifier(settings), LLMClassifier)


def test_openai_built(monkeypatch):
    settings = _settings(monkeypatch, OPENAI_API_KEY="test-key")
    assert settings.llm_provider == "openai"
    assert isinstance(build_llm_classifier(settings), LLMClassifier)


def test_incomplete_azure_config_fails_fast(monkeypatch):
    settings = _settings(monkeypatch, LLM_PROVIDER="azure")
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        build_llm_classifier(settings)


def test_openai_without_key_fails_fast(monkeypatch):
    settings = _settings(monkeypatch, LLM_PROVIDER="openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm_classifier(settings)


def test_unknown_provider_rejected(monkeypatch):
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        _settings(monkeypatch, LLM_PROVIDER="gemini")


def test_token_usage_is_tracked():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content='{"label": "spam", "reason": "x"}')
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=15)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = LLMClassifier(client, model="m")
    llm.classify("a", "b")
    llm.classify("a", "b")  # cached: no new call, no new tokens
    llm.classify("c", "d")
    assert llm.usage == {"calls": 2, "prompt_tokens": 240, "completion_tokens": 30}
