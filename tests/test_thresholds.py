import json

import pytest

from app.config import Settings, load_thresholds
from training.thresholds import DEFAULT_HIGH, DEFAULT_LOW, choose_thresholds


def test_band_covers_the_overlap():
    # Clean ham below 0.2, clean spam above 0.9, mixed cases in between.
    probs = [0.01 * i for i in range(20)] + [0.4, 0.5, 0.6, 0.7] + [0.9 + 0.005 * i for i in range(20)]
    labels = [0] * 20 + [1, 0, 1, 0] + [1] * 20
    t = choose_thresholds(probs, labels, target=0.99, min_support=10)

    assert t.calibrated
    assert t.low == pytest.approx(0.19)
    assert t.high == pytest.approx(0.9)
    assert t.uncertain_share == pytest.approx(4 / 44, abs=1e-4)


def test_perfectly_separated_data_sends_nothing_to_llm():
    probs = [0.1] * 15 + [0.9] * 15
    labels = [0] * 15 + [1] * 15
    t = choose_thresholds(probs, labels, min_support=10)
    assert (t.low, t.high, t.uncertain_share) == (0.1, 0.9, 0.0)


def test_too_little_data_uses_defaults():
    t = choose_thresholds([0.2, 0.8], [0, 1], min_support=10)
    assert not t.calibrated
    assert (t.low, t.high) == (DEFAULT_LOW, DEFAULT_HIGH)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        choose_thresholds([0.1, 0.2], [0])


def _settings(monkeypatch, model_dir, **env):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    for key in ("LOW_THRESHOLD", "HIGH_THRESHOLD", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings.from_env()


def test_load_thresholds_from_training_output(monkeypatch, tmp_path):
    (tmp_path / "thresholds.json").write_text(json.dumps({"low": 0.2, "high": 0.9}))
    assert load_thresholds(_settings(monkeypatch, tmp_path)) == (0.2, 0.9)


def test_env_overrides_file(monkeypatch, tmp_path):
    (tmp_path / "thresholds.json").write_text(json.dumps({"low": 0.2, "high": 0.9}))
    settings = _settings(monkeypatch, tmp_path, HIGH_THRESHOLD="0.95")
    assert load_thresholds(settings) == (0.2, 0.95)


def test_defaults_without_file(monkeypatch, tmp_path):
    assert load_thresholds(_settings(monkeypatch, tmp_path)) == (DEFAULT_LOW, DEFAULT_HIGH)


def test_invalid_override_rejected(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, tmp_path, LOW_THRESHOLD="0.9", HIGH_THRESHOLD="0.1")
    with pytest.raises(ValueError):
        load_thresholds(settings)
