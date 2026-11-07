"""Azure pieces: model download from Blob Storage and Application Insights telemetry.
No Azure account needed: the blob container and the exporter are replaced by fakes."""

import logging

import pytest
from fastapi.testclient import TestClient

from app import telemetry
from app.main import create_app
from app.model_store import ModelNotAvailableError, download_model_from_blob, ensure_model, parse_blob_url
from app.pipeline import SpamFilter

URL = "https://acct.blob.core.windows.net/models/spam_detection_model/"


class _Blob:
    def __init__(self, name):
        self.name = name


class FakeContainer:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.prefixes = []

    def list_blobs(self, name_starts_with=None):
        self.prefixes.append(name_starts_with)
        return [_Blob(n) for n in self.files if not name_starts_with or n.startswith(name_starts_with)]

    def download_blob(self, name):
        data = self.files[name]

        class _Stream:
            def readinto(self, f):
                f.write(data)

        return _Stream()


MODEL = {
    "spam_detection_model/config.json": b"{}",
    "spam_detection_model/tf_model.h5": b"weights",
    "spam_detection_model/tokenizer/vocab.txt": b"a\nb",
    "other_model/config.json": b"not this one",
}


def test_parse_blob_url():
    assert parse_blob_url(URL) == ("https://acct.blob.core.windows.net", "models", "spam_detection_model/")
    assert parse_blob_url("https://acct.blob.core.windows.net/models/m") == (
        "https://acct.blob.core.windows.net", "models", "m/")
    for bad in ("http://acct.blob.core.windows.net/models/", "https://acct.blob.core.windows.net/", "s3://x/y"):
        with pytest.raises(ModelNotAvailableError):
            parse_blob_url(bad)


def test_downloads_the_folder_and_keeps_subfolders(tmp_path):
    container = FakeContainer(MODEL)
    files = download_model_from_blob(URL, tmp_path / "m", container_client=container)
    assert sorted(p.relative_to((tmp_path / "m").resolve()).as_posix() for p in files) == [
        "config.json", "tf_model.h5", "tokenizer/vocab.txt"]
    assert (tmp_path / "m" / "tf_model.h5").read_bytes() == b"weights"
    assert container.prefixes == ["spam_detection_model/"]


def test_incomplete_or_empty_folder_raises(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="missing: tf_model.h5"):
        download_model_from_blob(URL, tmp_path, container_client=FakeContainer(
            {"spam_detection_model/config.json": b"{}"}))
    with pytest.raises(ModelNotAvailableError, match="No blobs found"):
        download_model_from_blob(URL, tmp_path, container_client=FakeContainer({}))


def test_path_traversal_is_blocked(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="Refusing"):
        download_model_from_blob(URL, tmp_path, container_client=FakeContainer(
            {"spam_detection_model/../../evil.txt": b"x"}))


def test_ensure_model_prefers_blob_over_s3(tmp_path):
    class NoS3:
        def get_paginator(self, *_):
            raise AssertionError("S3 must not be used when MODEL_BLOB_URL is set")

    ensure_model(tmp_path / "m", "bucket", "prefix/", client=NoS3(), blob_url=URL,
                 container_client=FakeContainer(MODEL))
    assert (tmp_path / "m" / "config.json").exists()


def test_ensure_model_error_names_both_sources(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="MODEL_BLOB_URL nor S3_BUCKET"):
        ensure_model(tmp_path, "", "")


class FakeModel:
    def spam_probability(self, text):
        return 0.95


def test_telemetry_is_off_without_connection_string(monkeypatch):
    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    assert telemetry.telemetry_enabled() is False
    assert telemetry.setup_telemetry(object()) is False  # never touches the app or Azure packages


def test_telemetry_instruments_the_app_and_sends_prediction_dimensions(monkeypatch):
    calls = {}

    def fake_configure(**kwargs):
        calls.update(kwargs)

    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING",
                       "InstrumentationKey=00000000-0000-0000-0000-000000000000")
    monkeypatch.setattr(telemetry, "_configured", False)
    monkeypatch.setattr("azure.monitor.opentelemetry.configure_azure_monitor", fake_configure)
    app = create_app(SpamFilter(model=FakeModel()))
    assert calls == {"logger_name": "email_filter"}
    assert getattr(app, "_is_instrumented_by_opentelemetry", False)

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logging.getLogger("email_filter").addHandler(handler)
    logging.getLogger("email_filter").setLevel(logging.INFO)
    try:
        r = TestClient(app).post("/predict", json={"subject": "Win", "text": "Claim your prize now"})
    finally:
        logging.getLogger("email_filter").removeHandler(handler)
    assert r.status_code == 200
    rec = next(x for x in records if x.getMessage().startswith("label="))
    assert (rec.label, rec.source, rec.spam_probability) == ("spam", "model", 0.95)


def test_health_still_works_with_telemetry_off(monkeypatch):
    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    r = TestClient(create_app(SpamFilter(model=FakeModel()))).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
