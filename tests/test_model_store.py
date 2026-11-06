import pytest

from app.model_store import ModelNotAvailableError, download_model, ensure_model


class FakeS3:
    def __init__(self, pages):
        self.pages = pages
        self.downloads = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):  # noqa: N803 (boto3 argument names)
        return self.pages

    def download_file(self, bucket, key, path):
        self.downloads.append(key)
        with open(path, "w") as f:
            f.write(key)


def page(*keys):
    return {"Contents": [{"Key": k} for k in keys]}


def test_downloads_all_pages_and_keeps_subfolders(tmp_path):
    s3 = FakeS3([
        page("model/", "model/config.json", "model/tf_model.h5"),
        page("model/tokenizer/vocab.txt"),
    ])
    files = download_model("bucket", "model/", tmp_path, client=s3)
    assert len(files) == 3
    assert (tmp_path / "tokenizer" / "vocab.txt").is_file()
    assert "model/" not in s3.downloads


def test_empty_bucket_raises(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="No objects"):
        download_model("bucket", "model/", tmp_path, client=FakeS3([{}]))


def test_incomplete_download_raises(tmp_path):
    s3 = FakeS3([page("model/config.json")])
    with pytest.raises(ModelNotAvailableError, match="tf_model.h5"):
        download_model("bucket", "model/", tmp_path, client=s3)


def test_path_traversal_blocked(tmp_path):
    s3 = FakeS3([page("model/../../evil.txt")])
    with pytest.raises(ModelNotAvailableError, match="outside"):
        download_model("bucket", "model/", tmp_path / "dest", client=s3)


def test_existing_model_is_not_downloaded(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tf_model.h5").write_text("weights")
    s3 = FakeS3([page("model/config.json")])
    ensure_model(tmp_path, "bucket", "model/", client=s3)
    assert s3.downloads == []


def test_missing_model_without_bucket_explains_fix(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="training.train"):
        ensure_model(tmp_path, "", "model/")


def test_missing_model_with_bucket_downloads(tmp_path):
    s3 = FakeS3([page("model/config.json", "model/tf_model.h5")])
    ensure_model(tmp_path, "bucket", "model/", client=s3)
    assert len(s3.downloads) == 2
