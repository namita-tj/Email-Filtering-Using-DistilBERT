"""Makes sure the trained model is on disk, downloading it from S3 if needed."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_FILES = ("config.json", "tf_model.h5")


class ModelNotAvailableError(RuntimeError):
    pass


def model_is_present(model_dir: Path) -> bool:
    return all((model_dir / name).is_file() for name in REQUIRED_FILES)


def download_model(bucket: str, prefix: str, dest: Path, client=None) -> list[Path]:
    """Downloads every object under `prefix`, keeping sub-folders, and checks the result."""
    if client is None:
        import boto3

        client = boto3.client("s3")

    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            relative = key[len(prefix):].lstrip("/") or Path(key).name
            target = (dest / relative).resolve()
            if not target.is_relative_to(dest):
                raise ModelNotAvailableError(f"Refusing to write outside {dest}: {key}")
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            downloaded.append(target)

    if not downloaded:
        raise ModelNotAvailableError(f"No objects found under s3://{bucket}/{prefix}")
    if not model_is_present(dest):
        missing = [n for n in REQUIRED_FILES if not (dest / n).is_file()]
        raise ModelNotAvailableError(f"Download incomplete, missing: {', '.join(missing)}")

    logger.info("Downloaded %d model files from s3://%s/%s", len(downloaded), bucket, prefix)
    return downloaded


def ensure_model(model_dir: Path, bucket: str, prefix: str, client=None) -> None:
    if model_is_present(model_dir):
        logger.info("Using local model in %s", model_dir)
        return
    if not bucket:
        raise ModelNotAvailableError(
            f"No model in {model_dir} and S3_BUCKET is not set. "
            "Run `python -m training.train` or set S3_BUCKET."
        )
    download_model(bucket, prefix, model_dir, client=client)
