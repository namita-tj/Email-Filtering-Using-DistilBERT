"""Makes sure the trained model is on disk, downloading it from Azure Blob Storage or S3 if needed."""

import logging
from pathlib import Path
from urllib.parse import urlparse

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


def parse_blob_url(url: str) -> tuple[str, str, str]:
    """'https://acct.blob.core.windows.net/models/spam_detection_model/'
    -> ('https://acct.blob.core.windows.net', 'models', 'spam_detection_model/')"""
    parsed = urlparse(url)
    parts = parsed.path.lstrip("/").split("/", 1)
    if parsed.scheme != "https" or not parsed.netloc or not parts[0]:
        raise ModelNotAvailableError(
            "MODEL_BLOB_URL must look like "
            f"https://<account>.blob.core.windows.net/<container>/<folder>/, got {url!r}"
        )
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"https://{parsed.netloc}", parts[0], prefix


def download_model_from_blob(url: str, dest: Path, container_client=None) -> list[Path]:
    """Downloads every blob under the folder in `url`, keeping sub-folders.

    Authenticates with DefaultAzureCredential: the container app's managed identity in
    Azure (AZURE_CLIENT_ID selects the user-assigned one), or `az login` on a laptop.
    No storage keys are involved.
    """
    account_url, container, prefix = parse_blob_url(url)
    if container_client is None:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import ContainerClient

        container_client = ContainerClient(account_url, container, credential=DefaultAzureCredential())

    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for blob in container_client.list_blobs(name_starts_with=prefix or None):
        name = blob.name
        if name.endswith("/"):
            continue
        relative = name[len(prefix):].lstrip("/") or Path(name).name
        target = (dest / relative).resolve()
        if not target.is_relative_to(dest):
            raise ModelNotAvailableError(f"Refusing to write outside {dest}: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            container_client.download_blob(name).readinto(f)
        downloaded.append(target)

    if not downloaded:
        raise ModelNotAvailableError(f"No blobs found under {url}")
    if not model_is_present(dest):
        missing = [n for n in REQUIRED_FILES if not (dest / n).is_file()]
        raise ModelNotAvailableError(f"Download incomplete, missing: {', '.join(missing)}")
    logger.info("Downloaded %d model files from %s", len(downloaded), url)
    return downloaded


def ensure_model(model_dir: Path, bucket: str, prefix: str, client=None, blob_url: str = "",
                 container_client=None) -> None:
    """Local folder first, then Azure Blob Storage (MODEL_BLOB_URL), then S3 (S3_BUCKET)."""
    if model_is_present(model_dir):
        logger.info("Using local model in %s", model_dir)
        return
    if blob_url:
        download_model_from_blob(blob_url, model_dir, container_client=container_client)
        return
    if not bucket:
        raise ModelNotAvailableError(
            f"No model in {model_dir} and neither MODEL_BLOB_URL nor S3_BUCKET is set. "
            "Run `python -m training.train` or set one of them."
        )
    download_model(bucket, prefix, model_dir, client=client)
