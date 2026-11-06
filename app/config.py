"""Settings read from environment variables (see .env.example)."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LOW = 0.3
DEFAULT_HIGH = 0.8


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    s3_bucket: str
    s3_prefix: str
    low_threshold: float | None
    high_threshold: float | None
    llm_provider: str
    openai_api_key: str
    openai_model: str
    azure_endpoint: str
    azure_api_key: str
    azure_deployment: str
    azure_api_version: str
    llm_timeout_seconds: float
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _env("LLM_PROVIDER").lower()
        if not provider:
            if _env("AZURE_OPENAI_ENDPOINT"):
                provider = "azure"
            elif _env("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "none"
        if provider not in {"azure", "openai", "none"}:
            raise ValueError(f"LLM_PROVIDER must be azure, openai or none, got {provider!r}")

        def optional_float(name: str) -> float | None:
            value = _env(name)
            return float(value) if value else None

        return cls(
            model_dir=Path(_env("MODEL_DIR", "spam_detection_model")),
            s3_bucket=_env("S3_BUCKET"),
            s3_prefix=_env("S3_PREFIX", "spam_detection_model/"),
            low_threshold=optional_float("LOW_THRESHOLD"),
            high_threshold=optional_float("HIGH_THRESHOLD"),
            llm_provider=provider,
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=_env("AZURE_OPENAI_API_KEY"),
            azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            azure_api_version=_env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            llm_timeout_seconds=float(_env("LLM_TIMEOUT_SECONDS", "10")),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
        )


def load_thresholds(settings: Settings) -> tuple[float, float]:
    """Environment overrides win, then thresholds.json from training, then defaults."""
    low, high = DEFAULT_LOW, DEFAULT_HIGH
    path = settings.model_dir / "thresholds.json"
    if path.exists():
        data = json.loads(path.read_text())
        low, high = float(data["low"]), float(data["high"])
    else:
        logger.warning("No %s found, using default thresholds %s/%s", path, low, high)

    if settings.low_threshold is not None:
        low = settings.low_threshold
    if settings.high_threshold is not None:
        high = settings.high_threshold

    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"Thresholds must satisfy 0 <= low <= high <= 1, got {low}, {high}")
    return low, high
