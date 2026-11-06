"""LLM second opinion for emails the model is unsure about.

Works with Azure OpenAI or the OpenAI API. Answers are cached, so the same
email never costs a second call.
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an email security filter. Classify the email as spam (unsolicited, "
    "fraudulent, phishing or bulk advertising) or ham (legitimate). Reply only with "
    'JSON: {"label": "spam" or "ham", "reason": "<one short sentence>"}.'
)
MAX_EMAIL_CHARS = 4000


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResult:
    label: str
    reason: str


def parse_response(content: str | None) -> LLMResult:
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError as exc:
        raise LLMError("LLM did not return valid JSON") from exc
    label = str(data.get("label", "")).strip().lower()
    if label not in {"spam", "ham"}:
        raise LLMError(f"LLM returned an unknown label: {label!r}")
    reason = str(data.get("reason", "")).strip()[:300]
    return LLMResult(label=label, reason=reason)


class LLMClassifier:
    def __init__(self, client, model: str, cache_size: int = 1024):
        self._client = client
        self._model = model
        self._cached_classify = lru_cache(maxsize=cache_size)(self._classify_uncached)
        # Running totals for cost tracking (calls that reached the API).
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def classify(self, subject: str, body: str) -> LLMResult:
        return self._cached_classify(subject, body[:MAX_EMAIL_CHARS])

    def _classify_uncached(self, subject: str, body: str) -> LLMResult:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Subject: {subject}\n\n{body}"},
                ],
            )
        except Exception as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc
        self.usage["calls"] += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return parse_response(response.choices[0].message.content)


def build_llm_classifier(settings) -> LLMClassifier | None:
    """Returns None when no LLM is configured; the service then falls back to rules."""
    if settings.llm_provider == "none":
        logger.info("LLM fallback disabled")
        return None

    import openai

    if settings.llm_provider == "azure":
        if not (settings.azure_endpoint and settings.azure_api_key and settings.azure_deployment):
            raise ValueError(
                "Azure needs AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT"
            )
        client = openai.AzureOpenAI(
            azure_endpoint=settings.azure_endpoint,
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )
        model = settings.azure_deployment
    else:
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai needs OPENAI_API_KEY")
        client = openai.OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )
        model = settings.openai_model

    logger.info("LLM fallback enabled (%s, %s)", settings.llm_provider, model)
    return LLMClassifier(client, model)
