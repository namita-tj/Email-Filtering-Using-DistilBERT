"""FastAPI service for the hybrid spam filter.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, load_thresholds
from app.pipeline import SpamFilter

logger = logging.getLogger("email_filter")


class PredictRequest(BaseModel):
    subject: str = Field(default="", max_length=1000, description="Email subject line")
    text: str = Field(min_length=1, max_length=20000, description="Email body")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class PredictResponse(BaseModel):
    label: str = Field(description="spam or ham")
    source: str = Field(description="Which stage decided: model, llm or rules")
    spam_probability: float = Field(description="DistilBERT spam probability")
    reason: str | None = Field(default=None, description="Explanation from the LLM or rules")


def build_spam_filter(settings: Settings) -> SpamFilter:
    from app.classifier import DistilBertClassifier
    from app.llm import build_llm_classifier
    from app.model_store import ensure_model

    ensure_model(settings.model_dir, settings.s3_bucket, settings.s3_prefix)
    low, high = load_thresholds(settings)
    return SpamFilter(
        model=DistilBertClassifier(settings.model_dir),
        llm=build_llm_classifier(settings),
        low=low,
        high=high,
    )


def create_app(spam_filter: SpamFilter | None = None) -> FastAPI:
    """`spam_filter` is injected in tests; in production it is built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.spam_filter is None:
            settings = Settings.from_env()
            logging.basicConfig(
                level=settings.log_level,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
            app.state.spam_filter = build_spam_filter(settings)
            logger.info("Spam filter ready")
        yield

    app = FastAPI(
        title="Email Filtering API",
        description="DistilBERT spam classifier with an LLM fallback for uncertain emails.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.spam_filter = spam_filter

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health")
    def health():
        spam_filter = app.state.spam_filter
        if spam_filter is None:
            return JSONResponse(status_code=503, content={"status": "loading"})
        return {
            "status": "ok",
            "llm_fallback": spam_filter.llm is not None,
            "thresholds": {"low": spam_filter.low, "high": spam_filter.high},
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        prediction = app.state.spam_filter.classify(request.subject, request.text)
        logger.info(
            "label=%s source=%s p=%.3f",
            prediction.label,
            prediction.source,
            prediction.spam_probability,
        )
        return PredictResponse(
            label=prediction.label,
            source=prediction.source,
            spam_probability=round(prediction.spam_probability, 4),
            reason=prediction.reason,
        )

    return app


app = create_app()
