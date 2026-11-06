"""Text cleaning shared by training and serving, so both see the same input."""

import re

_HTML = re.compile(r"<.*?>")
_URL = re.compile(r"http\S+")
_NON_LETTERS = re.compile(r"[^a-zA-Z\s]")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _NON_LETTERS.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def combine_subject_and_body(subject: str, body: str) -> str:
    """Builds the model input the same way training does: subject, then body."""
    return f"{clean_text(subject)} {clean_text(body)}".strip()
