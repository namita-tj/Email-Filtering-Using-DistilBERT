"""Zero-shot classification with an open-weight LLM running locally.

No API key, no cost, and the email never leaves the machine. Uses the same system
prompt as the API fallback in app/llm.py, so the two are directly comparable.
Default model: Qwen2.5-3B-Instruct (Apache 2.0, about 6 GB in float16, fits a free
Colab T4). Needs PyTorch and transformers.
"""

import re

from app.llm import MAX_EMAIL_CHARS, SYSTEM_PROMPT, LLMError, LLMResult, parse_response

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
_LABEL_WORD = re.compile(r"\b(spam|ham)\b", re.IGNORECASE)


def parse_local_response(text: str | None) -> LLMResult:
    """Small models sometimes wrap the JSON in code fences or add text around it.

    Takes the first valid JSON object with a spam/ham label; failing that, the first
    'spam' or 'ham' word. Anything else is an error.
    """
    text = text or ""
    for match in _JSON_OBJECT.finditer(text):
        try:
            return parse_response(match.group(0))
        except LLMError:
            continue
    word = _LABEL_WORD.search(text)
    if word:
        return LLMResult(label=word.group(1).lower(), reason="")
    raise LLMError(f"Local model gave no spam/ham label: {text[:80]!r}")


class LocalLLMClassifier:
    """Batched greedy generation with a Hugging Face causal language model."""

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        device: str | None = None,
        batch_size: int = 8,
        max_new_tokens: int = 48,
        tokenizer=None,
        model=None,
    ):
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if tokenizer is None or model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        # Decoder-only models must be padded on the left for batched generation.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()
        self.name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _prompt(self, subject: str, body: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Subject: {subject}\n\n{body[:MAX_EMAIL_CHARS]}"},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: list[str]) -> list[str]:
        """Raw completions, in the order of `prompts`."""
        torch = self._torch
        # Similar lengths in one batch means less padding and faster generation.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outputs: dict[int, str] = {}
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                # Only these two: some tokenizers add token_type_ids, which generate() rejects.
                generated = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = generated[:, batch["input_ids"].shape[1] :]
            self.usage["calls"] += len(idx)
            self.usage["prompt_tokens"] += int(batch["attention_mask"].sum())
            self.usage["completion_tokens"] += int((new_tokens != self.tokenizer.pad_token_id).sum())
            for i, text in zip(
                idx, self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True), strict=True
            ):
                outputs[i] = text
        return [outputs[i] for i in range(len(prompts))]

    def classify(self, subject: str, body: str) -> LLMResult:
        return parse_local_response(self.generate([self._prompt(subject, body)])[0])

    def classify_many(self, rows: list[dict]) -> list[LLMResult | LLMError]:
        """One result per row; an LLMError in place of rows the model did not label."""
        texts = self.generate([self._prompt(r["subject"], r["body"]) for r in rows])
        results: list[LLMResult | LLMError] = []
        for text in texts:
            try:
                results.append(parse_local_response(text))
            except LLMError as exc:
                results.append(exc)
        return results
