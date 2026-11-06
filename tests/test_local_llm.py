import pytest

from app.llm import LLMError
from app.local_llm import parse_local_response


# ── Parsing what small models actually write ──────────────────────────────────
@pytest.mark.parametrize(
    ("text", "label"),
    [
        ('{"label": "spam", "reason": "prize scam"}', "spam"),
        ('```json\n{"label": "ham", "reason": "work email"}\n```', "ham"),
        ('Sure! Here is the answer: {"label": "SPAM", "reason": "x"} Hope this helps.', "spam"),
        ('{"verdict": "unsure"} then {"label": "ham"}', "ham"),
        ("This email is ham.", "ham"),
        ("Label: spam", "spam"),
    ],
)
def test_parse_local_response(text, label):
    assert parse_local_response(text).label == label


@pytest.mark.parametrize("text", ["", None, "I cannot classify this email.", '{"label": "maybe"}'])
def test_parse_local_response_without_label_raises(text):
    with pytest.raises(LLMError):
        parse_local_response(text)


# ── Batching keeps answers attached to the right email ────────────────────────
torch = pytest.importorskip("torch")


class _Batch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """One 'token' per prompt: its length. The model's answer depends on it."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[-1]["content"]

    def __call__(self, texts, return_tensors, padding):
        ids = torch.tensor([[len(t)] for t in texts])
        return _Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

    def batch_decode(self, tokens, skip_special_tokens):
        return [f'{{"label": "{"spam" if int(t[0]) % 2 else "ham"}"}}' for t in tokens]


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        return torch.cat([input_ids, input_ids], dim=1)  # "answers" with its input length


def test_batched_answers_map_back_to_their_emails():
    from app.local_llm import LocalLLMClassifier

    clf = LocalLLMClassifier(device="cpu", batch_size=3, tokenizer=FakeTokenizer(), model=FakeModel())
    assert clf.tokenizer.padding_side == "left"
    bodies = ["a" * n for n in (5, 2, 9, 4, 7, 1, 8)]  # deliberately not sorted
    rows = [{"subject": "", "body": b} for b in bodies]
    expected = ["spam" if len(f"Subject: \n\n{b}") % 2 else "ham" for b in bodies]

    results = clf.classify_many(rows)
    assert [r.label for r in results] == expected
    assert clf.usage["calls"] == len(rows)
    assert clf.classify("", bodies[0]).label == expected[0]


def test_unparseable_answer_becomes_an_error_not_a_crash():
    from app.local_llm import LocalLLMClassifier

    class SilentTokenizer(FakeTokenizer):
        def batch_decode(self, tokens, skip_special_tokens):
            return ["I am not sure."] * len(tokens)

    clf = LocalLLMClassifier(device="cpu", tokenizer=SilentTokenizer(), model=FakeModel())
    results = clf.classify_many([{"subject": "s", "body": "b"}])
    assert isinstance(results[0], LLMError)


def test_real_tiny_model_runs_end_to_end():
    """Random-weight 2-layer Qwen2 and a hand-built tokenizer: no download needed.
    Checks the real generate/decode path; the answers are gibberish, so they are
    reported as errors or labels, never as a crash."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from app.local_llm import LocalLLMClassifier

    tok = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                w: i
                for i, w in enumerate(["<pad>", "<unk>", "spam", "ham", "subject", ":", "hello", "{", "}"])
            },
            unk_token="<unk>",
        )
    )
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    hf_tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>")
    hf_tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    config = transformers.Qwen2Config(
        vocab_size=9,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config)

    clf = LocalLLMClassifier(device="cpu", batch_size=2, max_new_tokens=4, tokenizer=hf_tok, model=model)
    results = clf.classify_many([{"subject": "hello", "body": "spam " * n} for n in (1, 5, 3)])
    assert len(results) == 3
    assert all(isinstance(r, LLMError) or r.label in {"spam", "ham"} for r in results)
    assert clf.usage["calls"] == 3 and clf.usage["prompt_tokens"] > 0
