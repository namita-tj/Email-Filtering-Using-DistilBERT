import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.pipeline import SpamFilter


@pytest.fixture
def client_for(fake_model, fake_llm):
    def make(probability=0.95, llm=None):
        spam_filter = SpamFilter(fake_model(probability), llm or fake_llm())
        return TestClient(create_app(spam_filter), raise_server_exceptions=False)

    return make


def test_health(client_for):
    response = client_for().get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "llm_fallback": True,
        "thresholds": {"low": 0.3, "high": 0.8},
    }


def test_health_while_loading():
    app = create_app()
    app.state.spam_filter = None
    response = TestClient(app).get("/health")
    assert response.status_code == 503


def test_predict_model_decision(client_for):
    response = client_for(0.97).post("/predict", json={"subject": "Win", "text": "Claim now"})
    assert response.status_code == 200
    assert response.json() == {
        "label": "spam",
        "source": "model",
        "spam_probability": 0.97,
        "reason": None,
        "explanation": None,
    }


def test_predict_llm_decision(client_for, fake_llm):
    client = client_for(0.55, llm=fake_llm(label="ham", reason="Normal invoice"))
    body = client.post("/predict", json={"text": "Your invoice is attached"}).json()
    assert (body["label"], body["source"], body["reason"]) == ("ham", "llm", "Normal invoice")


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": ""}, {"text": "   "}, {"text": "x" * 20_001}, {"text": "ok", "subject": "s" * 1001}],
)
def test_invalid_input_rejected(client_for, payload):
    assert client_for().post("/predict", json=payload).status_code == 422


def test_internal_errors_are_not_leaked(client_for):
    class BrokenModel:
        def spam_probability(self, text):
            raise RuntimeError("secret path /models/tf_model.h5")

    client = TestClient(create_app(SpamFilter(BrokenModel())), raise_server_exceptions=False)
    response = client.post("/predict", json={"text": "hello"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_explain_flag_adds_word_weights(fake_explainable_model):
    client = TestClient(create_app(SpamFilter(fake_explainable_model(0.9))))
    response = client.post(
        "/predict?explain=true&top_k=2",
        json={"subject": "Verify", "text": "your account now please"},
    )
    assert response.status_code == 200
    explanation = response.json()["explanation"]
    assert explanation["method"] == "integrated_gradients"
    assert explanation["words"] == [
        {"word": "Verify", "weight": 0.4},
        {"word": "please", "weight": -0.2},
    ]


def test_explanation_is_off_by_default(fake_explainable_model):
    client = TestClient(create_app(SpamFilter(fake_explainable_model(0.9))))
    assert client.post("/predict", json={"text": "hello"}).json()["explanation"] is None


def test_explain_unavailable_returns_501(client_for):
    response = client_for().post("/predict?explain=true", json={"text": "hello"})
    assert response.status_code == 501


def test_top_k_is_bounded(fake_explainable_model):
    client = TestClient(create_app(SpamFilter(fake_explainable_model(0.9))))
    assert client.post("/predict?explain=true&top_k=0", json={"text": "hi"}).status_code == 422
