from deepfind.rag.ingest import media
from deepfind.rag.ingest.media import _seconds_to_hms


def test_seconds_to_hms():
    assert _seconds_to_hms(0) == "00:00:00"
    assert _seconds_to_hms(65) == "00:01:05"
    assert _seconds_to_hms(3661) == "01:01:01"


def test_seconds_to_hms_rounds():
    assert _seconds_to_hms(299.6) == "00:05:00"


def test_shared_model_is_cached(monkeypatch):
    calls = []

    def fake_load_model(model_name):
        calls.append(model_name)
        return ("backend", object(), None, "cuda")

    media._load_shared_model.cache_clear()
    monkeypatch.setattr(media, "load_model", fake_load_model)
    first = media._load_shared_model("model")
    second = media._load_shared_model("model")
    media._load_shared_model.cache_clear()

    assert first is second
    assert calls == ["model"]
