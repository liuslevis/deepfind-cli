from deepfind import asr
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
        return "backend", object(), None, "cuda"

    monkeypatch.setattr(asr, "_ASR_SLOTS", [None, None])
    monkeypatch.setattr(media, "_asr_slot_model", asr._asr_slot_model)
    monkeypatch.setattr(asr, "load_model", fake_load_model)

    first = media._asr_slot_model(media.slot_index(), "model")
    second = media._asr_slot_model(media.slot_index(), "model")

    assert first[1] is second[1]
    assert calls == ["model"]
