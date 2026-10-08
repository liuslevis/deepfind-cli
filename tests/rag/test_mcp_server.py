import threading

from deepfind.rag.config import Config
from deepfind.rag.mcp_server import _auto_ingest_loop
from deepfind.rag.sync_service import SyncResult


def test_auto_ingest_runs_immediately_and_stops(monkeypatch):
    stop = threading.Event()
    calls = []

    def fake_sync(config):
        calls.append(config)
        stop.set()
        return SyncResult(results=[], pruned=[])

    monkeypatch.setattr("deepfind.rag.mcp_server.sync_documents", fake_sync)

    config = Config()
    _auto_ingest_loop(stop, interval=60, config=config)

    assert calls == [config]


def test_auto_ingest_keeps_running_after_scan_error(monkeypatch):
    stop = threading.Event()
    calls = 0

    def fake_sync(config):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        stop.set()
        return SyncResult(results=[], pruned=[])

    monkeypatch.setattr("deepfind.rag.mcp_server.sync_documents", fake_sync)
    monkeypatch.setattr(stop, "wait", lambda interval: False)

    _auto_ingest_loop(stop, interval=60, config=Config())

    assert calls == 2
