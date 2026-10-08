from deepfind.rag.config import Config
from deepfind.rag.pipeline import Result
from deepfind.rag.sync_service import SyncResult, sync_documents


def test_sync_result_counts_statuses():
    result = SyncResult(
        results=[
            Result("a.pdf", "indexed"),
            Result("b.pdf", "skipped"),
            Result("c.pdf", "failed"),
        ],
        pruned=[],
    )

    assert result.processed == 1
    assert result.skipped == 1
    assert result.failed == 1


def test_sync_without_sources_does_not_connect_to_qdrant(monkeypatch):
    monkeypatch.setattr("deepfind.rag.sync_service.discover", lambda **kwargs: [])

    def fail_indexer(config):
        raise AssertionError("Indexer should not be created")

    monkeypatch.setattr("deepfind.rag.sync_service.Indexer", fail_indexer)

    result = sync_documents(Config())

    assert result == SyncResult(results=[], pruned=[])
