import pytest

from deepfind.rag.search import SearchError, SearchService


@pytest.fixture(scope="module")
def service():
    return SearchService()


def test_empty_query_rejected(service):
    with pytest.raises(SearchError):
        service.search("   ")


def test_invalid_limit_rejected(service):
    with pytest.raises(SearchError):
        service.search("hello", limit=99)


def test_bad_mode_rejected(service):
    with pytest.raises(SearchError):
        service.search("hello", mode="fuzzy")


def test_path_traversal_rejected(service):
    with pytest.raises(SearchError):
        service.search("hello", path_prefix="../etc")


def test_absolute_path_rejected(service):
    with pytest.raises(SearchError):
        service.search("hello", path_prefix="/pdf/x")


def test_path_prefix_must_be_under_rag_source_directories(service):
    with pytest.raises(SearchError):
        service.search("hello", path_prefix="other/dir")
