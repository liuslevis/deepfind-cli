"""Hybrid search over Qdrant using the native Query API RRF fusion."""

from __future__ import annotations

from .config import Config, get_config
from .indexer import DENSE_VECTOR, SPARSE_VECTOR, Indexer, _dense_model, _sparse_model
from .models import SearchResponse, SearchResult

DEFAULT_LIMIT = 8
CANDIDATE_LIMIT = 20
MERGE_MAX_CHARS = 2400


class SearchError(RuntimeError):
    pass


def _dense_query(name: str, query: str):
    model = _dense_model(name)
    fn = getattr(model, "query_embed", None) or model.embed
    return list(map(float, next(iter(fn([query])))))


def _sparse_query(name: str, query: str):
    from qdrant_client import models

    model = _sparse_model(name)
    fn = getattr(model, "query_embed", None) or model.embed
    sv = next(iter(fn([query])))
    return models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())


def _build_filter(
    source_type: str | None,
    tags: list[str] | None,
):
    from qdrant_client import models

    must = []
    if source_type:
        must.append(
            models.FieldCondition(
                key="source_type", match=models.MatchValue(value=source_type)
            )
        )
    if tags:
        must.append(
            models.FieldCondition(key="tags", match=models.MatchAny(any=list(tags)))
        )
    return models.Filter(must=must) if must else None


class SearchService:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.indexer = Indexer(self.config)
        self.client = self.indexer.client
        self.collection = self.config.qdrant_collection

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LIMIT,
        mode: str = "hybrid",
        source_type: str | None = None,
        path_prefix: str | None = None,
        tags: list[str] | None = None,
        merge_adjacent: bool = False,
    ) -> SearchResponse:
        query = query.strip()
        if not query:
            raise SearchError("query must not be empty")
        if not (1 <= limit <= 20):
            raise SearchError("limit must be between 1 and 20")
        if mode not in ("hybrid", "dense", "sparse"):
            raise SearchError(f"unsupported mode: {mode}")
        if path_prefix is not None:
            if path_prefix.startswith("/") or ".." in path_prefix.split("/"):
                raise SearchError("path_prefix must be a relative path without '..'")
            if not path_prefix.startswith(("pdf/", "media/")):
                raise SearchError("path_prefix must be under pdf/ or media/")

        if not self.client.collection_exists(self.collection):
            raise SearchError(f"Qdrant collection '{self.collection}' does not exist")

        from qdrant_client import models

        flt = _build_filter(source_type, tags)
        # over-fetch when we need client-side path filtering / merging
        fetch = (
            limit
            if not (path_prefix or merge_adjacent)
            else max(limit * 3, CANDIDATE_LIMIT)
        )

        try:
            if mode == "dense":
                resp = self.client.query_points(
                    self.collection,
                    query=_dense_query(self.config.dense_model, query),
                    using=DENSE_VECTOR,
                    limit=fetch,
                    query_filter=flt,
                    with_payload=True,
                )
            elif mode == "sparse":
                resp = self.client.query_points(
                    self.collection,
                    query=_sparse_query(self.config.sparse_model, query),
                    using=SPARSE_VECTOR,
                    limit=fetch,
                    query_filter=flt,
                    with_payload=True,
                )
            else:
                resp = self.client.query_points(
                    self.collection,
                    prefetch=[
                        models.Prefetch(
                            query=_dense_query(self.config.dense_model, query),
                            using=DENSE_VECTOR,
                            limit=CANDIDATE_LIMIT,
                            filter=flt,
                        ),
                        models.Prefetch(
                            query=_sparse_query(self.config.sparse_model, query),
                            using=SPARSE_VECTOR,
                            limit=CANDIDATE_LIMIT,
                            filter=flt,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=fetch,
                    query_filter=flt,
                    with_payload=True,
                )
        except Exception as exc:
            raise SearchError(f"Qdrant query failed: {exc}") from exc

        points = list(resp.points)
        if path_prefix:
            points = [
                p
                for p in points
                if str((p.payload or {}).get("source", "")).startswith(path_prefix)
            ]

        if merge_adjacent:
            points = _merge_adjacent(points)

        results = [_to_result(p) for p in points[:limit]]
        return SearchResponse(query=query, mode=mode, results=results)


def _to_result(point) -> SearchResult:
    payload = point.payload or {}
    return SearchResult(
        text=payload.get("text", ""),
        source=payload.get("source", ""),
        title=payload.get("title", ""),
        section=payload.get("section", ""),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        start_seconds=payload.get("start_seconds"),
        end_seconds=payload.get("end_seconds"),
        score=float(getattr(point, "score", 0.0) or 0.0),
    )


def _merge_adjacent(points: list) -> list:
    """Merge same-source consecutive chunks, keeping best score and covered range."""
    by_id = {}
    for p in points:
        pl = p.payload or {}
        by_id.setdefault(pl.get("source"), []).append(p)

    merged_points = []
    used = set()
    for p in points:
        if id(p) in used:
            continue
        pl = p.payload or {}
        source = pl.get("source")
        idx = pl.get("chunk_index")
        group = [p]
        used.add(id(p))
        if idx is not None:
            for q in points:
                if id(q) in used:
                    continue
                ql = q.payload or {}
                if (
                    ql.get("source") == source
                    and abs((ql.get("chunk_index") or -99) - idx) == 1
                    and len(pl.get("text", "")) + len(ql.get("text", ""))
                    <= MERGE_MAX_CHARS
                ):
                    group.append(q)
                    used.add(id(q))
        merged_points.append(_combine(group))
    merged_points.sort(key=lambda p: -(getattr(p, "score", 0.0) or 0.0))
    return merged_points


class _MergedPoint:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


def _combine(group: list):
    group_sorted = sorted(
        group, key=lambda p: (p.payload or {}).get("chunk_index") or 0
    )
    base = dict(group_sorted[0].payload or {})
    base["text"] = "\n\n".join((p.payload or {}).get("text", "") for p in group_sorted)

    def rng(key, use_min):
        vals = [(p.payload or {}).get(key) for p in group_sorted]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return min(vals) if use_min else max(vals)

    base["page_start"] = rng("page_start", True)
    base["page_end"] = rng("page_end", False)
    base["start_seconds"] = rng("start_seconds", True)
    base["end_seconds"] = rng("end_seconds", False)
    score = max((getattr(p, "score", 0.0) or 0.0) for p in group_sorted)
    return _MergedPoint(base, score)
