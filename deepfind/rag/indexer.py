"""Qdrant hybrid index: dense + sparse named vectors via fastembed."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache

from .config import Config
from .models import Chunk

LOGGER = logging.getLogger("deepfind.rag.indexer")

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
PAYLOAD_INDEX_FIELDS = ["source", "source_type", "tags", "source_sha256"]


class IndexError_(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _dense_model(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=name)


@lru_cache(maxsize=4)
def _sparse_model(name: str):
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=name)


def _dense_dim(name: str) -> int:
    vec = next(iter(_dense_model(name).embed(["dimension probe"])))
    return len(vec)


class Indexer:
    def __init__(self, config: Config):
        self.config = config
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise IndexError_("qdrant-client is not installed") from exc
        self.client = QdrantClient(url=config.qdrant_url)
        self.collection = config.qdrant_collection

    # ---- collection management -------------------------------------------
    def ensure_collection(self) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            return
        dim = _dense_dim(self.config.dense_model)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        for field in PAYLOAD_INDEX_FIELDS:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        LOGGER.info(
            "Created Qdrant collection '%s' (dense dim=%d)", self.collection, dim
        )

    # ---- embedding -------------------------------------------------------
    def _embed_points(self, chunks: list[Chunk]):
        from qdrant_client import models

        texts = [c.text for c in chunks]
        dense_fn = (
            getattr(_dense_model(self.config.dense_model), "passage_embed", None)
            or _dense_model(self.config.dense_model).embed
        )
        dense_vecs = list(dense_fn(texts))
        sparse_vecs = list(_sparse_model(self.config.sparse_model).embed(texts))
        points = []
        for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
            payload = chunk.model_dump()
            payload["parsed_dir"] = _parsed_dir(chunk.source)
            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector={
                        DENSE_VECTOR: list(map(float, dense)),
                        SPARSE_VECTOR: models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )
        return points

    # ---- write -----------------------------------------------------------
    def replace_source(self, source: str, chunks: list[Chunk]) -> int:
        from qdrant_client import models

        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source", match=models.MatchValue(value=source)
                        )
                    ]
                )
            ),
        )
        if not chunks:
            return 0
        points = self._embed_points(chunks)
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def delete_source(self, source: str) -> None:
        from qdrant_client import models

        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source", match=models.MatchValue(value=source)
                        )
                    ]
                )
            ),
        )

    def existing_sources(self) -> set[str]:
        if not self.client.collection_exists(self.collection):
            return set()
        sources: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=["source"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for p in points:
                src = (p.payload or {}).get("source")
                if src:
                    sources.add(src)
            if offset is None:
                break
        return sources

    def prune(self, keep_sources: Iterable[str]) -> list[str]:
        keep = set(keep_sources)
        removed = []
        for src in self.existing_sources():
            if src not in keep:
                self.delete_source(src)
                removed.append(src)
        return removed


def _parsed_dir(source: str) -> str:
    if "." in source.rsplit("/", 1)[-1]:
        return source.rsplit(".", 1)[0]
    return source
