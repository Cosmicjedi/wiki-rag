#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Qdrant-specific implementation of the generic vector interface.

Every record is stored as one Qdrant point: the dense embedding and a BM25
sparse vector as named vectors, and all scalar fields as the payload. Hybrid
retrieval runs a dense (cosine, HNSW) search and a sparse (BM25) search in one
batched request and fuses the two result lists client side with weighted
scoring, so the ranking semantics match what the Milvus backend produced.

Collections are addressed through aliases. Qdrant cannot rename a collection,
but it can atomically re-point an alias, which is exactly what the indexer's
"build ``<name>_temp``, then swap it in" flow needs: every collection created
here gets a unique physical name, and the name the application uses is an
alias to it. Every other operation resolves the alias transparently, so the
rest of the application never sees physical names.
"""

import logging
import secrets
import sys
import uuid

from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.conversions import common_types as types

from wiki_rag.config import Config
from wiki_rag.vector import BaseVector
from wiki_rag.vector.bm25 import encode_document, encode_query, normalise_score

logger = logging.getLogger(__name__)

#: Name of the dense (embedding) named vector on every point.
DENSE_VECTOR = "dense_vector"
#: Name of the BM25 sparse named vector on every point.
SPARSE_VECTOR = "sparse_vector"


class QdrantVector(BaseVector):
    """Qdrant backend vector.

    Connection settings are provided via the :class:`~wiki_rag.config.Config`
    singleton: ``cfg.qdrant.url`` (e.g. ``'http://localhost:6333'``),
    ``cfg.qdrant.timeout`` and, optionally, ``cfg.qdrant_api_key``
    (``QDRANT_API_KEY``, env only).

    A pre-built client can be injected with ``client=`` (used by the tests with
    Qdrant's in-process local mode); otherwise one is created from the config.
    """

    #: Scalar/metadata fields returned as search output, in this order. Any
    #: payload key not listed here is ignored at query time, and a listed key
    #: missing from a record's payload (a collection indexed before the field
    #: existed) is simply omitted, so older collections keep working.
    OUTPUT_FIELDS: tuple[str, ...] = (
        "id",
        "title",
        "text",
        "source",
        "doc_id",
        "doc_title",
        "doc_hash",
        "parent",
        "children",
        "previous",
        "next",
        "relations",
        "categories",
        "page_id",
        "section_id",
        "chunk_index",
        "chunk_separator",
    )

    #: Payload fields that get an index: the ones used in filters.
    PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
        ("page_id", models.PayloadSchemaType.INTEGER),
        ("section_id", models.PayloadSchemaType.KEYWORD),
    )

    #: Points sent per upsert request. Each point carries a dense vector of
    #: up to a few thousand floats, so keep requests comfortably sized.
    UPSERT_BATCH_SIZE: int = 64

    # Retrieval tuning (mirrors the values the Milvus backend used).
    DENSE_SEARCH_LIMIT: int = 20
    SPARSE_SEARCH_LIMIT: int = 20
    HYBRID_RERANK_LIMIT: int = 30
    #: (dense, sparse) weights for the weighted fusion of the two channels.
    RERANK_WEIGHTS: tuple[float, float] = (0.7, 0.3)
    #: HNSW beam width at search time. Must be >= the search limit.
    HNSW_EF_SEARCH: int = 64

    def __init__(self, cfg: Config, client: QdrantClient | None = None) -> None:
        """Initialise the Qdrant backend.

        Args:
            cfg: Resolved application configuration.
            client: Optional pre-built client (tests). When omitted, a client is
                created from ``cfg.qdrant`` / ``cfg.qdrant_api_key``.

        """
        self.url: str = cfg.qdrant.url
        self.api_key: str | None = cfg.qdrant_api_key or None
        self.timeout: int = max(1, int(cfg.qdrant.timeout))
        if client is None:
            if not self.url:
                logger.error("Qdrant URL not found in configuration. Exiting.")
                sys.exit(1)
            client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)
        self._client: QdrantClient = client

    # BaseVector interface.

    def create_collection(self, collection_name: str, embedding_dimension: int) -> None:
        """Create (or recreate) a Qdrant collection with the required vectors.

        A pre-existing collection reachable as ``collection_name`` (alias or
        physical) is dropped first. The physical collection gets a unique name
        and ``collection_name`` becomes an alias to it, so that a later
        :meth:`rename_collection` is an atomic alias switch.

        The dense vector uses cosine distance on an HNSW index; the sparse
        vector carries BM25 term frequencies and is scored with Qdrant's
        server-side IDF modifier. Payload indexes are created for the fields
        used in filters.

        Args:
            collection_name: Name of the target collection.
            embedding_dimension: Dimensionality of the dense vector.

        """
        if self.collection_exists(collection_name):
            logger.debug("Dropping existing collection %r before recreation", collection_name)
            self.drop_collection(collection_name)

        physical = self._new_physical_name(collection_name)
        logger.debug("Creating collection %r as %r", collection_name, physical)
        self._client.create_collection(
            collection_name=physical,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=embedding_dimension,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=64, ef_construct=100),
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        for field, schema in self.PAYLOAD_INDEXES:
            self._client.create_payload_index(physical, field_name=field, field_schema=schema, wait=True)

        self._client.update_collection_aliases(change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=physical, alias_name=collection_name),
            ),
        ])
        logger.debug("Collection %r created; alias %r -> %r", collection_name, collection_name, physical)

    def collection_exists(self, name: str) -> bool:
        """Return True if ``name`` is an alias to, or the physical name of, a collection."""
        return self._resolve(name) is not None

    def drop_collection(self, name: str) -> None:
        """Delete the collection behind ``name`` and the alias itself (when it is one)."""
        physical = self._resolve(name)
        if physical is None:
            return
        if physical != name:
            self._client.update_collection_aliases(change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=name)),
            ])
        self._client.delete_collection(physical)

    def rename_collection(self, old: str, new: str) -> None:
        """Rename a collection by atomically re-pointing aliases.

        After the call ``new`` resolves to the collection ``old`` resolved to,
        and ``old`` no longer resolves. The target name must be free (the
        indexer drops the live collection before renaming the temporary one
        onto it), so nothing is ever orphaned behind a re-pointed alias.

        Args:
            old: Current name (alias or physical).
            new: Desired name.

        Raises:
            ValueError: When ``old`` does not exist or ``new`` already does.

        """
        physical = self._resolve(old)
        if physical is None:
            msg = f"Collection {old!r} does not exist."
            raise ValueError(msg)
        if self.collection_exists(new):
            msg = f"Cannot rename to {new!r}: a collection with that name already exists."
            raise ValueError(msg)

        operations: list[models.AliasOperations] = []
        if physical != old:
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=old)),
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=physical, alias_name=new),
            ),
        )
        self._client.update_collection_aliases(change_aliases_operations=operations)
        logger.debug("Renamed %r -> %r (physical %r)", old, new, physical)

    def flush_collection(self, name: str) -> None:
        """No-op: every upsert here already waits for the write to be applied."""

    def compact_collection(self, name: str) -> None:
        """Qdrant compacts and builds indexes in the background; nothing to trigger."""

    def load_collection(self, name: str) -> None:
        """No-op: Qdrant collections are always ready to serve queries."""

    def delete_by_page_ids(self, collection_name: str, page_ids: list[int]) -> None:
        """Delete all points whose ``page_id`` is in ``page_ids``. No-op when empty."""
        if not page_ids:
            return
        self._client.delete(
            collection_name,
            points_selector=models.Filter(must=[
                models.FieldCondition(key="page_id", match=models.MatchAny(any=[int(p) for p in page_ids])),
            ]),
            wait=True,
        )

    def insert_batch(self, collection_name: str, records: list[dict[str, Any]]) -> None:
        """Upsert a batch of records as points.

        Each record must carry ``id``, ``text`` and ``dense_vector``. The sparse
        BM25 vector is computed here from ``text``; every other key is stored
        verbatim in the payload (Qdrant payloads are schemaless, so nothing is
        ever dropped).

        Args:
            collection_name: Target collection.
            records: Records produced by the indexer.

        """
        if not records:
            return
        points = [
            models.PointStruct(
                id=self._point_id(str(record["id"])),
                vector={
                    DENSE_VECTOR: list(record["dense_vector"]),
                    SPARSE_VECTOR: encode_document(record.get("text") or ""),
                },
                payload={key: value for key, value in record.items() if key != "dense_vector"},
            )
            for record in records
        ]
        for start in range(0, len(points), self.UPSERT_BATCH_SIZE):
            self._client.upsert(collection_name, points=points[start:start + self.UPSERT_BATCH_SIZE], wait=True)

    def get_documents_contents_by_id(
        self,
        collection_name: str,
        ids: list[str],
    ) -> dict[str, str]:
        """Retrieve the title and text (as ``title + blank line + text``) of the given record ids.

        Args:
            collection_name: Target collection.
            ids: Record ids to retrieve.

        Returns:
            Dictionary of record ids as keys and contents as values. Ids that
            do not exist are absent from the result.

        """
        if not ids:
            return {}
        points = self._client.retrieve(
            collection_name,
            ids=[self._point_id(i) for i in ids],
            with_payload=["id", "title", "text"],
            with_vectors=False,
        )
        return {
            str(point.payload["id"]): f"{point.payload['title']}\n\n{point.payload['text']}"
            for point in points
            if point.payload
        }

    def get_documents_contents_by_section_ids(
        self,
        collection_name: str,
        section_ids: list[str],
    ) -> dict[str, str]:
        r"""Retrieve full section contents, reassembling chunks in order.

        All points sharing each ``section_id`` are fetched, sorted by
        ``chunk_index`` and joined using the ``chunk_separator`` recorded on
        the preceding chunk (falling back to ``"\n\n"`` for records that
        predate that field). The section title is emitted once.

        Section ids with no ``section_id`` match are looked up by record id
        instead, which covers collections indexed before chunking existed
        (every record is its own single-chunk section there).

        Args:
            collection_name: Target collection.
            section_ids: Section ids to retrieve.

        Returns:
            Dictionary of section ids as keys and reassembled contents as values.

        """
        if not section_ids:
            return {}

        rows = self._scroll_all(
            collection_name,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="section_id", match=models.MatchAny(any=list(section_ids))),
            ]),
            with_payload=["section_id", "chunk_index", "title", "text", "chunk_separator"],
        )

        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row["section_id"]), []).append(row)

        result: dict[str, str] = {}
        for section_id, chunks in grouped.items():
            sorted_chunks = sorted(chunks, key=lambda c: int(c.get("chunk_index") or 0))
            title = sorted_chunks[0]["title"]
            body_parts = [sorted_chunks[0]["text"]]
            for prev_chunk, next_chunk in zip(sorted_chunks, sorted_chunks[1:], strict=False):
                sep = prev_chunk.get("chunk_separator")
                body_parts.append(sep if sep is not None else "\n\n")
                body_parts.append(next_chunk["text"])
            result[section_id] = title + "\n\n" + "".join(body_parts)

        # Legacy records (no section_id payload) are keyed by their own id.
        missing = [sid for sid in section_ids if sid not in result]
        if missing:
            result.update(self.get_documents_contents_by_id(collection_name, missing))
        return result

    def retrieve(self,
            collection_name: str,
            embedding_model: str,
            embedding_dimensions: int,
            queries: list[str],
            sparse_query: str | None = None,
            embedding_api_base: str = "",
            embedding_api_key: str = "",
    ) -> list[dict]:
        """Retrieve the best matches for a question with hybrid search.

        The dense channel searches the averaged query embedding with cosine
        similarity; the sparse channel searches the BM25 encoding of
        ``sparse_query`` (defaulting to ``queries[0]``). Both run in a single
        batched request. The two result lists are then fused with weighted
        scoring: cosine similarities are clipped to ``[0, 1]`` (embeddings of
        unrelated text sit near 0, never meaningfully below it), BM25 scores
        are mapped to ``[0, 1)`` as ``2/pi * atan(score)``, and the weighted
        sum (see :attr:`RERANK_WEIGHTS`) ranks the union of both lists, of
        which the top :attr:`HYBRID_RERANK_LIMIT` are returned. A record
        present in only one list gets credit from that channel alone, so a
        strong keyword hit outranks dense noise, and a strong semantic hit
        outranks a keyword-only one.

        Each result is ``{"id": ..., "distance": <fused score>, "entity": {...}}``
        with ``entity`` holding the :attr:`OUTPUT_FIELDS` present on the point,
        the same shape the rest of the pipeline has always consumed.
        """
        embeddings = self._embed_and_average_queries(
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            queries=queries,
            api_base=embedding_api_base,
            api_key=embedding_api_key,
        )
        bm25_text = sparse_query if sparse_query is not None else queries[0]

        dense_response, sparse_response = self._client.query_batch_points(
            collection_name,
            requests=[
                models.QueryRequest(
                    query=embeddings,
                    using=DENSE_VECTOR,
                    limit=self.DENSE_SEARCH_LIMIT,
                    params=models.SearchParams(hnsw_ef=max(self.HNSW_EF_SEARCH, self.DENSE_SEARCH_LIMIT)),
                    with_payload=True,
                ),
                models.QueryRequest(
                    query=encode_query(bm25_text),
                    using=SPARSE_VECTOR,
                    limit=self.SPARSE_SEARCH_LIMIT,
                    with_payload=True,
                ),
            ],
        )

        dense_weight, sparse_weight = self.RERANK_WEIGHTS
        fused: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        for point in dense_response.points:
            key = str(point.id)
            payloads[key] = point.payload or {}
            fused[key] = fused.get(key, 0.0) + dense_weight * max(point.score, 0.0)
        for point in sparse_response.points:
            key = str(point.id)
            payloads.setdefault(key, point.payload or {})
            fused[key] = fused.get(key, 0.0) + sparse_weight * normalise_score(point.score)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[: self.HYBRID_RERANK_LIMIT]
        results = []
        for key, score in ranked:
            payload = payloads[key]
            entity = {field: payload[field] for field in self.OUTPUT_FIELDS if field in payload}
            results.append({"id": entity.get("id", key), "distance": score, "entity": entity})
        return results

    # Internal helpers.

    @staticmethod
    def _point_id(record_id: str) -> str:
        """Return the Qdrant point id for a record id.

        Qdrant point ids must be UUIDs or unsigned integers. Record ids are
        UUIDs already (sections and chunks are ``uuid5`` derived); anything
        else is mapped deterministically onto a UUID so that the same record
        id always addresses the same point. The original id is kept in the
        payload either way.
        """
        try:
            return str(uuid.UUID(record_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_OID, record_id))

    @staticmethod
    def _new_physical_name(name: str) -> str:
        """Return a fresh, unique physical collection name for alias ``name``."""
        return f"{name}__{secrets.token_hex(4)}"

    def _alias_target(self, name: str) -> str | None:
        """Return the physical collection an alias points to, or None when ``name`` is not an alias."""
        for alias in self._client.get_aliases().aliases:
            if alias.alias_name == name:
                return alias.collection_name
        return None

    def _physical_exists(self, name: str) -> bool:
        """Return True when a collection (not an alias) named ``name`` exists.

        ``QdrantClient.collection_exists`` may answer for aliases as well, so
        the collection list is consulted instead.
        """
        return any(c.name == name for c in self._client.get_collections().collections)

    def _resolve(self, name: str) -> str | None:
        """Resolve ``name`` (alias or physical) to a physical collection name, or None when absent."""
        target = self._alias_target(name)
        if target is not None:
            return target
        if self._physical_exists(name):
            return name
        return None

    def _scroll_all(self,
        collection_name: str,
        scroll_filter: models.Filter,
        with_payload: list[str],
        page_size: int = 256,
    ) -> list[dict]:
        """Return the payloads of every point matching ``scroll_filter``."""
        rows: list[dict] = []
        offset: types.PointId | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name,
                scroll_filter=scroll_filter,
                limit=page_size,
                offset=offset,
                with_payload=with_payload,
                with_vectors=False,
            )
            rows.extend(point.payload for point in points if point.payload)
            if offset is None:
                break
        return rows
