#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.vector.qdrant tests.

These run against Qdrant's in-process local mode (``QdrantClient(":memory:")``),
which implements the same API as the server, so the backend is exercised for
real: collections, aliases, upserts, filters, sparse and dense queries. Only the
embedding call (an external HTTP API) is replaced, with a fixed vector.
"""

import math
import unittest
import uuid
import warnings

from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import QdrantClient

from wiki_rag.vector.bm25 import encode_document
from wiki_rag.vector.qdrant import DENSE_VECTOR, SPARSE_VECTOR, QdrantVector

DIM = 4


def _cfg(url: str = "http://localhost:6333", timeout: float = 30.0, api_key: str | None = None) -> SimpleNamespace:
    """Build the slice of Config the backend reads."""
    return SimpleNamespace(qdrant=SimpleNamespace(url=url, timeout=timeout), qdrant_api_key=api_key)


def _make_vector() -> QdrantVector:
    """Return a QdrantVector wired to a fresh in-memory Qdrant."""
    return QdrantVector(_cfg(), client=QdrantClient(":memory:"))


def _unit(*components: float) -> list[float]:
    """Return the vector normalised to unit length (padded to DIM)."""
    vec = list(components) + [0.0] * (DIM - len(components))
    norm = math.sqrt(sum(c * c for c in vec))
    return [c / norm for c in vec]


def _record(
    id: str,
    text: str,
    vector: list[float],
    *,
    title: str | None = None,
    page_id: int = 1,
    section_id: str | None = None,
    chunk_index: int = 0,
    chunk_separator: str = "",
    **extra,
) -> dict:
    """Build a record shaped like the indexer produces."""
    return {
        "id": id,
        "section_id": section_id if section_id is not None else id,
        "chunk_index": chunk_index,
        "title": title if title is not None else f"Title {id[:8]}",
        "text": text,
        "chunk_separator": chunk_separator,
        "source": f"https://example.com/{id[:8]}",
        "parent": None,
        "children": [],
        "previous": [],
        "next": [],
        "relations": [],
        "categories": ["Cat"],
        "page_id": page_id,
        "doc_id": "doc",
        "doc_title": "Doc",
        "doc_hash": "hash",
        "dense_vector": vector,
        **extra,
    }


def _uid(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, seed))


def setUpModule() -> None:  # noqa: N802
    # Local mode has no payload indexes and says so on every create; that is
    # expected here and would only drown the test output.
    warnings.filterwarnings("ignore", message="Payload indexes have no effect in the local Qdrant")


class TestConstructor(unittest.TestCase):
    def test_reads_connection_settings_from_config(self):
        vector = QdrantVector(_cfg(url="http://q:6333", timeout=12.7, api_key="k"), client=QdrantClient(":memory:"))
        self.assertEqual("http://q:6333", vector.url)
        self.assertEqual(12, vector.timeout)
        self.assertEqual("k", vector.api_key)

    def test_empty_api_key_becomes_none(self):
        vector = QdrantVector(_cfg(api_key=""), client=QdrantClient(":memory:"))
        self.assertIsNone(vector.api_key)

    def test_missing_url_exits(self):
        with self.assertRaises(SystemExit):
            QdrantVector(_cfg(url=""))

    def test_builds_client_from_config(self):
        with patch("wiki_rag.vector.qdrant.QdrantClient") as client_cls:
            QdrantVector(_cfg(url="http://q:6333", timeout=5, api_key="secret"))
        client_cls.assert_called_once_with(url="http://q:6333", api_key="secret", timeout=5)


class TestCollections(unittest.TestCase):
    def setUp(self):
        self.vector = _make_vector()
        self.client = self.vector._client

    def test_create_collection_configures_vectors_and_alias(self):
        self.vector.create_collection("col", DIM)

        self.assertTrue(self.vector.collection_exists("col"))
        aliases = {a.alias_name: a.collection_name for a in self.client.get_aliases().aliases}
        self.assertIn("col", aliases)
        physical = aliases["col"]
        self.assertTrue(physical.startswith("col__"))

        info = self.client.get_collection(physical)
        dense = info.config.params.vectors[DENSE_VECTOR]
        self.assertEqual(DIM, dense.size)
        self.assertEqual("Cosine", dense.distance)
        self.assertIn(SPARSE_VECTOR, info.config.params.sparse_vectors)
        self.assertEqual("idf", info.config.params.sparse_vectors[SPARSE_VECTOR].modifier)

    def test_create_collection_replaces_existing_one(self):
        self.vector.create_collection("col", DIM)
        self.vector.insert_batch("col", [_record(_uid("a"), "some text", _unit(1))])
        self.assertEqual(1, self.client.count("col").count)

        self.vector.create_collection("col", DIM)

        self.assertEqual(0, self.client.count("col").count)
        # The old physical collection is gone, not just orphaned.
        self.assertEqual(1, len(self.client.get_collections().collections))

    def test_collection_exists_false_when_absent(self):
        self.assertFalse(self.vector.collection_exists("nope"))

    def test_drop_collection_removes_alias_and_data(self):
        self.vector.create_collection("col", DIM)
        self.vector.drop_collection("col")
        self.assertFalse(self.vector.collection_exists("col"))
        self.assertEqual([], self.client.get_aliases().aliases)
        self.assertEqual([], self.client.get_collections().collections)

    def test_drop_missing_collection_is_noop(self):
        self.vector.drop_collection("nope")
        self.assertFalse(self.vector.collection_exists("nope"))

    def test_rename_switches_alias_atomically(self):
        self.vector.create_collection("live", DIM)
        self.vector.insert_batch("live", [_record(_uid("old"), "old data", _unit(1))])
        self.vector.create_collection("live_temp", DIM)
        self.vector.insert_batch("live_temp", [
            _record(_uid("new1"), "new data", _unit(1)),
            _record(_uid("new2"), "new data", _unit(0, 1)),
        ])

        # This is exactly the indexer's swap sequence.
        self.vector.flush_collection("live_temp")
        self.vector.compact_collection("live_temp")
        self.vector.drop_collection("live")
        self.vector.rename_collection("live_temp", "live")
        self.vector.load_collection("live")

        self.assertTrue(self.vector.collection_exists("live"))
        self.assertFalse(self.vector.collection_exists("live_temp"))
        self.assertEqual(2, self.client.count("live").count)
        self.assertEqual(1, len(self.client.get_collections().collections))

    def test_reindex_does_not_disturb_live_collection_until_swap(self):
        self.vector.create_collection("live", DIM)
        self.vector.insert_batch("live", [_record(_uid("old"), "old data", _unit(1))])
        self.vector.create_collection("live_temp", DIM)
        self.vector.insert_batch("live_temp", [_record(_uid("new"), "new data", _unit(1))])
        self.vector.drop_collection("live")
        self.vector.rename_collection("live_temp", "live")
        # Second full reindex: creating live_temp again must not touch "live",
        # even though "live" is physically the collection created as live_temp.
        self.vector.create_collection("live_temp", DIM)
        self.assertTrue(self.vector.collection_exists("live"))
        self.assertTrue(self.vector.collection_exists("live_temp"))
        self.assertEqual(1, self.client.count("live").count)
        self.assertEqual(0, self.client.count("live_temp").count)
        self.assertEqual(2, len(self.client.get_collections().collections))

    def test_rename_missing_source_raises(self):
        with self.assertRaises(ValueError):
            self.vector.rename_collection("missing", "live")

    def test_rename_onto_existing_name_raises(self):
        self.vector.create_collection("src", DIM)
        self.client.create_collection("dst", vectors_config={})
        with self.assertRaises(ValueError):
            self.vector.rename_collection("src", "dst")
        self.vector.create_collection("dst_alias", DIM)
        with self.assertRaises(ValueError):
            self.vector.rename_collection("src", "dst_alias")
        # Nothing moved.
        self.assertTrue(self.vector.collection_exists("src"))
        self.assertEqual(3, len(self.client.get_collections().collections))

    def test_physical_collection_name_is_resolved_too(self):
        # A collection created outside this backend, by its plain name.
        self.client.create_collection("plain", vectors_config={})
        self.assertTrue(self.vector.collection_exists("plain"))
        self.vector.rename_collection("plain", "renamed")
        self.assertTrue(self.vector.collection_exists("renamed"))
        self.assertTrue(self.vector.collection_exists("plain"))  # physical still there, now aliased
        self.vector.drop_collection("renamed")
        self.assertFalse(self.vector.collection_exists("plain"))


class TestInsertAndDelete(unittest.TestCase):
    def setUp(self):
        self.vector = _make_vector()
        self.client = self.vector._client
        self.vector.create_collection("col", DIM)

    def test_insert_stores_payload_and_both_vectors(self):
        rid = _uid("a")
        self.vector.insert_batch("col", [_record(rid, "Moodle gradebook settings", _unit(1, 1), page_id=42)])

        [point] = self.client.retrieve("col", ids=[rid], with_payload=True, with_vectors=True)
        self.assertEqual(rid, point.payload["id"])
        self.assertEqual(42, point.payload["page_id"])
        self.assertEqual(["Cat"], point.payload["categories"])
        self.assertNotIn("dense_vector", point.payload)
        self.assertEqual(DIM, len(point.vector[DENSE_VECTOR]))
        expected_sparse = encode_document("Moodle gradebook settings")
        self.assertEqual(expected_sparse.indices, point.vector[SPARSE_VECTOR].indices)

    def test_insert_empty_batch_is_noop(self):
        self.vector.insert_batch("col", [])
        self.assertEqual(0, self.client.count("col").count)

    def test_insert_splits_large_batches(self):
        records = [_record(_uid(f"r{i}"), f"text {i}", _unit(1, i % 3)) for i in range(150)]
        self.vector.insert_batch("col", records)
        self.assertEqual(150, self.client.count("col").count)

    def test_insert_is_an_upsert(self):
        rid = _uid("a")
        self.vector.insert_batch("col", [_record(rid, "first", _unit(1))])
        self.vector.insert_batch("col", [_record(rid, "second", _unit(1))])
        self.assertEqual(1, self.client.count("col").count)
        self.assertEqual({rid: f"Title {rid[:8]}\n\nsecond"}, self.vector.get_documents_contents_by_id("col", [rid]))

    def test_non_uuid_ids_are_mapped_deterministically(self):
        self.vector.insert_batch("col", [_record("legacy-id-1", "text", _unit(1))])
        self.assertEqual(
            {"legacy-id-1": "Title legacy-i\n\ntext"},
            self.vector.get_documents_contents_by_id("col", ["legacy-id-1"]),
        )

    def test_delete_by_page_ids(self):
        self.vector.insert_batch("col", [
            _record(_uid("a"), "a", _unit(1), page_id=1),
            _record(_uid("b"), "b", _unit(1), page_id=2),
            _record(_uid("c"), "c", _unit(1), page_id=2),
            _record(_uid("d"), "d", _unit(1), page_id=3),
        ])
        self.vector.delete_by_page_ids("col", [2, 3])
        remaining = self.client.scroll("col", with_payload=["page_id"], limit=10)[0]
        self.assertEqual([1], [p.payload["page_id"] for p in remaining])

    def test_delete_by_page_ids_empty_is_noop(self):
        self.vector.insert_batch("col", [_record(_uid("a"), "a", _unit(1))])
        self.vector.delete_by_page_ids("col", [])
        self.assertEqual(1, self.client.count("col").count)


class TestGetContents(unittest.TestCase):
    def setUp(self):
        self.vector = _make_vector()
        self.vector.create_collection("col", DIM)

    def test_get_documents_contents_by_id(self):
        a, b = _uid("a"), _uid("b")
        self.vector.insert_batch("col", [
            _record(a, "body a", _unit(1), title="A"),
            _record(b, "body b", _unit(1), title="B"),
        ])
        self.assertEqual(
            {a: "A\n\nbody a", b: "B\n\nbody b"},
            self.vector.get_documents_contents_by_id("col", [a, b, _uid("missing")]),
        )

    def test_get_documents_contents_by_id_empty(self):
        self.assertEqual({}, self.vector.get_documents_contents_by_id("col", []))

    def test_section_reassembly_uses_recorded_separators(self):
        section = _uid("section")
        self.vector.insert_batch("col", [
            # Inserted out of order on purpose: chunk_index must drive the order.
            _record(_uid("c2"), "third", _unit(1), title="Sec", section_id=section, chunk_index=2),
            _record(section, "first", _unit(1), title="Sec", section_id=section, chunk_index=0,
                    chunk_separator=" "),
            _record(_uid("c1"), "second", _unit(1), title="Sec", section_id=section, chunk_index=1,
                    chunk_separator="\n"),
        ])
        self.assertEqual(
            {section: "Sec\n\nfirst second\nthird"},
            self.vector.get_documents_contents_by_section_ids("col", [section]),
        )

    def test_section_reassembly_falls_back_to_paragraph_break_without_separator(self):
        section = _uid("section")
        first = _record(section, "first", _unit(1), title="Sec", section_id=section, chunk_index=0)
        second = _record(_uid("c1"), "second", _unit(1), title="Sec", section_id=section, chunk_index=1)
        del first["chunk_separator"], second["chunk_separator"]
        self.vector.insert_batch("col", [first, second])
        self.assertEqual(
            {section: "Sec\n\nfirst\n\nsecond"},
            self.vector.get_documents_contents_by_section_ids("col", [section]),
        )

    def test_section_lookup_falls_back_to_record_id_for_legacy_records(self):
        legacy = _uid("legacy")
        record = _record(legacy, "legacy body", _unit(1), title="Old")
        for key in ("section_id", "chunk_index", "chunk_separator"):
            del record[key]
        chunked = _uid("new")
        self.vector.insert_batch("col", [record, _record(chunked, "new body", _unit(1), title="New")])

        self.assertEqual(
            {legacy: "Old\n\nlegacy body", chunked: "New\n\nnew body"},
            self.vector.get_documents_contents_by_section_ids("col", [legacy, chunked, _uid("missing")]),
        )

    def test_section_lookup_pages_through_large_results(self):
        section = _uid("big")
        records = [
            _record(section if i == 0 else _uid(f"chunk{i}"), f"p{i}", _unit(1), title="Big",
                    section_id=section, chunk_index=i, chunk_separator=" ")
            for i in range(300)
        ]
        self.vector.insert_batch("col", records)
        [content] = self.vector.get_documents_contents_by_section_ids("col", [section]).values()
        self.assertEqual("Big\n\n" + " ".join(f"p{i}" for i in range(300)), content)

    def test_section_lookup_empty(self):
        self.assertEqual({}, self.vector.get_documents_contents_by_section_ids("col", []))


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        self.vector = _make_vector()
        self.vector.create_collection("col", DIM)
        self.dense_hit = _uid("dense")
        self.sparse_hit = _uid("sparse")
        self.both_hit = _uid("both")
        self.noise = _uid("noise")
        self.vector.insert_batch("col", [
            # Near the query vector but lexically unrelated.
            _record(self.dense_hit, "unrelated words entirely", _unit(1, 0.05), title="Dense"),
            # Orthogonal to the query vector but a strong keyword match.
            _record(self.sparse_hit, "gradebook gradebook settings", _unit(0, 0, 1), title="Sparse"),
            # Close vector and keyword match: should win.
            _record(self.both_hit, "the gradebook settings page", _unit(1, 0.1), title="Both"),
            # Nothing in common with the query.
            _record(self.noise, "completely different topic", _unit(0, 0, 0, 1), title="Noise",
                    section_id="s", chunk_index=0),
        ])

    def _retrieve(self, queries: list[str], sparse_query: str | None = None) -> list[dict]:
        with patch.object(QdrantVector, "_embed_and_average_queries", return_value=_unit(1)) as embed:
            results = self.vector.retrieve(
                collection_name="col",
                embedding_model="emb",
                embedding_dimensions=DIM,
                queries=queries,
                sparse_query=sparse_query,
                embedding_api_base="https://emb.example.com/v1",
                embedding_api_key="key",
            )
        embed.assert_called_once_with(
            embedding_model="emb", embedding_dimensions=DIM, queries=queries,
            api_base="https://emb.example.com/v1", api_key="key",
        )
        return results

    def test_hybrid_ranking_combines_dense_and_sparse(self):
        results = self._retrieve(["gradebook settings"])
        ids = [r["id"] for r in results]
        self.assertEqual(self.both_hit, ids[0])
        self.assertIn(self.dense_hit, ids)
        self.assertIn(self.sparse_hit, ids)
        # Scores are fused weights in [0, 1], ordered.
        scores = [r["distance"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        # Fusion order: semantic+keyword > strong semantic > keyword-only > noise.
        self.assertEqual([self.both_hit, self.dense_hit, self.sparse_hit, self.noise], ids)
        # The keyword-only hit (cosine 0) is carried entirely by the sparse channel.
        self.assertGreater(scores[2], 0.0)
        self.assertLessEqual(scores[2], QdrantVector.RERANK_WEIGHTS[1])
        self.assertEqual(0.0, scores[3])

    def test_result_shape_matches_pipeline_contract(self):
        [top, *_] = self._retrieve(["gradebook settings"])
        self.assertEqual({"id", "distance", "entity"}, set(top))
        entity = top["entity"]
        self.assertEqual(self.both_hit, entity["id"])
        self.assertEqual("Both", entity["title"])
        self.assertEqual("the gradebook settings page", entity["text"])
        self.assertEqual([], entity["children"])
        self.assertEqual(1, entity["page_id"])
        self.assertEqual(self.both_hit, entity["section_id"])
        self.assertEqual("", entity["chunk_separator"])
        # Only declared output fields are exposed.
        self.assertTrue(set(entity) <= set(QdrantVector.OUTPUT_FIELDS))

    def test_sparse_query_overrides_queries_for_keyword_channel(self):
        # HyDE path: queries carry the hypothetical passages (embedded), while the
        # keyword channel must use the original question.
        def score_of(results: list[dict], id: str) -> float:
            return next(r["distance"] for r in results if r["id"] == id)

        with_override = self._retrieve(["a hypothetical passage"], sparse_query="gradebook")
        without = self._retrieve(["a hypothetical passage"])
        # The keyword doc only gets sparse credit when the keyword channel sees "gradebook".
        self.assertGreater(score_of(with_override, self.sparse_hit), score_of(without, self.sparse_hit))
        # And the dense-only docs are unaffected by what the keyword channel searched.
        self.assertAlmostEqual(score_of(with_override, self.dense_hit), score_of(without, self.dense_hit))

    def test_dense_only_ranking_when_no_keyword_matches(self):
        results = self._retrieve(["zzzz qqqq"])
        ids = [r["id"] for r in results]
        # Pure cosine order: the closest vector first, orthogonal ones last.
        self.assertEqual([self.dense_hit, self.both_hit], ids[:2])
        self.assertEqual({self.sparse_hit, self.noise}, set(ids[2:]))
        # Without a keyword hit, nothing can exceed the dense weight.
        self.assertTrue(all(r["distance"] <= QdrantVector.RERANK_WEIGHTS[0] for r in results))

    def test_missing_output_fields_are_omitted_not_faked(self):
        legacy = _uid("legacy")
        record = _record(legacy, "gradebook", _unit(1))
        for key in ("section_id", "chunk_index", "chunk_separator"):
            del record[key]
        self.vector.insert_batch("col", [record])
        [hit] = [r for r in self._retrieve(["gradebook"]) if r["id"] == legacy]
        self.assertNotIn("section_id", hit["entity"])
        self.assertIn("title", hit["entity"])

    def test_limit_is_applied(self):
        self.vector.insert_batch("col", [
            _record(_uid(f"more{i}"), "gradebook", _unit(1, i / 100)) for i in range(60)
        ])
        results = self._retrieve(["gradebook"])
        self.assertEqual(QdrantVector.HYBRID_RERANK_LIMIT, len(results))


class TestLoadVectorStore(unittest.TestCase):
    def test_qdrant_backend_is_loadable_by_name(self):
        import wiki_rag.config as config_module

        from wiki_rag.vector import load_vector_store

        with patch.object(config_module, "cfg", _cfg(url="http://q:6333")), \
                patch("wiki_rag.vector.qdrant.QdrantClient") as client_cls:
            store = load_vector_store("qdrant")
        self.assertIsInstance(store, QdrantVector)
        client_cls.assert_called_once_with(url="http://q:6333", api_key=None, timeout=30)
