#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.index.util tests."""

import unittest
import uuid

from unittest.mock import MagicMock, patch

from wiki_rag.index.util import (
    chunking_signature,
    index_pages,
    index_pages_incremental,
    marker_content,
    marker_matches,
    marker_signature,
    replace_previous_collection,
)


def _make_section(page_id: int, idx: int = 0) -> dict:
    """Return a minimal section dict for testing."""
    return {
        "id": f"sec-{page_id}-{idx}",
        "title": f"Section {idx}",
        "doc_title": f"Page {page_id}",
        "text": "Some text content",
        "source": f"https://example.com/page_{page_id}",
        "parent": None,
        "children": [],
        "previous": [],
        "next": [],
        "relations": [],
        "page_id": page_id,
        "doc_id": f"doc-{page_id}",
        "doc_hash": f"hash-{page_id}",
    }


def _make_page(page_id: int, change_type: str | None = None, num_sections: int = 1) -> dict:
    """Return a minimal page dict with the given change_type."""
    page: dict = {
        "id": page_id,
        "title": f"Page {page_id}",
        "sections": [_make_section(page_id, i) for i in range(num_sections)],
        "categories": [],
        "templates": [],
        "internal_links": [],
        "external_links": [],
        "language_links": [],
    }
    if change_type is not None:
        page["change_type"] = change_type
    return page


def _embed_side_effect(dim: int = 4):
    """Return an embed_documents side effect yielding one vector per input text.

    The batched embedder calls embed_documents() with a list of N texts and
    expects N vectors back, so the mock must size its output to the input.
    """
    return lambda texts: [[0.1] * dim for _ in texts]


def _inserted_records(mock_vector) -> list[dict]:
    """Flatten every record passed across all insert_batch() calls."""
    return [record for call in mock_vector.store.insert_batch.call_args_list for record in call.args[1]]


class TestIndexPagesSkipsDeletedPages(unittest.TestCase):
    """index_pages() must skip pages whose change_type is 'deleted'."""

    @patch("wiki_rag.index.util.vector")
    @patch("wiki_rag.index.util.OpenAIEmbeddings")
    def test_index_pages_skips_deleted_pages(self, mock_embeddings_cls, mock_vector):
        """Deleted pages produce zero insert_batch calls."""
        mock_embeddings_cls.return_value.embed_documents.return_value = [[0.1] * 4]

        pages = [_make_page(1, change_type="deleted", num_sections=2)]
        index_pages(pages, "test_col", "model", 4)

        mock_vector.store.insert_batch.assert_not_called()

    @patch("wiki_rag.index.util.vector")
    @patch("wiki_rag.index.util.OpenAIEmbeddings")
    def test_index_pages_indexes_non_deleted_pages(self, mock_embeddings_cls, mock_vector):
        """Pages without change_type (full dump) are indexed normally."""
        mock_embeddings_cls.return_value.embed_documents.side_effect = _embed_side_effect()

        pages = [_make_page(1, num_sections=2)]  # no change_type key
        index_pages(pages, "test_col", "model", 4)

        self.assertEqual(2, len(_inserted_records(mock_vector)))


class TestIndexPagesSkipsEmptySections(unittest.TestCase):
    """index_pages() must skip sections whose text is empty after tidying."""

    @patch("wiki_rag.index.util.vector")
    @patch("wiki_rag.index.util.OpenAIEmbeddings")
    def test_index_pages_skips_empty_and_whitespace_sections(self, mock_embeddings_cls, mock_vector):
        """Empty and whitespace-only sections are neither embedded nor inserted."""
        mock_embeddings = mock_embeddings_cls.return_value
        mock_embeddings.embed_documents.return_value = [[0.1] * 4]

        page = _make_page(1, num_sections=3)
        page["sections"][0]["text"] = "Real content here."  # kept
        page["sections"][1]["text"] = ""  # empty (e.g. a heading-only container)
        page["sections"][2]["text"] = "   \n\t "  # whitespace-only

        index_pages([page], "test_col", "model", 4)

        # Only the single section with real content is embedded and inserted.
        self.assertEqual(1, mock_embeddings.embed_documents.call_count)
        self.assertEqual(1, mock_vector.store.insert_batch.call_count)


class TestIndexPagesTrimsTextToByteLimit(unittest.TestCase):
    """index_pages() must trim section text to the 5000-byte storage limit."""

    @patch("wiki_rag.index.util.vector")
    @patch("wiki_rag.index.util.OpenAIEmbeddings")
    def test_multibyte_text_trimmed_to_byte_limit(self, mock_embeddings_cls, mock_vector):
        """A section under 5000 chars but over 5000 bytes is trimmed to fit the byte limit."""
        mock_embeddings_cls.return_value.embed_documents.return_value = [[0.1] * 4]

        page = _make_page(1, num_sections=1)
        # 4000 three-byte characters = 12000 bytes: under 5000 chars but well over 5000 bytes.
        page["sections"][0]["text"] = "€" * 4000
        index_pages([page], "test_col", "model", 4)

        record = mock_vector.store.insert_batch.call_args.args[1][0]
        stored = record["text"]
        # Stored text fits the byte limit and stays valid UTF-8 (no split codepoint).
        self.assertLessEqual(len(stored.encode("utf-8")), 5000)
        self.assertEqual(stored, stored.encode("utf-8").decode("utf-8"))
        self.assertGreater(len(stored), 0)


class TestIndexPagesAttachesCategories(unittest.TestCase):
    """index_pages() must copy each page's categories onto every chunk record."""

    @patch("wiki_rag.index.util.vector")
    @patch("wiki_rag.index.util.OpenAIEmbeddings")
    def test_page_categories_attached_to_each_chunk(self, mock_embeddings_cls, mock_vector):
        """Every section record carries its page's categories (page-level value)."""
        mock_embeddings_cls.return_value.embed_documents.side_effect = _embed_side_effect()

        page = _make_page(1, num_sections=2)
        page["categories"] = ["Beta content", "Game Concepts"]
        index_pages([page], "test_col", "model", 4)

        records = _inserted_records(mock_vector)
        self.assertEqual(2, len(records))
        for record in records:
            self.assertEqual(["Beta content", "Game Concepts"], record["categories"])


class TestIndexPagesRetries(unittest.TestCase):
    """index_pages() must configure the embedding client's retry budget."""

    @patch("wiki_rag.index.util.vector")
    @patch("wiki_rag.index.util.OpenAIEmbeddings")
    def test_max_retries_defaults_to_three(self, mock_embeddings_cls, mock_vector):
        """The OpenAI client (which honours Retry-After) is given 3 retries by default."""
        mock_embeddings_cls.return_value.embed_documents.return_value = [[0.1] * 4]

        index_pages([_make_page(1, num_sections=1)], "test_col", "model", 4)

        self.assertEqual(3, mock_embeddings_cls.call_args.kwargs["max_retries"])

    @patch("wiki_rag.index.util.vector")
    @patch("wiki_rag.index.util.OpenAIEmbeddings")
    def test_max_retries_is_forwarded(self, mock_embeddings_cls, mock_vector):
        """A caller-supplied retry budget reaches the embedding client."""
        mock_embeddings_cls.return_value.embed_documents.return_value = [[0.1] * 4]

        index_pages([_make_page(1, num_sections=1)], "test_col", "model", 4, embedding_max_retries=7)

        self.assertEqual(7, mock_embeddings_cls.call_args.kwargs["max_retries"])


class TestIndexPagesBatching(unittest.TestCase):
    """index_pages() must embed and insert sections in batches."""

    @patch("wiki_rag.index.util.vector")
    @patch("wiki_rag.index.util.OpenAIEmbeddings")
    def test_sections_within_one_batch_use_a_single_embed_and_insert(self, mock_embeddings_cls, mock_vector):
        """Sections under the batch size are embedded and inserted in one call each."""
        mock_embeddings = mock_embeddings_cls.return_value
        mock_embeddings.embed_documents.side_effect = _embed_side_effect()

        index_pages([_make_page(1, num_sections=3)], "test_col", "model", 4, embedding_batch_size=8)

        # One embed call carrying all 3 texts, one insert call carrying all 3 records.
        self.assertEqual([3], [len(c.args[0]) for c in mock_embeddings.embed_documents.call_args_list])
        self.assertEqual([3], [len(c.args[1]) for c in mock_vector.store.insert_batch.call_args_list])
        self.assertEqual(3, len(_inserted_records(mock_vector)))

    @patch("wiki_rag.index.util.vector")
    @patch("wiki_rag.index.util.OpenAIEmbeddings")
    def test_sections_are_flushed_when_batch_size_is_reached(self, mock_embeddings_cls, mock_vector):
        """With batch size 2, five sections flush as batches of 2, 2 and 1."""
        mock_embeddings = mock_embeddings_cls.return_value
        mock_embeddings.embed_documents.side_effect = _embed_side_effect()

        index_pages([_make_page(1, num_sections=5)], "test_col", "model", 4, embedding_batch_size=2)

        self.assertEqual([2, 2, 1], [len(c.args[0]) for c in mock_embeddings.embed_documents.call_args_list])
        self.assertEqual([2, 2, 1], [len(c.args[1]) for c in mock_vector.store.insert_batch.call_args_list])
        self.assertEqual(5, len(_inserted_records(mock_vector)))

    @patch("wiki_rag.index.util.vector")
    @patch("wiki_rag.index.util.OpenAIEmbeddings")
    def test_embedding_failure_skips_the_batch_without_inserting(self, mock_embeddings_cls, mock_vector):
        """If a batch fails to embed, it is skipped (no insert) and not counted."""
        mock_embeddings_cls.return_value.embed_documents.side_effect = RuntimeError("boom")

        [_, sections] = index_pages([_make_page(1, num_sections=2)], "test_col", "model", 4)

        self.assertEqual(0, sections)
        mock_vector.store.insert_batch.assert_not_called()


class TestIndexPagesChunking(unittest.TestCase):
    """index_pages() must split oversized sections into chunked records."""

    def _run(self, pages: list[dict], **chunking) -> list[dict]:
        """Run index_pages with mocks and return all inserted records.

        Records and their texts accumulate into embedding batches, so flatten
        across every embed/insert call rather than assuming one call per chunk.
        """
        with (
            patch("wiki_rag.index.util.vector") as mock_vector,
            patch("wiki_rag.index.util.OpenAIEmbeddings") as mock_embeddings_cls,
        ):
            embed = mock_embeddings_cls.return_value.embed_documents
            embed.side_effect = _embed_side_effect()
            index_pages(pages, "test_col", "model", 4, **chunking)
            self.embed_calls = [text for call in embed.call_args_list for text in call.args[0]]
            return _inserted_records(mock_vector)

    def test_single_chunk_section_keeps_the_section_record_shape(self):
        page = _make_page(1, num_sections=1)
        page["sections"][0]["parent"] = "parent-uuid"
        page["sections"][0]["children"] = ["child-uuid"]
        [record] = self._run([page], chunk_strategy="paragraph", chunk_max_bytes=1000)

        self.assertEqual("sec-1-0", record["id"])
        self.assertEqual("sec-1-0", record["section_id"])
        self.assertEqual(0, record["chunk_index"])
        self.assertEqual("Some text content", record["text"])
        self.assertEqual("parent-uuid", record["parent"])
        self.assertEqual(["child-uuid"], record["children"])

    def test_oversized_section_produces_ordered_chunks(self):
        page = _make_page(1, num_sections=1)
        page["sections"][0]["parent"] = "parent-uuid"
        paragraphs = [(f"para{i} " * 30).strip() for i in range(4)]
        page["sections"][0]["text"] = "\n\n".join(paragraphs)
        records = self._run([page], chunk_strategy="paragraph", chunk_max_bytes=300)

        self.assertGreater(len(records), 1)
        section_id = "sec-1-0"
        for chunk_index, record in enumerate(records):
            if chunk_index == 0:
                self.assertEqual(section_id, record["id"])
            else:
                expected_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{section_id}-{chunk_index}".encode()))
                self.assertEqual(expected_id, record["id"])
            self.assertEqual(section_id, record["section_id"])
            self.assertEqual(chunk_index, record["chunk_index"])
            # Graph fields are copied verbatim from the section (section ids only).
            self.assertEqual("parent-uuid", record["parent"])
            self.assertLessEqual(len(record["text"].encode("utf-8")), 300)
        # No content lost: all paragraphs present across the chunks.
        joined = "\n\n".join(record["text"] for record in records)
        for paragraph in paragraphs:
            self.assertIn(paragraph, joined)

    def test_strategy_none_trims_exactly_like_before(self):
        page = _make_page(1, num_sections=1)
        page["sections"][0]["text"] = "€" * 4000  # 12000 bytes.
        records = self._run([page], chunk_strategy="none", chunk_max_bytes=5000)

        self.assertEqual(1, len(records))
        self.assertEqual("€" * 1666, records[0]["text"])  # 5000 bytes // 3.
        self.assertEqual(0, records[0]["chunk_index"])

    def test_each_chunk_is_embedded_with_the_preamble(self):
        page = _make_page(1, num_sections=1)
        page["sections"][0]["text"] = "\n\n".join((f"para{i} " * 30).strip() for i in range(4))
        records = self._run([page], chunk_strategy="paragraph", chunk_max_bytes=300)

        self.assertEqual(len(records), len(self.embed_calls))
        for record, embedded in zip(records, self.embed_calls, strict=True):
            self.assertEqual(f"Page 1 / Section 0\n\n{record['text']}", embedded)


class TestIndexPagesIncremental(unittest.TestCase):
    """index_pages_incremental() must route pages correctly."""

    def _run(self, pages: list[dict]) -> tuple[dict, MagicMock, MagicMock]:
        """Run index_pages_incremental with mocked vector store and embeddings."""
        with (
            patch("wiki_rag.index.util.vector") as mock_vector,
            patch("wiki_rag.index.util.OpenAIEmbeddings") as mock_embeddings_cls,
        ):
            mock_embeddings_cls.return_value.embed_documents.side_effect = _embed_side_effect()
            summary = index_pages_incremental(pages, "live_col", "model", 4)
            return summary, mock_vector, mock_embeddings_cls

    def test_incremental_deleted_pages_triggers_deletion(self):
        """delete_by_page_ids called with correct IDs; no insertions."""
        pages = [_make_page(10, change_type="deleted")]
        summary, mock_vector, _ = self._run(pages)

        mock_vector.store.delete_by_page_ids.assert_called_once_with("live_col", [10])
        mock_vector.store.insert_batch.assert_not_called()

    def test_incremental_created_pages_triggers_insertion(self):
        """Insertions only for created pages; no deletions."""
        pages = [_make_page(20, change_type="created", num_sections=1)]
        summary, mock_vector, _ = self._run(pages)

        mock_vector.store.delete_by_page_ids.assert_called_once_with("live_col", [])
        self.assertEqual(1, mock_vector.store.insert_batch.call_count)

    def test_incremental_updated_pages_deletes_then_inserts(self):
        """Updated page ID appears in deletion list and is also re-inserted."""
        pages = [_make_page(30, change_type="updated", num_sections=1)]
        summary, mock_vector, _ = self._run(pages)

        mock_vector.store.delete_by_page_ids.assert_called_once_with("live_col", [30])
        self.assertEqual(1, mock_vector.store.insert_batch.call_count)

    def test_incremental_unchanged_pages_skipped(self):
        """Pages with change_type=None trigger neither deletion nor insertion."""
        pages = [_make_page(40, change_type=None)]
        summary, mock_vector, _ = self._run(pages)

        mock_vector.store.delete_by_page_ids.assert_called_once_with("live_col", [])
        mock_vector.store.insert_batch.assert_not_called()

    def test_incremental_mixed_changes(self):
        """All four change types are routed correctly in a single call."""
        pages = [
            _make_page(1, change_type="deleted", num_sections=2),
            _make_page(2, change_type="updated", num_sections=1),
            _make_page(3, change_type="created", num_sections=3),
            _make_page(4, change_type=None, num_sections=1),
        ]
        summary, mock_vector, _ = self._run(pages)

        # Pages 1 (deleted) and 2 (updated) must be in the delete call.
        mock_vector.store.delete_by_page_ids.assert_called_once_with("live_col", [1, 2])
        # Pages 2 (updated) and 3 (created) are inserted: 1 + 3 = 4 sections.
        self.assertEqual(4, len(_inserted_records(mock_vector)))

    def test_incremental_returns_summary_counts(self):
        """Returned summary dict contains correct per-category counts."""
        pages = [
            _make_page(1, change_type="deleted"),
            _make_page(2, change_type="updated", num_sections=2),
            _make_page(3, change_type="created", num_sections=1),
            _make_page(4, change_type=None),
        ]
        summary, _, _ = self._run(pages)

        self.assertEqual(1, summary["deleted"])
        self.assertEqual(1, summary["updated"])
        self.assertEqual(1, summary["created"])
        self.assertEqual(1, summary["skipped"])
        self.assertEqual(3, summary["sections_indexed"])  # 2 (updated) + 1 (created)


class TestReplacePreviousCollection(unittest.TestCase):
    """replace_previous_collection() must flush, compact, drop old, rename temp, then load."""

    @patch("wiki_rag.index.util.vector")
    def test_replace_drops_old_renames_temp_and_loads(self, mock_vector):
        """flush, compact, drop, rename, and load are called in the correct order."""
        mock_vector.store.collection_exists.return_value = True

        replace_previous_collection("my_col", "my_col_temp")

        mock_vector.store.flush_collection.assert_called_once_with("my_col_temp")
        mock_vector.store.compact_collection.assert_called_once_with("my_col_temp")
        mock_vector.store.drop_collection.assert_called_once_with("my_col")
        mock_vector.store.rename_collection.assert_called_once_with("my_col_temp", "my_col")
        mock_vector.store.load_collection.assert_called_once_with("my_col")

        calls = [c[0] for c in mock_vector.store.method_calls]
        self.assertLess(calls.index("flush_collection"), calls.index("compact_collection"))


if __name__ == "__main__":
    unittest.main()


class TestIdempotencyMarker(unittest.TestCase):
    """Marker helpers must invalidate the skip when chunking settings change."""

    def test_signature_is_normalised_for_strategy_none(self):
        self.assertEqual("none:5000:0", chunking_signature("none", 3000, 300))

    def test_signature_records_strategy_and_sizes(self):
        self.assertEqual("paragraph:3000:300", chunking_signature("paragraph", 3000, 300))

    def test_marker_roundtrip_matches(self):
        content = marker_content("dump-2026.json", "paragraph:3000:300")
        self.assertTrue(marker_matches(content, "dump-2026.json", "paragraph:3000:300"))

    def test_legacy_single_line_marker_matches_only_none(self):
        self.assertTrue(marker_matches("dump-2026.json", "dump-2026.json", "none:5000:0"))
        self.assertFalse(marker_matches("dump-2026.json", "dump-2026.json", "paragraph:3000:300"))

    def test_different_dump_name_does_not_match(self):
        content = marker_content("dump-old.json", "none:5000:0")
        self.assertFalse(marker_matches(content, "dump-new.json", "none:5000:0"))

    def test_changed_chunking_settings_do_not_match(self):
        content = marker_content("dump-2026.json", "paragraph:3000:300")
        self.assertFalse(marker_matches(content, "dump-2026.json", "paragraph:1500:300"))

    def test_empty_marker_does_not_match(self):
        self.assertFalse(marker_matches("", "dump-2026.json", "none:5000:0"))

    def test_marker_signature_reads_recorded_signature(self):
        content = marker_content("dump-2026.json", "paragraph:3000:300")
        self.assertEqual("paragraph:3000:300", marker_signature(content))

    def test_marker_signature_defaults_legacy_marker_to_none(self):
        self.assertEqual("none:5000:0", marker_signature("dump-2026.json"))
