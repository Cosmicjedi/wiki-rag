#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.search.util tests."""

import unittest

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from wiki_rag.search.util import (
    RagState,
    build_poc_context,
    load_prompts_for_rag_from_local,
    optimise,
    retrieve_all_elements,
    route_after_rewrite,
)


def _make_context(
    enable_hyde: bool = False,
    contextualisation_model: str = "",
    hyde_model: str | None = None,
    hyde_passages: int = 1,
) -> SimpleNamespace:
    """Build a minimal runtime-like namespace for routing tests."""
    # hyde_model falls back to contextualisation_model, mirroring build_context_schema().
    resolved_hyde_model = hyde_model or contextualisation_model or ""
    return SimpleNamespace(context={
        "hyde_enabled": enable_hyde,
        "contextualisation_model": contextualisation_model,
        "hyde_model": resolved_hyde_model,
        "hyde_passages": hyde_passages,
    })


def _make_state(**kwargs) -> RagState:
    """Build a minimal RagState for routing tests."""
    defaults: dict = {
        "history": [],
        "question": "What is a Moodle quiz?",
        "hyde_texts": [],
        "vector_search": [],
        "context": [],
        "sources": [],
    }
    defaults.update(kwargs)
    return defaults  # type: ignore[return-value]


class TestRouteAfterRewrite(unittest.TestCase):

    def test_route_after_rewrite_chitchat(self):
        """Returns 'chitchat' when answer is already set, regardless of hyde settings."""
        state = _make_state(answer="Hello there!")
        runtime = _make_context(enable_hyde=True, contextualisation_model="gpt-4o")
        self.assertEqual("chitchat", route_after_rewrite(state, runtime))  # type: ignore[arg-type]

    def test_route_after_rewrite_hyde_enabled(self):
        """Returns 'hyde_rewrite' when hyde_enabled is True and a model is available."""
        state = _make_state()
        runtime = _make_context(enable_hyde=True, contextualisation_model="gpt-4o-mini")
        self.assertEqual("hyde_rewrite", route_after_rewrite(state, runtime))  # type: ignore[arg-type]

    def test_route_after_rewrite_hyde_enabled_with_dedicated_model(self):
        """Returns 'hyde_rewrite' when hyde_enabled is True and hyde_model is set directly."""
        state = _make_state()
        runtime = _make_context(enable_hyde=True, hyde_model="hyde-specific-model")
        self.assertEqual("hyde_rewrite", route_after_rewrite(state, runtime))  # type: ignore[arg-type]

    def test_route_after_rewrite_hyde_disabled(self):
        """Returns 'retrieve' when hyde_enabled is False, even with a model set."""
        state = _make_state()
        runtime = _make_context(enable_hyde=False, contextualisation_model="gpt-4o-mini")
        self.assertEqual("retrieve", route_after_rewrite(state, runtime))  # type: ignore[arg-type]

    def test_route_after_rewrite_hyde_no_model(self):
        """Returns 'retrieve' when hyde_enabled is True but no model is available."""
        state = _make_state()
        runtime = _make_context(enable_hyde=True, contextualisation_model="")
        self.assertEqual("retrieve", route_after_rewrite(state, runtime))  # type: ignore[arg-type]

    def test_route_after_rewrite_default_retrieve(self):
        """Returns 'retrieve' when hyde is disabled and no contextualisation model."""
        state = _make_state()
        runtime = _make_context(enable_hyde=False, contextualisation_model="")
        self.assertEqual("retrieve", route_after_rewrite(state, runtime))  # type: ignore[arg-type]


class TestLoadPromptsForRagFromLocalHyde(unittest.TestCase):

    def test_hyde_prompt_input_variables(self):
        """The wiki-rag-hyde prompt exposes the expected input variables."""
        prompt = load_prompts_for_rag_from_local("wiki-rag-hyde")
        variables = set(prompt.input_variables)
        self.assertIn("question", variables)
        self.assertIn("task_def", variables)
        self.assertIn("kb_name", variables)
        self.assertIn("product", variables)

    def test_hyde_prompt_message_count(self):
        """The wiki-rag-hyde prompt has system + history placeholder + user messages."""
        from langchain_core.prompts import MessagesPlaceholder
        prompt = load_prompts_for_rag_from_local("wiki-rag-hyde")
        self.assertEqual(3, len(prompt.messages))
        self.assertIsInstance(prompt.messages[1], MessagesPlaceholder)


class TestRetrieveNode(unittest.IsolatedAsyncioTestCase):

    def _make_runtime(self) -> SimpleNamespace:
        return SimpleNamespace(context={
            "collection_name": "test_collection",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 512,
            "embedding_api_base": "https://api.example.com/v1",
            "embedding_api_key": "test-embed-key",  # pragma: allowlist secret
        })

    async def test_retrieve_without_hyde_uses_question_for_both(self):
        """Without HyDE, the question is passed as queries[0]; sparse_query is None."""
        from wiki_rag.search.util import retrieve

        state = _make_state(question="What is a quiz?", hyde_texts=[])
        runtime = self._make_runtime()

        mock_store = MagicMock()
        mock_store.retrieve = MagicMock(return_value=[{"id": "1"}])

        with patch("wiki_rag.search.util.vector") as mock_vector:
            mock_vector.store = mock_store
            await retrieve(state, runtime)  # type: ignore[arg-type]

        mock_store.retrieve.assert_called_once_with(
            collection_name="test_collection",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=512,
            queries=["What is a quiz?"],
            sparse_query=None,
            embedding_api_base="https://api.example.com/v1",
            embedding_api_key="test-embed-key",  # pragma: allowlist secret
        )

    async def test_retrieve_with_hyde_uses_passages_for_dense_question_for_sparse(self):
        """With HyDE, only passages go to dense queries; original question is sparse_query."""
        from wiki_rag.search.util import retrieve

        passages = ["A quiz is a formative assessment tool.", "Teachers use quizzes to test recall."]
        state = _make_state(question="What is a quiz?", hyde_texts=passages)
        runtime = self._make_runtime()

        mock_store = MagicMock()
        mock_store.retrieve = MagicMock(return_value=[{"id": "1"}])

        with patch("wiki_rag.search.util.vector") as mock_vector:
            mock_vector.store = mock_store
            await retrieve(state, runtime)  # type: ignore[arg-type]

        mock_store.retrieve.assert_called_once_with(
            collection_name="test_collection",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=512,
            queries=passages,
            sparse_query="What is a quiz?",
            embedding_api_base="https://api.example.com/v1",
            embedding_api_key="test-embed-key",  # pragma: allowlist secret
        )

    async def test_retrieve_with_empty_hyde_texts_falls_back_to_question(self):
        """An empty hyde_texts list is treated the same as no HyDE (falsy check)."""
        from wiki_rag.search.util import retrieve

        state = _make_state(question="What is a quiz?", hyde_texts=[])
        runtime = self._make_runtime()

        mock_store = MagicMock()
        mock_store.retrieve = MagicMock(return_value=[])

        with patch("wiki_rag.search.util.vector") as mock_vector:
            mock_vector.store = mock_store
            await retrieve(state, runtime)  # type: ignore[arg-type]

        mock_store.retrieve.assert_called_once_with(
            collection_name="test_collection",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=512,
            queries=["What is a quiz?"],
            sparse_query=None,
            embedding_api_base="https://api.example.com/v1",
            embedding_api_key="test-embed-key",  # pragma: allowlist secret
        )


class TestHydeNode(unittest.IsolatedAsyncioTestCase):

    async def test_hyde_node_single_passage_returns_texts(self):
        """The hyde_rewrite node returns hyde_texts with the generated passage."""
        from wiki_rag.search.util import hyde_rewrite

        fake_passage = "Moodle quizzes allow teachers to build question banks."

        mock_llm_response = MagicMock()
        mock_llm_response.content = fake_passage

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        mock_prompt = AsyncMock()
        mock_prompt.ainvoke = AsyncMock(return_value=MagicMock())

        state = _make_state()
        runtime = _make_context(
            enable_hyde=True,
            contextualisation_model="gpt-4o-mini",
            hyde_passages=1,
        )
        runtime.context.update({
            "prompt_name": "wiki-rag",
            "product": "Moodle",
            "task_def": "Moodle user documentation",
            "kb_name": "Moodle Docs",
            "hyde_api_base": "https://api.example.com/v1",
            "hyde_api_key": "test-hyde-key",  # pragma: allowlist secret
        })

        with (
            patch("wiki_rag.search.util.load_prompts_for_rag", return_value=mock_prompt),
            patch("wiki_rag.search.util.ChatOpenAI", MagicMock(return_value=mock_llm)),
        ):
            result = await hyde_rewrite(state, runtime)  # type: ignore[arg-type]

        self.assertIn("hyde_texts", result)
        self.assertEqual([fake_passage], result["hyde_texts"])

    async def test_hyde_node_multiple_passages_returns_all_texts(self):
        """The hyde_rewrite node returns hyde_texts with all generated passages."""
        from wiki_rag.search.util import hyde_rewrite

        fake_passage = "Some passage."

        mock_llm_response = MagicMock()
        mock_llm_response.content = fake_passage

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        mock_prompt = AsyncMock()
        mock_prompt.ainvoke = AsyncMock(return_value=MagicMock())

        state = _make_state()
        runtime = _make_context(
            enable_hyde=True,
            contextualisation_model="gpt-4o-mini",
            hyde_passages=3,
        )
        runtime.context.update({
            "prompt_name": "wiki-rag",
            "product": "Moodle",
            "task_def": "Moodle user documentation",
            "kb_name": "Moodle Docs",
            "hyde_api_base": "https://api.example.com/v1",
            "hyde_api_key": "test-hyde-key",  # pragma: allowlist secret
        })

        with (
            patch("wiki_rag.search.util.load_prompts_for_rag", return_value=mock_prompt),
            patch("wiki_rag.search.util.ChatOpenAI", MagicMock(return_value=mock_llm)),
        ):
            result = await hyde_rewrite(state, runtime)  # type: ignore[arg-type]

        self.assertIn("hyde_texts", result)
        self.assertEqual(3, len(result["hyde_texts"]))
        self.assertTrue(all(t == fake_passage for t in result["hyde_texts"]))


def _make_doc(
    id: str,
    distance: float = 1.0,
    parent: str | None = None,
    children: list[str] | None = None,
    previous: list[str] | None = None,
    next: list[str] | None = None,  # noqa: A002
    title: str | None = None,
    text: str = "some text",
    source: str = "https://example.com/page#anchor",
    doc_title: str = "Doc",
    **extra,
) -> dict:
    """Build a retrieved doc shaped like QdrantVector.retrieve() output."""
    return {
        "id": id,
        "distance": distance,
        "entity": {
            "id": id,
            "title": title if title is not None else f"Title {id}",
            "text": text,
            "source": source,
            "doc_title": doc_title,
            "parent": parent,
            "children": children or [],
            "previous": previous or [],
            "next": next or [],
            **extra,
        },
    }


def _poc_runtime() -> SimpleNamespace:
    """Build a minimal runtime-like namespace for optimise tests."""
    return SimpleNamespace(context={"collection_name": "test_col"})


class TestOptimise(unittest.IsolatedAsyncioTestCase):
    """Characterise the optimise() popularity weighting."""

    async def test_empty_vector_search_returns_empty_context(self):
        state = _make_state(vector_search=[])
        result = await optimise(state, _poc_runtime())  # type: ignore[arg-type]
        self.assertEqual({"context": [], "sources": []}, result)

    async def test_sorted_items_are_ranked_by_weighted_popularity(self):
        # "a" is hit directly AND referenced as parent by "b": it must
        # outrank "b" (hit once) and "c" (only referenced as a child).
        docs = [
            _make_doc("a", distance=1.0),
            _make_doc("b", distance=1.0, parent="a", children=["c"]),
        ]
        state = _make_state(vector_search=docs)
        with patch("wiki_rag.search.util.build_poc_context", return_value=[[], []]) as mock_poc:
            await optimise(state, _poc_runtime())  # type: ignore[arg-type]
        sorted_items = mock_poc.call_args.kwargs["sorted_items"]
        self.assertEqual(["a", "b", "c"], [item[0] for item in sorted_items])

    async def test_earlier_position_weighs_more_than_later(self):
        docs = [
            _make_doc("first", distance=1.0),
            _make_doc("second", distance=1.0),
        ]
        state = _make_state(vector_search=docs)
        with patch("wiki_rag.search.util.build_poc_context", return_value=[[], []]) as mock_poc:
            await optimise(state, _poc_runtime())  # type: ignore[arg-type]
        sorted_items = dict(mock_poc.call_args.kwargs["sorted_items"])
        self.assertGreater(sorted_items["first"], sorted_items["second"])


class TestBuildPocContext(unittest.TestCase):
    """Characterise the POC (parent-own-children) context expansion."""

    def test_parent_self_children_order_without_duplicates(self):
        docs = [
            _make_doc("own", distance=1.0, parent="dad", children=["kid1", "kid2"]),
            _make_doc("dad", distance=0.5),
            _make_doc("kid1", distance=0.4),
            _make_doc("kid2", distance=0.3),
        ]
        sorted_items = [("own", 2.0), ("dad", 1.0), ("kid1", 0.5), ("kid2", 0.4)]
        with patch("wiki_rag.search.util.get_missing_from_vector_store", return_value={}):
            context, sources = build_poc_context(
                retrieved_docs=docs, sorted_items=sorted_items,
                collection_name="test_col", top=4,
            )
        # Parent before self, children after; later sorted entries add nothing new.
        self.assertEqual(
            ["Title dad\n\nsome text", "Title own\n\nsome text",
             "Title kid1\n\nsome text", "Title kid2\n\nsome text"],
            context,
        )

    def test_sources_include_doc_title_only_for_non_top_level_sections(self):
        docs = [
            _make_doc("own", distance=1.0, parent="dad"),
            _make_doc("dad", distance=0.5),
        ]
        sorted_items = [("own", 2.0), ("dad", 1.0)]
        with patch("wiki_rag.search.util.get_missing_from_vector_store", return_value={}):
            _, sources = build_poc_context(
                retrieved_docs=docs, sorted_items=sorted_items,
                collection_name="test_col", top=2,
            )
        # "dad" entered the context as a parent (not as "own"), so it gets
        # no source link of its own — only directly selected sections do.
        self.assertEqual(["[https://example.com/page#anchor|Doc: Title own]"], sources)

    def test_not_retrieved_elements_are_appended_at_the_end(self):
        docs = [_make_doc("own", distance=1.0)]
        sorted_items = [("ghost", 3.0), ("own", 2.0)]
        with patch(
            "wiki_rag.search.util.get_missing_from_vector_store",
            return_value={"ghost": "Ghost\n\nghost text"},
        ):
            context, _ = build_poc_context(
                retrieved_docs=docs, sorted_items=sorted_items,
                collection_name="test_col", top=1,
            )
        self.assertEqual(["Title own\n\nsome text", "Ghost\n\nghost text"], context)


class TestRetrieveAllElements(unittest.TestCase):
    """Characterise the context text retrieval (cache + store fallback)."""

    def test_cached_and_missing_texts_keep_context_list_order(self):
        docs = [_make_doc("cached", text="cached text")]
        with patch(
            "wiki_rag.search.util.get_missing_from_vector_store",
            return_value={"missing": "Missing\n\nfetched text"},
        ) as mock_missing:
            result = retrieve_all_elements(docs, ["missing", "cached"], "test_col")
        self.assertEqual(["Missing\n\nfetched text", "Title cached\n\ncached text"], result)
        mock_missing.assert_called_once_with(["missing"], "test_col")

    def test_unresolved_id_is_dropped(self):
        """An id neither cached nor returned by the store is dropped, not kept as None.

        Regression guard: the returned list must never leak None entries for ids
        that could not be resolved to any text.
        """
        docs = [_make_doc("cached", text="cached text")]
        with patch("wiki_rag.search.util.get_missing_from_vector_store", return_value={}):
            result = retrieve_all_elements(docs, ["cached", "missing"], "test_col")
        self.assertEqual(["Title cached\n\ncached text"], result)

    def test_store_none_text_is_dropped(self):
        """An id the store resolves to None (e.g. an empty section) is dropped."""
        docs = [_make_doc("cached", text="cached text")]
        with patch(
            "wiki_rag.search.util.get_missing_from_vector_store",
            return_value={"empty": None},
        ):
            result = retrieve_all_elements(docs, ["cached", "empty"], "test_col")
        self.assertEqual(["Title cached\n\ncached text"], result)


class TestOptimiseWithChunks(unittest.IsolatedAsyncioTestCase):
    """Chunked records must pool popularity under their owning section."""

    async def test_chunks_of_the_same_section_pool_their_weights(self):
        # Two chunks of section "s1" plus one other section: "s1" must
        # accumulate both hits (and outrank "other" despite equal distances).
        docs = [
            _make_doc("s1", distance=1.0, section_id="s1", chunk_index=0),
            _make_doc("other", distance=1.0, section_id="other", chunk_index=0),
            _make_doc("s1-chunk1", distance=1.0, section_id="s1", chunk_index=1),
        ]
        state = _make_state(vector_search=docs)
        with patch("wiki_rag.search.util.build_poc_context", return_value=[[], []]) as mock_poc:
            await optimise(state, _poc_runtime())  # type: ignore[arg-type]
        sorted_items = mock_poc.call_args.kwargs["sorted_items"]
        self.assertEqual("s1", sorted_items[0][0])
        self.assertNotIn("s1-chunk1", [item[0] for item in sorted_items])


class TestBuildPocContextWithChunks(unittest.TestCase):
    """POC expansion must match retrieved chunks by their owning section."""

    def test_hit_on_a_later_chunk_still_exposes_section_metadata(self):
        docs = [
            _make_doc("s1-chunk1", section_id="s1", chunk_index=1,
                      parent="dad", children=["kid"]),
        ]
        sorted_items = [("s1", 2.0)]
        with patch(
            "wiki_rag.search.util.get_missing_from_vector_store",
            return_value={
                "dad": "Dad\n\ndad text",
                "s1": "Title s1-chunk1\n\nfirst\n\nsecond",
                "kid": "Kid\n\nkid text",
            },
        ) as mock_missing:
            context, sources = build_poc_context(
                retrieved_docs=docs, sorted_items=sorted_items,
                collection_name="test_col", top=1,
            )
        # Parent, own (reassembled) and children, in POC order.
        self.assertEqual(
            ["Dad\n\ndad text", "Title s1-chunk1\n\nfirst\n\nsecond", "Kid\n\nkid text"],
            context,
        )
        self.assertEqual(["dad", "s1", "kid"], mock_missing.call_args.args[0])
        self.assertEqual(1, len(sources))


class TestRetrieveAllElementsWithChunks(unittest.TestCase):
    """Sections from chunk-aware collections are always fetched whole."""

    def test_chunk_aware_records_are_not_trusted_as_whole_sections(self):
        # Even a chunk_index 0 hit may be partial (total chunk count is
        # unknown), so the section must be fetched from the store.
        docs = [_make_doc("s1", section_id="s1", chunk_index=0, text="partial")]
        with patch(
            "wiki_rag.search.util.get_missing_from_vector_store",
            return_value={"s1": "Title s1\n\nfull\n\nsection"},
        ) as mock_missing:
            result = retrieve_all_elements(docs, ["s1"], "test_col")
        self.assertEqual(["Title s1\n\nfull\n\nsection"], result)
        mock_missing.assert_called_once_with(["s1"], "test_col")

    def test_legacy_records_keep_the_cache_fast_path(self):
        docs = [_make_doc("legacy", text="whole text")]
        with patch(
            "wiki_rag.search.util.get_missing_from_vector_store", return_value={},
        ) as mock_missing:
            result = retrieve_all_elements(docs, ["legacy"], "test_col")
        self.assertEqual(["Title legacy\n\nwhole text"], result)
        mock_missing.assert_called_once_with([], "test_col")


if __name__ == "__main__":
    unittest.main()
