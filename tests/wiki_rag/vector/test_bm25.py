#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.vector.bm25 tests."""

import math
import unittest

from qdrant_client import QdrantClient, models

from wiki_rag.vector.bm25 import (
    AVG_LEN,
    K1,
    B,
    encode_document,
    encode_query,
    normalise_score,
    porter_stem,
    term_index,
    tokenize,
)


class TestPorterStem(unittest.TestCase):
    # Pairs taken from the reference vocabulary published with the algorithm.
    REFERENCE = {
        "caresses": "caress", "ponies": "poni", "ties": "ti", "cats": "cat",
        "feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled",
        "motoring": "motor", "sing": "sing", "conflated": "conflat",
        "troubled": "troubl", "sized": "size", "hopping": "hop", "tanned": "tan",
        "falling": "fall", "hissing": "hiss", "fizzed": "fizz", "failing": "fail",
        "filing": "file", "happy": "happi", "sky": "sky", "relational": "relat",
        "conditional": "condit", "rational": "ration", "valenci": "valenc",
        "digitizer": "digit", "conformabli": "conform", "radicalli": "radic",
        "differentli": "differ", "vileli": "vile", "analogousli": "analog",
        "vietnamization": "vietnam", "predication": "predic", "operator": "oper",
        "feudalism": "feudal", "decisiveness": "decis", "hopefulness": "hope",
        "callousness": "callous", "formaliti": "formal", "sensitiviti": "sensit",
        "sensibiliti": "sensibl", "triplicate": "triplic", "formative": "form",
        "formalize": "formal", "electriciti": "electr", "electrical": "electr",
        "hopeful": "hope", "goodness": "good", "revival": "reviv",
        "allowance": "allow", "inference": "infer", "airliner": "airlin",
        "gyroscopic": "gyroscop", "adjustable": "adjust", "defensible": "defens",
        "irritant": "irrit", "replacement": "replac", "adjustment": "adjust",
        "dependent": "depend", "adoption": "adopt", "homologou": "homolog",
        "communism": "commun", "activate": "activ", "angulariti": "angular",
        "homologous": "homolog", "effective": "effect", "bowdlerize": "bowdler",
        "probate": "probat", "rate": "rate", "cease": "ceas", "controll": "control",
        "roll": "roll", "generalizations": "gener", "oscillators": "oscil",
    }

    def test_reference_vocabulary(self):
        for word, expected in self.REFERENCE.items():
            with self.subTest(word=word):
                self.assertEqual(expected, porter_stem(word))

    def test_short_words_untouched(self):
        self.assertEqual("as", porter_stem("as"))
        self.assertEqual("i", porter_stem("i"))

    def test_idempotent_on_common_stems(self):
        for word in ("configur", "administr", "gradebook", "moodl"):
            self.assertEqual(word, porter_stem(word))


class TestTokenize(unittest.TestCase):
    def test_lowercases_strips_stopwords_and_stems(self):
        self.assertEqual(
            ["moodl", "gradebook", "set", "configur", "administr"],
            tokenize("The Moodle's gradebook settings are configured by the administrators!"),
        )

    def test_underscores_and_punctuation_separate_words(self):
        self.assertEqual(["wiki", "link", "foo", "bar"], tokenize("wiki_link foo-bar"))

    def test_digits_are_tokens(self):
        self.assertEqual(["moodl", "4", "5"], tokenize("Moodle 4.5"))

    def test_unicode_letters_kept(self):
        tokens = tokenize("Configuración évaluée")
        self.assertEqual(2, len(tokens))
        self.assertEqual("configuración", tokens[0])
        self.assertTrue(tokens[1].startswith("évalu"))

    def test_only_stopwords_yields_nothing(self):
        self.assertEqual([], tokenize("the of and"))

    def test_preserves_duplicates_in_order(self):
        self.assertEqual(["grade", "grade", "book"], tokenize("grade grade book"))


class TestTermIndex(unittest.TestCase):
    def test_stable_and_within_u32(self):
        self.assertEqual(term_index("gradebook"), term_index("gradebook"))
        self.assertLess(term_index("gradebook"), 2**32)
        self.assertNotEqual(term_index("gradebook"), term_index("gradebooks"))


class TestEncodeDocument(unittest.TestCase):
    def test_weights_follow_bm25_term_frequency(self):
        vector = encode_document("grade grade grade book")
        tokens = tokenize("grade grade grade book")
        doc_len = len(tokens)
        norm = K1 * (1 - B + B * doc_len / AVG_LEN)
        expected = {
            term_index("grade"): 3 * (K1 + 1) / (3 + norm),
            term_index("book"): 1 * (K1 + 1) / (1 + norm),
        }
        self.assertEqual(sorted(expected), vector.indices)
        for index, value in zip(vector.indices, vector.values, strict=True):
            self.assertAlmostEqual(expected[index], value)

    def test_repeated_terms_weigh_more_but_saturate(self):
        once = dict(zip(*[encode_document("grade").indices, encode_document("grade").values], strict=True))
        thrice = dict(zip(*[encode_document("grade grade grade").indices, encode_document("grade grade grade").values],
                          strict=True))
        index = term_index("grade")
        self.assertLess(once[index], thrice[index])
        self.assertLess(thrice[index], K1 + 1)  # the BM25 saturation ceiling

    def test_longer_documents_are_penalised(self):
        short = encode_document("grade")
        long = encode_document("grade " + " ".join(f"filler{i}" for i in range(200)))
        index = term_index("grade")
        self.assertGreater(
            short.values[short.indices.index(index)],
            long.values[long.indices.index(index)],
        )

    def test_empty_document(self):
        vector = encode_document("the and of")
        self.assertEqual([], vector.indices)
        self.assertEqual([], vector.values)

    def test_indices_sorted_and_unique(self):
        vector = encode_document("one two three two one two")
        self.assertEqual(sorted(set(vector.indices)), vector.indices)


class TestEncodeQuery(unittest.TestCase):
    def test_unit_weights_over_unique_terms(self):
        vector = encode_query("grading the gradebook and grading")
        self.assertEqual(sorted({term_index("grade"), term_index("gradebook")}), vector.indices)
        self.assertEqual([1.0, 1.0], vector.values)

    def test_query_and_document_share_dimensions(self):
        self.assertTrue(set(encode_query("configured").indices) <= set(encode_document("Configuration").indices))


class TestNormaliseScore(unittest.TestCase):
    def test_maps_to_unit_interval_monotonically(self):
        self.assertEqual(0.0, normalise_score(0.0))
        self.assertEqual(0.0, normalise_score(-3.0))
        self.assertLess(normalise_score(1.0), normalise_score(5.0))
        self.assertLess(normalise_score(100.0), 1.0)
        self.assertAlmostEqual(0.5, normalise_score(1.0))
        self.assertAlmostEqual(2 / math.pi * math.atan(7.0), normalise_score(7.0))


class TestScoringInQdrant(unittest.TestCase):
    """The encoder's contract with Qdrant: IDF applied server side, BM25 ranking out."""

    def test_idf_modifier_ranks_rare_terms_higher(self):
        client = QdrantClient(":memory:")
        client.create_collection(
            "c", vectors_config={},
            sparse_vectors_config={"s": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        docs = {
            1: "moodle course page",          # "moodle" appears in every document
            2: "moodle gradebook page",       # "gradebook" is rare
            3: "moodle forum page",
        }
        client.upsert("c", points=[
            models.PointStruct(id=i, vector={"s": encode_document(text)}, payload={"text": text})
            for i, text in docs.items()
        ], wait=True)

        # A query with both a common and a rare term ranks the rare-term doc first.
        hits = client.query_points("c", query=encode_query("moodle gradebook"), using="s", limit=3).points
        self.assertEqual(2, hits[0].id)
        self.assertGreater(hits[0].score, hits[1].score)

        # A term absent from the corpus matches nothing.
        self.assertEqual([], client.query_points("c", query=encode_query("quiz"), using="s", limit=3).points)
