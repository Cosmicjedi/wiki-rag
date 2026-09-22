#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Tests for wiki_rag.config."""

import os
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

import wiki_rag.config as config_module

from wiki_rag.config import (
    Config,
    _parse_bool,
    _parse_excluded,
    _parse_float,
    _parse_int,
    _parse_list,
    _parse_namespaces,
    _resolve_yaml,
    load_config,
)

# ---------------------------------------------------------------------------
# Minimal valid env for a given command (non-secret fields only)
# ---------------------------------------------------------------------------

_MINIMAL_ENV_ALL = {
    "MEDIAWIKI_URL": "https://example.com",
    "MEDIAWIKI_NAMESPACES": "0,4",
    "COLLECTION_NAME": "test_col",
}

_MINIMAL_ENV_INDEX = {
    **_MINIMAL_ENV_ALL,
    "OPENAI_API_BASE": "https://api.example.com/v1",
    "EMBEDDING_MODEL": "text-embed",
    "EMBEDDING_DIMENSIONS": "768",
}

_MINIMAL_ENV_SEARCH = {
    **_MINIMAL_ENV_INDEX,
    "LLM_MODEL": "gpt-4o",
}

_MINIMAL_ENV_SERVER = {
    **_MINIMAL_ENV_SEARCH,
    "WRAPPER_API_BASE": "0.0.0.0:8080",
}

_MINIMAL_ENV_MCP = {
    **_MINIMAL_ENV_SEARCH,
    "MCP_API_BASE": "0.0.0.0:8081",
}


def _write_yaml(content: str) -> Path:
    """Write YAML content to a named temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


class TestResolveYaml(unittest.TestCase):

    def test_simple_key(self):
        data = {"mediawiki": {"url": "https://x.com"}}
        self.assertEqual("https://x.com", _resolve_yaml(data, "mediawiki.url"))

    def test_deeply_nested(self):
        data = {"a": {"b": {"c": 42}}}
        self.assertEqual(42, _resolve_yaml(data, "a.b.c"))

    def test_missing_key_returns_none(self):
        data = {"a": {"b": 1}}
        self.assertIsNone(_resolve_yaml(data, "a.c"))

    def test_missing_top_level_returns_none(self):
        self.assertIsNone(_resolve_yaml({}, "mediawiki.url"))

    def test_non_dict_intermediate_returns_none(self):
        data = {"a": "string"}
        self.assertIsNone(_resolve_yaml(data, "a.b"))


class TestParseNamespaces(unittest.TestCase):

    def test_comma_separated_string(self):
        self.assertEqual([0, 4, 12], _parse_namespaces("0,4,12"))

    def test_string_with_spaces(self):
        self.assertEqual([0, 4], _parse_namespaces(" 0 , 4 "))

    def test_native_list(self):
        self.assertEqual([0, 4], _parse_namespaces([4, 0]))

    def test_deduplication(self):
        self.assertEqual([0, 4], _parse_namespaces("0,4,0,4"))

    def test_empty_returns_empty(self):
        self.assertEqual([], _parse_namespaces(""))
        self.assertEqual([], _parse_namespaces(None))


class TestParseList(unittest.TestCase):

    def test_comma_separated(self):
        self.assertEqual(["a", "b"], _parse_list("a, b"))

    def test_native_list(self):
        self.assertEqual(["x", "y"], _parse_list(["x", "y"]))

    def test_empty_string(self):
        self.assertEqual([], _parse_list(""))

    def test_none(self):
        self.assertEqual([], _parse_list(None))


class TestParseExcluded(unittest.TestCase):

    def test_semicolon_format(self):
        raw = "categories:Cat A, Cat B;wikitext:Some regex"
        result = _parse_excluded(raw)
        self.assertEqual(["Cat A", "Cat B"], result["categories"])
        self.assertEqual(["Some regex"], result["wikitext"])

    def test_native_dict(self):
        raw = {"categories": ["Cat A"], "wikitext": ["regex"]}
        self.assertEqual(raw, _parse_excluded(raw))

    def test_empty_returns_empty_dict(self):
        self.assertEqual({}, _parse_excluded(""))
        self.assertEqual({}, _parse_excluded(None))

    def test_no_colon_entry_skipped(self):
        result = _parse_excluded("bad_entry;categories:A, B")
        self.assertNotIn("bad_entry", result)
        self.assertEqual(["A", "B"], result["categories"])


class TestParseBool(unittest.TestCase):

    def test_true_string(self):
        self.assertTrue(_parse_bool("true"))
        self.assertTrue(_parse_bool("True"))
        self.assertTrue(_parse_bool("TRUE"))

    def test_false_string(self):
        self.assertFalse(_parse_bool("false"))

    def test_native_bool(self):
        self.assertTrue(_parse_bool(True))
        self.assertFalse(_parse_bool(False))

    def test_none_uses_default(self):
        self.assertTrue(_parse_bool(None, default=True))
        self.assertFalse(_parse_bool(None, default=False))

    def test_empty_string_uses_default(self):
        self.assertFalse(_parse_bool("", default=False))


class TestParseInt(unittest.TestCase):

    def test_string_int(self):
        self.assertEqual(768, _parse_int("768"))

    def test_native_int(self):
        self.assertEqual(42, _parse_int(42))

    def test_none_uses_default(self):
        self.assertEqual(0, _parse_int(None))
        self.assertEqual(10, _parse_int(None, default=10))


class TestParseFloat(unittest.TestCase):

    def test_string_float(self):
        self.assertAlmostEqual(0.6, _parse_float("0.6"))

    def test_native_float(self):
        self.assertAlmostEqual(0.85, _parse_float(0.85))

    def test_none_uses_default(self):
        self.assertAlmostEqual(0.6, _parse_float(None, default=0.6))


class TestLoadConfigEnvOnly(unittest.TestCase):
    """Verify that env-only setup (no config.yml) works correctly."""

    def setUp(self):
        # Reset the global singleton before each test.
        config_module.cfg = None

    def _load(self, command: str, extra_env: dict | None = None) -> Config:
        """Load config with a clean env, without loading the real .env file."""
        base = _MINIMAL_ENV_SERVER if command == "server" else (
            _MINIMAL_ENV_MCP if command == "mcp" else (
                _MINIMAL_ENV_SEARCH if command in {"search"} else (
                    _MINIMAL_ENV_INDEX if command == "index" else _MINIMAL_ENV_ALL
                )
            )
        )
        env = {**base, **(extra_env or {})}
        # Patch load_dotenv so the real .env file is never loaded.
        # Use a non-existent config.yml path so YAML loading is also skipped.
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            return load_config(command=command, config_path=Path("/nonexistent/config.yml"))

    def test_load_command_returns_config(self):
        cfg = self._load("load")
        self.assertIsInstance(cfg, Config)

    def test_sites_list_has_one_entry_for_env_only(self):
        cfg = self._load("load")
        self.assertEqual(1, len(cfg.sites))

    def test_mediawiki_url_set(self):
        cfg = self._load("load")
        self.assertEqual("https://example.com", cfg.mediawiki.url)

    def test_mediawiki_property_equals_sites_zero(self):
        cfg = self._load("load")
        self.assertIs(cfg.sites[0], cfg.mediawiki)

    def test_namespaces_parsed(self):
        cfg = self._load("load")
        self.assertIn(0, cfg.mediawiki.namespaces)
        self.assertIn(4, cfg.mediawiki.namespaces)

    def test_collection_name(self):
        cfg = self._load("load")
        self.assertEqual("test_col", cfg.collection_name)

    def test_index_vendor_defaults_to_qdrant(self):
        cfg = self._load("load")
        self.assertEqual("qdrant", cfg.index_vendor)

    def test_embedding_max_retries_defaults_to_three(self):
        cfg = self._load("index")
        self.assertEqual(3, cfg.embedding_max_retries)

    def test_embedding_max_retries_from_env(self):
        cfg = self._load("index", {"EMBEDDING_MAX_RETRIES": "9"})
        self.assertEqual(9, cfg.embedding_max_retries)

    def test_embedding_batch_size_defaults_to_eight(self):
        cfg = self._load("index")
        self.assertEqual(8, cfg.embedding_batch_size)

    def test_embedding_batch_size_from_env(self):
        cfg = self._load("index", {"EMBEDDING_BATCH_SIZE": "32"})
        self.assertEqual(32, cfg.embedding_batch_size)

    def test_chunking_defaults(self):
        cfg = self._load("index")
        self.assertEqual("none", cfg.chunking.strategy)
        self.assertEqual(3000, cfg.chunking.max_bytes)
        self.assertEqual(300, cfg.chunking.overlap_bytes)

    def test_chunking_from_env(self):
        cfg = self._load("index", extra_env={
            "INDEX_CHUNK_STRATEGY": "paragraph",
            "INDEX_CHUNK_MAX_BYTES": "1500",
            "INDEX_CHUNK_OVERLAP_BYTES": "100",
        })
        self.assertEqual("paragraph", cfg.chunking.strategy)
        self.assertEqual(1500, cfg.chunking.max_bytes)
        self.assertEqual(100, cfg.chunking.overlap_bytes)

    def test_chunking_from_yaml_overrides_env(self):
        yaml_content = "index:\n  chunking:\n    strategy: fixed\n    max_bytes: 2000\n    overlap_bytes: 50\n"
        config_path = _write_yaml(yaml_content)
        try:
            env = {**_MINIMAL_ENV_INDEX, "INDEX_CHUNK_STRATEGY": "paragraph"}
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
                cfg = load_config(command="index", config_path=config_path)
            self.assertEqual("fixed", cfg.chunking.strategy)
            self.assertEqual(2000, cfg.chunking.max_bytes)
            self.assertEqual(50, cfg.chunking.overlap_bytes)
        finally:
            config_path.unlink()

    def test_chunking_invalid_strategy_exits(self):
        with self.assertRaises(SystemExit):
            self._load("index", extra_env={"INDEX_CHUNK_STRATEGY": "bogus"})

    def test_chunking_max_bytes_over_storage_limit_exits(self):
        with self.assertRaises(SystemExit):
            self._load("index", extra_env={"INDEX_CHUNK_MAX_BYTES": "6000"})

    def test_chunking_overlap_not_smaller_than_max_exits(self):
        with self.assertRaises(SystemExit):
            self._load("index", extra_env={
                "INDEX_CHUNK_MAX_BYTES": "1000",
                "INDEX_CHUNK_OVERLAP_BYTES": "1000",
            })

    def test_rate_limiting_defaults_to_true(self):
        cfg = self._load("load")
        self.assertTrue(cfg.loader.rate_limiting)

    def test_rate_limiting_false(self):
        cfg = self._load("load", extra_env={"ENABLE_RATE_LIMITING": "false"})
        self.assertFalse(cfg.loader.rate_limiting)

    def test_hyde_defaults(self):
        cfg = self._load("search")
        self.assertFalse(cfg.search.hyde_enabled)
        self.assertEqual(1, cfg.search.hyde_passages)

    def test_hyde_enabled_from_env(self):
        cfg = self._load("search", extra_env={"SEARCH_HYDE_ENABLED": "true"})
        self.assertTrue(cfg.search.hyde_enabled)

    def test_search_defaults(self):
        cfg = self._load("search")
        self.assertAlmostEqual(0.6, cfg.search.distance_cutoff)
        self.assertEqual(1536, cfg.search.max_completion_tokens)
        self.assertAlmostEqual(0.05, cfg.search.temperature)
        self.assertAlmostEqual(0.85, cfg.search.top_p)

    def test_openai_generation_params_from_yaml(self):
        yaml_content = "openai:\n  max_completion_tokens: 512\n  temperature: 0.1\n  top_p: 0.9\n"
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_SEARCH, clear=True):
                cfg = load_config(command="search", config_path=config_path)
            self.assertEqual(512, cfg.search.max_completion_tokens)
            self.assertAlmostEqual(0.1, cfg.search.temperature)
            self.assertAlmostEqual(0.9, cfg.search.top_p)
        finally:
            config_path.unlink()

    def test_wrapper_chat_defaults(self):
        cfg = self._load("server")
        self.assertEqual(0, cfg.wrapper.chat_max_turns)
        self.assertEqual(0, cfg.wrapper.chat_max_tokens)

    def test_singleton_set(self):
        self._load("load")
        self.assertIsNotNone(config_module.cfg)

    def test_log_level_defaults_to_info(self):
        cfg = self._load("load")
        self.assertEqual("INFO", cfg.log_level)

    def test_auth_tokens_empty_when_not_set(self):
        cfg = self._load("load")
        self.assertEqual([], cfg.auth_tokens)

    def test_auth_tokens_parsed(self):
        cfg = self._load("server", extra_env={"AUTH_TOKENS": "tok1,tok2"})
        self.assertIn("tok1", cfg.auth_tokens)
        self.assertIn("tok2", cfg.auth_tokens)


class TestLoadConfigMissingRequired(unittest.TestCase):
    """Verify that missing required fields exit with SystemExit."""

    def setUp(self):
        config_module.cfg = None

    def _load_bad(self, command: str, env: dict) -> None:
        """Call load_config expecting failure, without loading the real .env file."""
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            load_config(command=command, config_path=Path("/nonexistent/config.yml"))

    def test_missing_mediawiki_url_exits(self):
        env = {k: v for k, v in _MINIMAL_ENV_ALL.items() if k != "MEDIAWIKI_URL"}
        with self.assertRaises(SystemExit):
            self._load_bad("load", env)

    def test_missing_collection_name_exits(self):
        env = {k: v for k, v in _MINIMAL_ENV_ALL.items() if k != "COLLECTION_NAME"}
        with self.assertRaises(SystemExit):
            self._load_bad("load", env)

    def test_missing_embedding_model_for_index_exits(self):
        env = {k: v for k, v in _MINIMAL_ENV_INDEX.items() if k != "EMBEDDING_MODEL"}
        with self.assertRaises(SystemExit):
            self._load_bad("index", env)

    def test_missing_llm_model_for_search_exits(self):
        env = {k: v for k, v in _MINIMAL_ENV_SEARCH.items() if k != "LLM_MODEL"}
        with self.assertRaises(SystemExit):
            self._load_bad("search", env)

    def test_missing_wrapper_api_base_for_server_exits(self):
        env = {k: v for k, v in _MINIMAL_ENV_SERVER.items() if k != "WRAPPER_API_BASE"}
        with self.assertRaises(SystemExit):
            self._load_bad("server", env)

    def test_missing_mcp_api_base_for_mcp_exits(self):
        env = {k: v for k, v in _MINIMAL_ENV_MCP.items() if k != "MCP_API_BASE"}
        with self.assertRaises(SystemExit):
            self._load_bad("mcp", env)

    def test_load_command_does_not_require_llm_model(self):
        """wr-load does not use the LLM, so LLM_MODEL is not required."""
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_ALL, clear=True):
            cfg = load_config(command="load", config_path=Path("/nonexistent/config.yml"))
        self.assertIsInstance(cfg, Config)

    def test_missing_openai_api_base_for_index_exits(self):
        """wr-index requires OPENAI_API_BASE (or openai.api_base in YAML)."""
        env = {k: v for k, v in _MINIMAL_ENV_INDEX.items() if k != "OPENAI_API_BASE"}
        with self.assertRaises(SystemExit):
            self._load_bad("index", env)


class TestLoadConfigYamlOverride(unittest.TestCase):
    """Verify that YAML values override env values."""

    def setUp(self):
        config_module.cfg = None

    def test_yaml_url_overrides_env(self):
        yaml_content = 'mediawiki:\n  url: "https://from-yaml.com"\n'
        config_path = _write_yaml(yaml_content)
        try:
            env = {**_MINIMAL_ENV_ALL, "MEDIAWIKI_URL": "https://from-env.com"}
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
                cfg = load_config(command="load", config_path=config_path)
            self.assertEqual("https://from-yaml.com", cfg.mediawiki.url)
        finally:
            config_path.unlink()

    def test_yaml_collection_name_overrides_env(self):
        yaml_content = 'collection:\n  name: "yaml_col"\n'
        config_path = _write_yaml(yaml_content)
        try:
            env = {**_MINIMAL_ENV_ALL, "COLLECTION_NAME": "env_col"}
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
                cfg = load_config(command="load", config_path=config_path)
            self.assertEqual("yaml_col", cfg.collection_name)
        finally:
            config_path.unlink()

    def test_yaml_rate_limiting_false(self):
        yaml_content = "loader:\n  rate_limiting: false\n"
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_ALL, clear=True):
                cfg = load_config(command="load", config_path=config_path)
            self.assertFalse(cfg.loader.rate_limiting)
        finally:
            config_path.unlink()

    def test_yaml_namespaces_as_list(self):
        yaml_content = "mediawiki:\n  namespaces: [0, 4, 12]\n"
        config_path = _write_yaml(yaml_content)
        try:
            env = {**_MINIMAL_ENV_ALL, "MEDIAWIKI_NAMESPACES": "0"}
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
                cfg = load_config(command="load", config_path=config_path)
            self.assertIn(12, cfg.mediawiki.namespaces)
        finally:
            config_path.unlink()

    def test_yaml_search_distance_cutoff(self):
        yaml_content = "search:\n  distance_cutoff: 0.75\n"
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_SEARCH, clear=True):
                cfg = load_config(command="search", config_path=config_path)
            self.assertAlmostEqual(0.75, cfg.search.distance_cutoff)
        finally:
            config_path.unlink()

    def test_yaml_observability_endpoint(self):
        yaml_content = (
            "observability:\n"
            "  langsmith:\n"
            '    endpoint: "https://eu.api.smith.langchain.com"\n'
        )
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_SEARCH, clear=True):
                cfg = load_config(command="search", config_path=config_path)
            self.assertEqual("https://eu.api.smith.langchain.com", cfg.langsmith.endpoint)
        finally:
            config_path.unlink()

    def test_yaml_sites_list_multi_site(self):
        """sites: list in YAML is loaded; cfg.mediawiki property returns sites[0]."""
        yaml_content = (
            "sites:\n"
            '  - url: "https://site0.example.com"\n'
            "    namespaces: [0, 4]\n"
            '  - url: "https://site1.example.com"\n'
            "    namespaces: [0]\n"
        )
        config_path = _write_yaml(yaml_content)
        try:
            # MEDIAWIKI_URL in env should be overridden by sites[0].url from YAML.
            env = {**_MINIMAL_ENV_ALL, "MEDIAWIKI_URL": "https://from-env.com"}
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
                cfg = load_config(command="load", config_path=config_path)
            self.assertEqual(2, len(cfg.sites))
            self.assertEqual("https://site0.example.com", cfg.sites[0].url)
            self.assertEqual("https://site1.example.com", cfg.sites[1].url)
            # Backward-compat property returns sites[0].
            self.assertEqual("https://site0.example.com", cfg.mediawiki.url)
        finally:
            config_path.unlink()

    def test_yaml_sites_single_entry_bc_property(self):
        """Single-entry sites: list is equivalent to old mediawiki: section."""
        yaml_content = (
            "sites:\n"
            '  - url: "https://single.example.com"\n'
            "    namespaces: [0]\n"
        )
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_ALL, clear=True):
                cfg = load_config(command="load", config_path=config_path)
            self.assertEqual(1, len(cfg.sites))
            self.assertEqual("https://single.example.com", cfg.mediawiki.url)
        finally:
            config_path.unlink()


class TestLoadConfigSecretsEnvOnly(unittest.TestCase):
    """Verify that secret fields are never read from YAML."""

    def setUp(self):
        config_module.cfg = None

    def test_openai_api_key_from_env_not_yaml(self):
        # The YAML contains a fake key; only the env key should be used.
        yaml_content = "OPENAI_API_KEY: yaml-key\n"
        config_path = _write_yaml(yaml_content)
        try:
            env = {**_MINIMAL_ENV_ALL, "OPENAI_API_KEY": "env-key"}  # pragma: allowlist secret
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
                cfg = load_config(command="load", config_path=config_path)
            self.assertEqual("env-key", cfg.openai_api_key)
        finally:
            config_path.unlink()

    def test_secret_absent_from_env_is_none(self):
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_ALL, clear=True):
            cfg = load_config(command="load", config_path=Path("/nonexistent/config.yml"))
        self.assertIsNone(cfg.openai_api_key)
        self.assertIsNone(cfg.langsmith_api_key)
        self.assertIsNone(cfg.langfuse_secret_key)
        self.assertIsNone(cfg.qdrant_api_key)
        # openai_api_base is now non-secret; absent for "load" command (not required).
        self.assertEqual("", cfg.openai_api_base)

    def test_qdrant_api_key_from_env(self):
        env = {**_MINIMAL_ENV_ALL, "QDRANT_API_KEY": "mykey"}  # pragma: allowlist secret
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            cfg = load_config(command="load", config_path=Path("/nonexistent/config.yml"))
        self.assertEqual("mykey", cfg.qdrant_api_key)

    def test_qdrant_connection_settings_from_env(self):
        env = {**_MINIMAL_ENV_ALL, "QDRANT_URL": "http://qdrant:6333", "QDRANT_TIMEOUT": "7.5"}
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            cfg = load_config(command="load", config_path=Path("/nonexistent/config.yml"))
        self.assertEqual("http://qdrant:6333", cfg.qdrant.url)
        self.assertEqual(7.5, cfg.qdrant.timeout)

    def test_qdrant_api_key_never_read_from_yaml(self):
        config_path = Path(tempfile.mkstemp(suffix=".yml")[1])
        config_path.write_text(
            "qdrant:\n"
            '  url: "http://from-yaml:6333"\n'
            '  api_key: "leaked"\n'
            'QDRANT_API_KEY: "leaked"\n'
        )
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_ALL, clear=True):
                cfg = load_config(command="load", config_path=config_path)
        finally:
            config_path.unlink()
        self.assertEqual("http://from-yaml:6333", cfg.qdrant.url)
        self.assertIsNone(cfg.qdrant_api_key)


class TestLoadConfigObservabilityValidation(unittest.TestCase):
    """Verify conditional observability validation."""

    def setUp(self):
        config_module.cfg = None

    def test_langsmith_tracing_without_key_exits(self):
        env = {**_MINIMAL_ENV_SEARCH, "LANGSMITH_TRACING": "true", "LANGSMITH_ENDPOINT": "https://x"}
        # Missing LANGSMITH_API_KEY.
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit):
                load_config(command="search", config_path=Path("/nonexistent/config.yml"))

    def test_langfuse_tracing_without_public_key_exits(self):
        env = {
            **_MINIMAL_ENV_SEARCH,
            "LANGFUSE_TRACING": "true",
            "LANGFUSE_HOST": "https://x",
            "LANGFUSE_SECRET_KEY": "s",
            # Missing LANGFUSE_PUBLIC_KEY.
        }
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit):
                load_config(command="search", config_path=Path("/nonexistent/config.yml"))

    def test_langsmith_tracing_not_checked_for_load_command(self):
        """wr-load doesn't use LLM or observability, so missing keys are not required."""
        env = {**_MINIMAL_ENV_ALL, "LANGSMITH_TRACING": "true"}
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            cfg = load_config(command="load", config_path=Path("/nonexistent/config.yml"))
        self.assertIsInstance(cfg, Config)


class TestLoadConfigOsEnvironPropagation(unittest.TestCase):
    """Verify that os.environ is updated for third-party SDK consumption."""

    def setUp(self):
        config_module.cfg = None

    def tearDown(self):
        # Clean up any keys set by load_config.
        for key in ("LANGSMITH_PROJECT", "LANGSMITH_PROMPT_PREFIX"):
            os.environ.pop(key, None)

    def test_openai_api_key_stored_in_config(self):
        """OPENAI_API_KEY is stored in cfg but no longer written to os.environ."""
        env = {**_MINIMAL_ENV_ALL, "OPENAI_API_KEY": "test-key"}  # pragma: allowlist secret
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            cfg = load_config(command="load", config_path=Path("/nonexistent/config.yml"))
        self.assertEqual("test-key", cfg.openai_api_key)

    def test_langsmith_project_set_when_tracing_enabled(self):
        env = {
            **_MINIMAL_ENV_SEARCH,
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_ENDPOINT": "https://x",
            "LANGSMITH_API_KEY": "k",
        }
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            cfg = load_config(command="search", config_path=Path("/nonexistent/config.yml"))
            # Check inside the with block before patch.dict restores os.environ.
            self.assertEqual(cfg.collection_name, os.environ.get("LANGSMITH_PROJECT"))


class TestLoadConfigApiBaseFields(unittest.TestCase):
    """Verify per-model api_base fields and their fallback resolution."""

    def setUp(self):
        config_module.cfg = None

    def _load(self, command: str, extra_env: dict | None = None) -> Config:
        """Load config with a clean env, without loading the real .env file."""
        base = _MINIMAL_ENV_SERVER if command == "server" else (
            _MINIMAL_ENV_MCP if command == "mcp" else (
                _MINIMAL_ENV_SEARCH if command in {"search"} else (
                    _MINIMAL_ENV_INDEX if command == "index" else _MINIMAL_ENV_ALL
                )
            )
        )
        env = {**base, **(extra_env or {})}
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            return load_config(command=command, config_path=Path("/nonexistent/config.yml"))

    def test_openai_api_base_set_from_env(self):
        cfg = self._load("index")
        self.assertEqual("https://api.example.com/v1", cfg.openai_api_base)

    def test_per_model_api_base_defaults_to_none(self):
        cfg = self._load("index")
        self.assertIsNone(cfg.embedding_api_base)
        self.assertIsNone(cfg.contextualisation_api_base)
        self.assertIsNone(cfg.hyde_api_base)

    def test_embedding_api_base_from_yaml(self):
        yaml_content = 'embedding:\n  api_base: "https://embed.example.com/v1"\n'
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_INDEX, clear=True):
                cfg = load_config(command="index", config_path=config_path)
            self.assertEqual("https://embed.example.com/v1", cfg.embedding_api_base)
        finally:
            config_path.unlink()

    def test_contextualisation_api_base_from_yaml(self):
        yaml_content = 'search:\n  contextualisation:\n    api_base: "https://ctx.example.com/v1"\n'
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_SEARCH, clear=True):
                cfg = load_config(command="search", config_path=config_path)
            self.assertEqual("https://ctx.example.com/v1", cfg.contextualisation_api_base)
        finally:
            config_path.unlink()

    def test_hyde_api_base_from_yaml(self):
        yaml_content = 'search:\n  hyde:\n    api_base: "https://hyde.example.com/v1"\n'
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_SEARCH, clear=True):
                cfg = load_config(command="search", config_path=config_path)
            self.assertEqual("https://hyde.example.com/v1", cfg.hyde_api_base)
        finally:
            config_path.unlink()

    def test_openai_api_base_from_yaml(self):
        yaml_content = 'openai:\n  api_base: "https://yaml-api.example.com/v1"\n'
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_INDEX, clear=True):
                cfg = load_config(command="index", config_path=config_path)
            self.assertEqual("https://yaml-api.example.com/v1", cfg.openai_api_base)
        finally:
            config_path.unlink()

    def test_openai_api_base_not_required_for_load(self):
        """wr-load does not use embeddings, so OPENAI_API_BASE is not required."""
        with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_ALL, clear=True):
            cfg = load_config(command="load", config_path=Path("/nonexistent/config.yml"))
        self.assertEqual("", cfg.openai_api_base)

    def test_hyde_model_defaults_to_none(self):
        cfg = self._load("search")
        self.assertIsNone(cfg.hyde_model)

    def test_hyde_model_from_yaml(self):
        yaml_content = 'search:\n  hyde:\n    model: "hyde-specific-model"\n'
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_SEARCH, clear=True):
                cfg = load_config(command="search", config_path=config_path)
            self.assertEqual("hyde-specific-model", cfg.hyde_model)
        finally:
            config_path.unlink()

    def test_per_model_api_key_defaults_to_none(self):
        cfg = self._load("index")
        self.assertIsNone(cfg.embedding_api_key)
        self.assertIsNone(cfg.contextualisation_api_key)
        self.assertIsNone(cfg.hyde_api_key)

    def test_embedding_api_key_from_env(self):
        cfg = self._load("index", extra_env={"EMBEDDING_API_KEY": "embed-key"})  # pragma: allowlist secret
        self.assertEqual("embed-key", cfg.embedding_api_key)

    def test_contextualisation_api_key_from_env(self):
        cfg = self._load("search", extra_env={
            "CONTEXTUALISATION_API_KEY": "ctx-key"  # pragma: allowlist secret
        })
        self.assertEqual("ctx-key", cfg.contextualisation_api_key)

    def test_hyde_api_key_from_env(self):
        cfg = self._load("search", extra_env={"HYDE_API_KEY": "hyde-key"})  # pragma: allowlist secret
        self.assertEqual("hyde-key", cfg.hyde_api_key)

    def test_per_model_keys_not_readable_from_yaml(self):
        """Per-model API keys must never be read from YAML (all are in _SECRETS)."""
        from wiki_rag.config import _SECRETS
        for key in ("EMBEDDING_API_KEY", "CONTEXTUALISATION_API_KEY", "HYDE_API_KEY"):
            self.assertIn(key, _SECRETS)

    def test_openai_model_from_env_via_llm_model(self):
        """LLM_MODEL env var still populates cfg.openai_model (backward compatibility)."""
        cfg = self._load("search")
        self.assertEqual("gpt-4o", cfg.openai_model)

    def test_openai_model_from_yaml(self):
        yaml_content = 'openai:\n  model: "gpt-4-turbo"\n'
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_SEARCH, clear=True):
                cfg = load_config(command="search", config_path=config_path)
            # YAML overrides env var.
            self.assertEqual("gpt-4-turbo", cfg.openai_model)
        finally:
            config_path.unlink()

    def test_contextualisation_model_from_new_yaml_path(self):
        yaml_content = 'search:\n  contextualisation:\n    model: "ctx-model"\n'
        config_path = _write_yaml(yaml_content)
        try:
            with patch("wiki_rag.config.load_dotenv"), patch.dict(os.environ, _MINIMAL_ENV_SEARCH, clear=True):
                cfg = load_config(command="search", config_path=config_path)
            self.assertEqual("ctx-model", cfg.contextualisation_model)
        finally:
            config_path.unlink()


if __name__ == "__main__":
    unittest.main()
