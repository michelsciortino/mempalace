"""
test_turboquant_embeddings.py — Tests for TurboQuant embedding compression.

Covers:
  - Round-trip fidelity: quantize → dequantize preserves cosine similarity
  - Embedding function interface: returns correct shape and type
  - Seed persistence: same seed produces identical output
  - Search consistency: top-1 result matches before and after compression
  - Config integration: use_turboquant / turboquant_bits properties
  - compress-vectors flow: existing embeddings can be migrated
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip(
    "turboquant"
)  # skip entire file on Python <3.10 where turboquant is unavailable

from mempalace.config import MempalaceConfig  # noqa: E402
from mempalace.turboquant_embeddings import TurboQuantEmbeddingFunction, build_embedding_function  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────────


def make_random_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    # Normalize to unit sphere (typical for sentence embeddings)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-9)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) matrices."""
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return np.sum(a_norm * b_norm, axis=1)


# ── TurboQuantEmbeddingFunction unit tests ──────────────────────────────────


class TestCompressVectors:
    """Tests for the core compress_vectors() method."""

    def test_output_shape_preserved(self):
        ef = TurboQuantEmbeddingFunction(bits=4)
        vecs = make_random_vectors(16, 128)
        out = ef.compress_vectors(vecs)
        assert out.shape == vecs.shape

    def test_output_dtype_float32(self):
        ef = TurboQuantEmbeddingFunction(bits=4)
        vecs = make_random_vectors(8, 64)
        out = ef.compress_vectors(vecs)
        assert out.dtype == np.float32

    def test_cosine_similarity_preserved(self):
        """After compression, cosine similarity to original should be high (>0.9)."""
        ef = TurboQuantEmbeddingFunction(bits=4)
        vecs = make_random_vectors(32, 256)
        compressed = ef.compress_vectors(vecs)
        sims = cosine_sim(vecs, compressed)
        mean_sim = float(np.mean(sims))
        assert mean_sim > 0.90, f"Mean cosine sim too low: {mean_sim:.3f}"

    def test_higher_bits_better_fidelity(self):
        """4-bit compression should be more faithful than 2-bit."""
        vecs = make_random_vectors(32, 256, seed=7)
        ef2 = TurboQuantEmbeddingFunction(bits=2)
        ef4 = TurboQuantEmbeddingFunction(bits=4)
        sim2 = float(np.mean(cosine_sim(vecs, ef2.compress_vectors(vecs))))
        sim4 = float(np.mean(cosine_sim(vecs, ef4.compress_vectors(vecs))))
        assert sim4 > sim2, f"4-bit ({sim4:.3f}) should beat 2-bit ({sim2:.3f})"

    def test_different_vectors_different_output(self):
        """Distinct input vectors should produce distinct outputs."""
        ef = TurboQuantEmbeddingFunction(bits=4)
        v1 = make_random_vectors(4, 64, seed=0)
        v2 = make_random_vectors(4, 64, seed=1)
        out1 = ef.compress_vectors(v1)
        out2 = ef.compress_vectors(v2)
        assert not np.allclose(out1, out2)


class TestSeedPersistence:
    """Tests for rotation seed save/load via params_path."""

    def test_same_seed_same_output(self, tmp_path):
        params = str(tmp_path / "tq_params.json")
        vecs = make_random_vectors(8, 64)

        ef1 = TurboQuantEmbeddingFunction(bits=4, params_path=params)
        out1 = ef1.compress_vectors(vecs)

        # Reinstantiate — should reload the same seed
        ef2 = TurboQuantEmbeddingFunction(bits=4, params_path=params)
        out2 = ef2.compress_vectors(vecs)

        np.testing.assert_array_equal(out1, out2)

    def test_params_file_written(self, tmp_path):
        params = str(tmp_path / "tq_params.json")
        ef = TurboQuantEmbeddingFunction(bits=4, params_path=params)
        ef.compress_vectors(make_random_vectors(4, 64))

        assert os.path.exists(params)
        with open(params) as f:
            data = json.load(f)
        assert "seed" in data
        assert data["bits"] == 4
        assert data["dim"] == 64

    def test_different_paths_independent_seeds(self, tmp_path):
        p1 = str(tmp_path / "a.json")
        p2 = str(tmp_path / "b.json")
        vecs = make_random_vectors(8, 64)

        ef1 = TurboQuantEmbeddingFunction(bits=4, params_path=p1)
        ef2 = TurboQuantEmbeddingFunction(bits=4, params_path=p2)

        out1 = ef1.compress_vectors(vecs)
        out2 = ef2.compress_vectors(vecs)

        # Two independently seeded quantizers likely produce different output
        # (very low probability of same seed by chance)
        with open(p1) as f:
            seed1 = json.load(f)["seed"]
        with open(p2) as f:
            seed2 = json.load(f)["seed"]

        if seed1 != seed2:
            assert not np.allclose(out1, out2)


class TestEmbeddingFunctionInterface:
    """Tests for the ChromaDB EmbeddingFunction callable interface."""

    def test_callable_with_mock_base(self):
        """__call__ returns list of list of float, same length as input."""

        class MockBase:
            def __call__(self, texts):
                return [list(np.ones(64, dtype=np.float32)) for _ in texts]

        ef = TurboQuantEmbeddingFunction(bits=4, base_ef=MockBase())
        result = ef(["hello world", "foo bar"])

        assert len(result) == 2
        assert len(result[0]) == 64
        assert all(isinstance(x, float) for x in result[0])

    def test_single_text_input(self):
        class MockBase:
            def __call__(self, texts):
                return [list(np.random.randn(128).astype(np.float32)) for _ in texts]

        ef = TurboQuantEmbeddingFunction(bits=4, base_ef=MockBase())
        result = ef(["single sentence"])
        assert len(result) == 1
        assert len(result[0]) == 128

    def test_params_path_for_palace(self, tmp_path):
        palace = str(tmp_path / "palace")
        path = TurboQuantEmbeddingFunction.params_path_for_palace(palace)
        assert path == str(tmp_path / "palace" / "turboquant_params.json")


# ── Retrieval consistency test ───────────────────────────────────────────────


class TestRetrievalConsistency:
    """Top-k retrieval should stay consistent after TurboQuant compression."""

    def test_top1_match_preserved(self):
        """The nearest neighbour of a query should be the same before and after compression."""
        n, dim = 50, 256
        rng = np.random.default_rng(42)
        corpus = make_random_vectors(n, dim, seed=42)
        query = corpus[7:8] + rng.standard_normal((1, dim)).astype(np.float32) * 0.05

        # Find top-1 in raw space
        raw_sims = cosine_sim(np.tile(query, (n, 1)), corpus)
        raw_top1 = int(np.argmax(raw_sims))

        # Compress corpus and query with same ef
        ef = TurboQuantEmbeddingFunction(bits=4)
        compressed_corpus = ef.compress_vectors(corpus)
        compressed_query = ef.compress_vectors(query)

        comp_sims = cosine_sim(np.tile(compressed_query, (n, 1)), compressed_corpus)
        comp_top1 = int(np.argmax(comp_sims))

        assert raw_top1 == comp_top1, (
            f"Top-1 changed after compression: raw={raw_top1}, compressed={comp_top1}"
        )


# ── Config integration tests ─────────────────────────────────────────────────


class TestConfigIntegration:
    def test_use_turboquant_defaults_false(self, tmp_dir):
        cfg_dir = os.path.join(tmp_dir, "cfg")
        os.makedirs(cfg_dir)
        with open(os.path.join(cfg_dir, "config.json"), "w") as f:
            json.dump({}, f)
        cfg = MempalaceConfig(config_dir=cfg_dir)
        assert cfg.use_turboquant is False

    def test_use_turboquant_enabled(self, tmp_dir):
        cfg_dir = os.path.join(tmp_dir, "cfg")
        os.makedirs(cfg_dir)
        with open(os.path.join(cfg_dir, "config.json"), "w") as f:
            json.dump({"use_turboquant": True}, f)
        cfg = MempalaceConfig(config_dir=cfg_dir)
        assert cfg.use_turboquant is True

    def test_turboquant_bits_default(self, tmp_dir):
        cfg_dir = os.path.join(tmp_dir, "cfg")
        os.makedirs(cfg_dir)
        with open(os.path.join(cfg_dir, "config.json"), "w") as f:
            json.dump({}, f)
        cfg = MempalaceConfig(config_dir=cfg_dir)
        assert cfg.turboquant_bits == 4

    def test_turboquant_bits_custom(self, tmp_dir):
        cfg_dir = os.path.join(tmp_dir, "cfg")
        os.makedirs(cfg_dir)
        with open(os.path.join(cfg_dir, "config.json"), "w") as f:
            json.dump({"turboquant_bits": 3}, f)
        cfg = MempalaceConfig(config_dir=cfg_dir)
        assert cfg.turboquant_bits == 3

    def test_build_embedding_function_disabled(self, tmp_dir):
        cfg_dir = os.path.join(tmp_dir, "cfg")
        os.makedirs(cfg_dir)
        with open(os.path.join(cfg_dir, "config.json"), "w") as f:
            json.dump({"use_turboquant": False}, f)
        cfg = MempalaceConfig(config_dir=cfg_dir)
        palace = os.path.join(tmp_dir, "palace")
        result = build_embedding_function(cfg, palace)
        assert result is None

    def test_build_embedding_function_enabled(self, tmp_dir):
        cfg_dir = os.path.join(tmp_dir, "cfg")
        os.makedirs(cfg_dir)
        with open(os.path.join(cfg_dir, "config.json"), "w") as f:
            json.dump({"use_turboquant": True, "turboquant_bits": 4}, f)
        cfg = MempalaceConfig(config_dir=cfg_dir)
        palace = os.path.join(tmp_dir, "palace")
        ef = build_embedding_function(cfg, palace)
        assert isinstance(ef, TurboQuantEmbeddingFunction)
        assert ef.bits == 4

    def test_turboquant_rerank_defaults_true(self, tmp_dir):
        cfg_dir = os.path.join(tmp_dir, "cfg")
        os.makedirs(cfg_dir)
        with open(os.path.join(cfg_dir, "config.json"), "w") as f:
            json.dump({}, f)
        cfg = MempalaceConfig(config_dir=cfg_dir)
        assert cfg.turboquant_rerank is True

    def test_turboquant_rerank_factor_default(self, tmp_dir):
        cfg_dir = os.path.join(tmp_dir, "cfg")
        os.makedirs(cfg_dir)
        with open(os.path.join(cfg_dir, "config.json"), "w") as f:
            json.dump({}, f)
        cfg = MempalaceConfig(config_dir=cfg_dir)
        assert cfg.turboquant_rerank_factor == 2


# ── Rerank tests ─────────────────────────────────────────────────────────────


class TestRerank:
    """Tests for the over-fetch + exact-cosine rerank step in searcher.py."""

    def test_rerank_returns_n_results(self):
        """Output length must equal n_results, never the over-fetched count."""
        from mempalace.searcher import _rerank

        n, dim, n_results = 20, 64, 5
        docs = [f"doc_{i}" for i in range(n)]
        metas = [{"wing": "w", "room": "r"} for _ in range(n)]
        dists = list(np.linspace(0.1, 0.9, n))

        class MockBase:
            def __call__(self, texts):
                rng = np.random.default_rng(abs(hash(texts[0])) % 2**31)
                return [list(rng.standard_normal(dim).astype(np.float32)) for _ in texts]

        out_docs, out_metas, out_dists = _rerank("query text", docs, metas, dists, n_results)
        assert len(out_docs) == n_results
        assert len(out_metas) == n_results
        assert len(out_dists) == n_results

    def test_rerank_improves_or_matches_recall(self, tmp_path):
        """Recall@5 with rerank should be >= recall@5 without rerank."""
        import unittest.mock as mock
        from mempalace.searcher import _rerank

        n, dim, k = 50, 128, 5
        rng = np.random.default_rng(0)

        # Build a corpus of unit vectors
        corpus_np = rng.standard_normal((n, dim)).astype(np.float32)
        corpus_np /= np.linalg.norm(corpus_np, axis=1, keepdims=True)

        ef = TurboQuantEmbeddingFunction(bits=4, params_path=str(tmp_path / "p.json"))
        tq_corpus = ef.compress_vectors(corpus_np)

        n_queries = 10
        queries_np = corpus_np[:n_queries]
        tq_queries = ef.compress_vectors(queries_np)
        doc_texts = [f"doc_{i}" for i in range(n)]
        metas_list = [{"wing": "w", "room": "r"} for _ in range(n)]

        recalls_no_rr, recalls_rr = [], []

        for i in range(n_queries):
            q_raw = queries_np[i : i + 1]
            q_tq = tq_queries[i : i + 1]

            sims_base = cosine_sim(np.tile(q_raw, (n, 1)), corpus_np)
            sims_tq = cosine_sim(np.tile(q_tq, (n, 1)), tq_corpus)
            top_base = set(np.argsort(sims_base)[::-1][:k])

            # Recall without rerank
            top_tq = set(np.argsort(sims_tq)[::-1][:k])
            recalls_no_rr.append(len(top_base & top_tq) / k)

            # Over-fetch 2k candidates
            candidate_ids = np.argsort(sims_tq)[::-1][: k * 2]
            candidate_docs = [doc_texts[j] for j in candidate_ids]
            candidate_metas = [metas_list[j] for j in candidate_ids]
            candidate_dists = [float(1 - sims_tq[j]) for j in candidate_ids]

            # Mock DefaultEmbeddingFunction: query → q_raw, docs → their raw vectors
            def make_mock_ef(q_vec, c_ids, c_vecs):
                call_count = [0]

                def _call(texts):
                    call_count[0] += 1
                    if call_count[0] == 1:  # first call: query
                        return [list(q_vec.flatten())]
                    # second call: candidate docs
                    idx_map = {doc_texts[j]: j for j in c_ids}
                    return [list(c_vecs[idx_map[t]]) for t in texts]

                return _call

            mock_ef = make_mock_ef(q_raw, candidate_ids, corpus_np)
            with mock.patch(
                "chromadb.utils.embedding_functions.DefaultEmbeddingFunction",
                return_value=mock_ef,
            ):
                out_docs, _, _ = _rerank(
                    "any_query_text", candidate_docs, candidate_metas, candidate_dists, k
                )

            top_rr = {int(d.split("_")[1]) for d in out_docs}
            recalls_rr.append(len(top_base & top_rr) / k)

        mean_no_rr = float(np.mean(recalls_no_rr))
        mean_rr = float(np.mean(recalls_rr))
        assert mean_rr >= mean_no_rr, (
            f"Rerank recall ({mean_rr:.3f}) should be >= no-rerank ({mean_no_rr:.3f})"
        )
