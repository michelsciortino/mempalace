#!/usr/bin/env python3
"""
TurboQuant vs Baseline Performance Benchmark
=============================================

Measures four dimensions of TurboQuant's impact on MemPalace:

  1. Compression fidelity   — cosine similarity of compressed vs original vectors
  2. Retrieval quality      — % of top-k results that agree between baseline and TQ
  3. Indexing overhead      — extra time TQ compression adds per vector
  4. Query latency          — search time baseline vs TQ (median + p95)
  5. Memory savings         — theoretical reduction in vector storage bytes

Usage:
    python benchmarks/turboquant_bench.py
    python benchmarks/turboquant_bench.py --corpus-size 500 --n-queries 100 --bits 4
    python benchmarks/turboquant_bench.py --bits 3 --bits 4 --top-k 5
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from mempalace.turboquant_embeddings import TurboQuantEmbeddingFunction

# ── Synthetic corpus ──────────────────────────────────────────────────────────

TEMPLATES = [
    "The authentication module uses {tech} for session management with {duration} expiry.",
    "Database migrations are handled by {tool}. We use {db} with connection pooling.",
    "The {layer} uses {framework} for {purpose} management across all API calls.",
    "Sprint planning: migrate {feature} to {target} by {quarter}.",
    "Discovered that {component} causes {issue} under high load. Fixed by {solution}.",
    "Decision: use {choice} over {alternative} because of {reason}.",
    "User {name} prefers {preference} when working on {domain} tasks.",
    "The {service} scales horizontally. Deployed on {platform} with {count} replicas.",
    "Regression in {module}: {description}. Root cause was {cause}. Fixed in {version}.",
    "Performance: {operation} takes {time}ms at p99. Target is under {target}ms.",
]

WORDS = {
    "tech": ["JWT tokens", "OAuth2", "session cookies", "API keys", "passkeys"],
    "duration": ["24 hours", "7 days", "30 minutes", "1 hour", "never"],
    "tool": ["Alembic", "Flyway", "Liquibase", "Django migrations", "Prisma"],
    "db": ["PostgreSQL 15", "MySQL 8", "SQLite", "CockroachDB", "Supabase"],
    "layer": ["frontend", "backend", "API gateway", "worker", "cron job"],
    "framework": ["TanStack Query", "SWR", "React Query", "Apollo", "RTK Query"],
    "purpose": ["server state", "cache", "real-time data", "pagination", "prefetch"],
    "feature": ["auth", "billing", "search", "notifications", "analytics"],
    "target": ["passkeys", "WebAuthn", "SSO", "OAuth2", "magic links"],
    "quarter": ["Q1", "Q2", "Q3", "Q4", "end of year"],
    "component": ["Redis cache", "task queue", "WebSocket server", "CDN", "load balancer"],
    "issue": ["memory leaks", "race conditions", "deadlocks", "timeouts", "OOM errors"],
    "solution": ["connection pooling", "backpressure", "circuit breakers", "retries", "caching"],
    "choice": ["GraphQL", "REST", "gRPC", "tRPC", "WebSockets"],
    "alternative": ["REST", "GraphQL", "REST", "REST", "polling"],
    "reason": ["type safety", "performance", "simplicity", "ecosystem", "team familiarity"],
    "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "preference": ["dark mode", "vim keybindings", "monospace fonts", "split panes", "tabs"],
    "domain": ["backend", "frontend", "devops", "ML", "mobile"],
    "service": ["API server", "worker fleet", "scheduler", "ingestion pipeline", "export job"],
    "platform": ["AWS ECS", "GKE", "Fly.io", "Railway", "Render"],
    "count": ["3", "5", "10", "20", "50"],
    "module": ["auth", "payments", "search", "export", "notifications"],
    "description": ["500 errors on login", "slow queries", "missing emails", "broken exports"],
    "cause": ["missing index", "N+1 query", "wrong timezone", "race condition", "typo in regex"],
    "version": ["v2.3.1", "v1.9.4", "v3.0.0-rc1", "v2.1.0", "v4.2.1"],
    "operation": ["full-text search", "vector query", "aggregation", "export", "batch insert"],
    "time": ["42", "120", "8", "350", "15"],
    "target": ["100", "200", "50", "500", "30"],
}


def make_corpus(n: int, seed: int = 42) -> List[str]:
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        template = TEMPLATES[i % len(TEMPLATES)]
        filled = template
        for key, choices in WORDS.items():
            if "{" + key + "}" in filled:
                filled = filled.replace("{" + key + "}", rng.choice(choices), 1)
        docs.append(filled)
    return docs


# ── Metrics helpers ───────────────────────────────────────────────────────────

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of a and b."""
    a_n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b_n = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return np.sum(a_n * b_n, axis=1)


def top_k_ids(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(scores)[::-1][:k]


def recall_at_k(baseline_vecs, tq_vecs, queries_baseline, queries_tq, k: int) -> float:
    """
    Fraction of queries where TurboQuant's top-k overlaps with baseline's top-k.
    """
    hits = 0
    n_queries = len(queries_baseline)
    n_corpus = len(baseline_vecs)

    for i in range(n_queries):
        q_base = queries_baseline[i : i + 1]
        q_tq = queries_tq[i : i + 1]

        scores_base = cosine_sim_matrix(
            np.tile(q_base, (n_corpus, 1)), baseline_vecs
        )
        scores_tq = cosine_sim_matrix(
            np.tile(q_tq, (n_corpus, 1)), tq_vecs
        )

        top_base = set(top_k_ids(scores_base, k))
        top_tq = set(top_k_ids(scores_tq, k))
        hits += len(top_base & top_tq) / k

    return hits / n_queries


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(corpus_size: int, n_queries: int, bits_list: List[int], top_k: int):
    print(f"\n{'=' * 65}")
    print("  TurboQuant vs Baseline — MemPalace Performance Benchmark")
    print(f"{'=' * 65}")
    print(f"  Corpus size : {corpus_size:,} documents")
    print(f"  Queries     : {n_queries}")
    print(f"  Top-k       : {top_k}")
    print(f"  Bits tested : {bits_list}")
    print()

    # ── Generate corpus and queries ───────────────────────────────────────────
    print("  Generating synthetic corpus...")
    corpus = make_corpus(corpus_size)
    queries = make_corpus(n_queries, seed=99)

    # ── Baseline: get raw embeddings ──────────────────────────────────────────
    print("  Embedding corpus (baseline)...")
    base_ef = DefaultEmbeddingFunction()

    t0 = time.perf_counter()
    raw_embeddings = np.array(base_ef(corpus), dtype=np.float32)
    baseline_index_time = time.perf_counter() - t0

    dim = raw_embeddings.shape[1]
    print(f"  Embedding dim : {dim}")
    print(f"  Baseline index time: {baseline_index_time:.2f}s "
          f"({baseline_index_time / corpus_size * 1000:.1f}ms/doc)")

    # Baseline query embeddings
    raw_queries = np.array(base_ef(queries), dtype=np.float32)

    # Baseline query latency
    latencies_base = []
    for i in range(n_queries):
        q = raw_queries[i : i + 1]
        t0 = time.perf_counter()
        scores = cosine_sim_matrix(np.tile(q, (corpus_size, 1)), raw_embeddings)
        _ = top_k_ids(scores, top_k)
        latencies_base.append((time.perf_counter() - t0) * 1000)

    # Memory baseline
    mem_baseline_mb = corpus_size * dim * 4 / 1_000_000

    print(f"\n  {'Metric':<35} {'Baseline':>12}")
    print(f"  {'-' * 49}")
    print(f"  {'Vector memory (MB)':<35} {mem_baseline_mb:>11.1f}")
    print(f"  {'Index time (s)':<35} {baseline_index_time:>11.2f}")
    print(f"  {'Query latency median (ms)':<35} {np.median(latencies_base):>11.2f}")
    print(f"  {'Query latency p95 (ms)':<35} {np.percentile(latencies_base, 95):>11.2f}")

    # ── TurboQuant variants ───────────────────────────────────────────────────
    # Warm up torch/TurboQuant once before timing any variant — the first call
    # loads native libraries which can take 30-90s and is a one-time startup
    # cost, not per-document overhead.
    print("  Warming up TurboQuant (one-time torch init)...", end=" ", flush=True)
    _warmup_ef = TurboQuantEmbeddingFunction(bits=bits_list[0])
    _warmup_ef.compress_vectors(raw_embeddings[:2])
    print("done")

    for bits in bits_list:
        print(f"\n  {'─' * 65}")
        print(f"  TurboQuant {bits}-bit")
        print(f"  {'─' * 65}")

        with tempfile.TemporaryDirectory() as tmp:
            params_path = os.path.join(tmp, "tq_params.json")
            ef = TurboQuantEmbeddingFunction(bits=bits, params_path=params_path)

            # Compress corpus vectors
            t0 = time.perf_counter()
            tq_embeddings = ef.compress_vectors(raw_embeddings)
            tq_compress_time = time.perf_counter() - t0
            total_index_time = baseline_index_time + tq_compress_time

            # Compress query vectors (same quantizer, loaded from params)
            ef2 = TurboQuantEmbeddingFunction(bits=bits, params_path=params_path)
            tq_queries = ef2.compress_vectors(raw_queries)

            # Compression fidelity: cosine sim of compressed vs original
            fidelity_scores = cosine_sim_matrix(raw_embeddings, tq_embeddings)
            mean_fidelity = float(np.mean(fidelity_scores))
            min_fidelity = float(np.min(fidelity_scores))

            # Retrieval quality: top-k agreement rate
            recall = recall_at_k(raw_embeddings, tq_embeddings, raw_queries, tq_queries, top_k)

            # Query latency with TQ
            latencies_tq = []
            for i in range(n_queries):
                q = tq_queries[i : i + 1]
                t0 = time.perf_counter()
                scores = cosine_sim_matrix(
                    np.tile(q, (corpus_size, 1)), tq_embeddings
                )
                _ = top_k_ids(scores, top_k)
                latencies_tq.append((time.perf_counter() - t0) * 1000)

            # Memory (theoretical bit-packed)
            mem_tq_mb = corpus_size * dim * bits / 8 / 1_000_000
            mem_reduction = mem_baseline_mb / mem_tq_mb

        print(f"\n  {'Metric':<35} {'Baseline':>12} {'TQ ' + str(bits) + '-bit':>12} {'Delta':>10}")
        print(f"  {'-' * 71}")

        def row(label, base_val, tq_val, fmt=".2f", suffix="", higher_is_better=True):
            delta = tq_val - base_val
            sign = "+" if delta >= 0 else ""
            arrow = "▲" if (delta >= 0) == higher_is_better else "▼"
            print(f"  {label:<35} {base_val:>11{fmt}}{suffix} "
                  f"{tq_val:>11{fmt}}{suffix} "
                  f"{arrow} {sign}{delta:.2f}{suffix}")

        row("Compression fidelity (cosine)", 1.0, mean_fidelity, higher_is_better=True)
        print(f"  {'  min fidelity':<35} {'1.000':>12} {min_fidelity:>11.3f}")
        row(f"Retrieval recall@{top_k}", 1.0, recall, higher_is_better=True)
        print(f"  {'Index time (s)':<35} {baseline_index_time:>11.2f}  "
              f"{total_index_time:>10.2f}  "
              f"▼ +{tq_compress_time:.2f}s overhead")
        row("Query latency median (ms)", np.median(latencies_base),
            np.median(latencies_tq), higher_is_better=False)
        row("Query latency p95 (ms)", np.percentile(latencies_base, 95),
            np.percentile(latencies_tq, 95), higher_is_better=False)
        print(f"  {'Vector memory — theoretical (MB)':<35} {mem_baseline_mb:>11.1f}  "
              f"{mem_tq_mb:>10.1f}  ▲ {mem_reduction:.1f}x smaller")

    print(f"\n{'=' * 65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant vs Baseline performance benchmark for MemPalace"
    )
    parser.add_argument(
        "--corpus-size", type=int, default=200,
        help="Number of documents to index (default: 200)",
    )
    parser.add_argument(
        "--n-queries", type=int, default=50,
        help="Number of search queries to benchmark (default: 50)",
    )
    parser.add_argument(
        "--bits", type=int, action="append", dest="bits_list",
        help="Bits per dimension to test (repeat for multiple; default: 3 4)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Top-k for recall measurement (default: 5)",
    )
    args = parser.parse_args()

    bits_list = args.bits_list or [3, 4]
    run_benchmark(
        corpus_size=args.corpus_size,
        n_queries=args.n_queries,
        bits_list=bits_list,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
