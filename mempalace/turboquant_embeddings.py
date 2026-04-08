"""
turboquant_embeddings.py — TurboQuant-backed ChromaDB embedding function.

Applies TurboQuant inner-product-optimal quantization (ICLR 2026,
arxiv 2504.19874) to reduce vector memory footprint 6-8x while
preserving retrieval quality.

The core idea: embedding vectors are quantized to N bits per dimension
(vs 32-bit float), then dequantized back to float32 for storage in ChromaDB.
The dequantized approximations preserve cosine/inner-product similarity
with the originals, so search quality is maintained.

Usage (standalone):
    ef = TurboQuantEmbeddingFunction(
        bits=4,
        params_path="~/.mempalace/palace/turboquant_params.json",
    )
    col = client.get_collection("mempalace_drawers", embedding_function=ef)

Usage (via config):
    from mempalace.config import MempalaceConfig
    from mempalace.turboquant_embeddings import build_embedding_function
    ef = build_embedding_function(cfg, palace_path)  # None if TQ disabled
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

# np.trapz was removed in NumPy 2.0 (renamed to np.trapezoid).
# turboquant 0.2.0 still uses the old name — restore it as an alias.
# Must run before turboquant is imported (lazily inside _get_quantizer).
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

logger = logging.getLogger("mempalace_mcp")


class TurboQuantEmbeddingFunction:
    """
    ChromaDB-compatible embedding function backed by TurboQuant (ICLR 2026).

    Wraps any base embedding function (default: ChromaDB DefaultEmbeddingFunction)
    and applies TurboQuant inner-product-optimal quantization to compress vectors
    6-8x while preserving retrieval quality.

    The quantizer is seeded deterministically: the rotation seed is saved to a
    sidecar JSON file at params_path so that query vectors are always transformed
    consistently with the stored drawer vectors.

    Args:
        bits: Bits per dimension (1-4). Default: 4.
              Higher bits → better quality, less compression.
              4-bit gives ~8x memory reduction vs float32.
        params_path: Path to persist rotation parameters (seed, dim, bits).
                     Strongly recommended for persistent palaces — without it, a
                     fresh random seed is generated on every instantiation, making
                     stored and query vectors inconsistent.
        base_ef: Base embedding function for text → float32.
                 Defaults to ChromaDB's DefaultEmbeddingFunction.
    """

    def __init__(
        self,
        bits: int = 4,
        params_path: Optional[str] = None,
        base_ef=None,
    ):
        self.bits = bits
        self.params_path = Path(params_path).expanduser() if params_path else None
        self._params: dict = {}
        self._quantizer: Optional[Any] = None

        if self.params_path and self.params_path.exists():
            self._params = self._load_params()

        if base_ef is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._base_ef = DefaultEmbeddingFunction()
        else:
            self._base_ef = base_ef

    # ── Params persistence ────────────────────────────────────────────────────

    def _load_params(self) -> dict:
        try:
            with open(self.params_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_params(self) -> None:
        if self.params_path is None:
            return
        self.params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.params_path, "w") as f:
            json.dump(self._params, f, indent=2)

    # ── Quantizer lifecycle ───────────────────────────────────────────────────

    def _get_quantizer(self, dim: int) -> Any:
        """Lazy-init the TurboQuantIP quantizer; reuse across calls."""
        if self._quantizer is not None:
            return self._quantizer

        try:
            from turboquant import TurboQuantIP
        except ImportError as exc:
            raise ImportError(
                "turboquant>=0.2.0 (Python >=3.10) is required for TurboQuant compression. "
                "Install with: pip install 'mempalace[turboquant]'"
            ) from exc

        seed = self._params.get("seed")
        if seed is None:
            # Generate a new seed and persist it so future calls stay consistent
            seed = int.from_bytes(os.urandom(4), "big") % (2**31)
            self._params["seed"] = seed
            self._params["dim"] = dim
            self._params["bits"] = self.bits
            self._save_params()
            logger.info(
                "TurboQuant: initialized new quantizer (dim=%d, bits=%d, seed=%d)",
                dim,
                self.bits,
                seed,
            )

        self._quantizer = TurboQuantIP(dim=dim, bits=self.bits, device="cpu", seed=seed)
        return self._quantizer

    # ── Core compression ──────────────────────────────────────────────────────

    def compress_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Quantize then dequantize a batch of float32 embedding vectors.

        Applies TurboQuant's two-stage inner-product compression:
          Stage 1 (TurboQuant_MSE): rotate + scalar quantize to (bits-1) bits
          Stage 2 (QJL): quantize residual to 1 sign bit for unbiased inner products

        The returned dequantized float32 vectors preserve cosine similarity with
        the originals and can be stored directly in ChromaDB.

        Args:
            vectors: (N, dim) float32 numpy array of raw embeddings.

        Returns:
            (N, dim) float32 numpy array of dequantized approximations.
        """
        import torch  # lazy: installed alongside turboquant (Python >=3.10)

        dim = vectors.shape[1]
        quantizer = self._get_quantizer(dim)
        t = torch.from_numpy(vectors).float()
        mse_idx, norms, qjl_signs, res_norms = quantizer.quantize(t)
        restored = quantizer.dequantize(mse_idx, norms, qjl_signs, res_norms)
        return restored.numpy()

    # ── ChromaDB EmbeddingFunction interface ──────────────────────────────────

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed texts and return TurboQuant-compressed dequantized float32 vectors.

        This is the ChromaDB EmbeddingFunction interface. ChromaDB calls this
        method when inserting documents or querying by text.
        """
        raw = self._base_ef(input)
        vectors = np.array(raw, dtype=np.float32)
        compressed = self.compress_vectors(vectors)
        return compressed.tolist()

    # ── Convenience ───────────────────────────────────────────────────────────

    @staticmethod
    def params_path_for_palace(palace_path: str) -> str:
        """Return the canonical sidecar JSON path for a given palace directory."""
        return str(Path(palace_path) / "turboquant_params.json")


def build_embedding_function(cfg, palace_path: str) -> Optional[TurboQuantEmbeddingFunction]:
    """
    Return a TurboQuantEmbeddingFunction if TurboQuant is enabled in config,
    otherwise return None (ChromaDB will use its built-in default).

    Args:
        cfg: MempalaceConfig instance.
        palace_path: Path to the palace directory (for params sidecar file).

    Returns:
        TurboQuantEmbeddingFunction or None.
    """
    if not cfg.use_turboquant:
        return None
    return TurboQuantEmbeddingFunction(
        bits=cfg.turboquant_bits,
        params_path=TurboQuantEmbeddingFunction.params_path_for_palace(palace_path),
    )
