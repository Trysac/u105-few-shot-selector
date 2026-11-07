"""Select few-shot examples by similarity and diversity with maximal marginal relevance."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "Embedder",
    "Example",
    "HashingEmbedder",
    "Selection",
    "SelectorError",
    "SelectorErrorCode",
    "__version__",
    "cosine",
    "format_examples",
    "mmr",
    "select",
    "word_count",
]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SelectorErrorCode = Literal[
    "invalid_k",
    "invalid_lambda",
    "dimension_mismatch",
    "invalid_embedding",
    "pool_too_large",
    "input_too_large",
    "invalid_counter",
]


class SelectorError(ValueError):
    """Base exception raised by few_shot_selector."""

    def __init__(self, message: str, code: SelectorErrorCode) -> None:
        super().__init__(message)
        self.code: SelectorErrorCode = code


@dataclass(frozen=True)
class Example:
    """A few-shot candidate example."""

    id: str
    input: str
    output: str


@dataclass(frozen=True)
class Selection:
    """Result of few-shot example selection."""

    examples: tuple[Example, ...]
    scores: tuple[float, ...]
    skipped_for_budget: tuple[str, ...]


Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def word_count(text: str) -> int:
    """Default token counter based on whitespace splitting."""
    return len(text.split())


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return [0.0 for _ in vec]
    return [x / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric sequences.

    If either vector has zero L2 norm, returns 0.0.
    """
    if len(a) != len(b):
        raise SelectorError(
            "Vector dimensions must match for cosine similarity",
            code="dimension_mismatch",
        )
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    cos = dot / (norm_a * norm_b)
    # Numerical clamp
    if cos > 1.0:
        return 1.0
    if cos < -1.0:
        return -1.0
    return cos


_WORD_RE = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic embedder using feature hashing and blake2b."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        res: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            normalized = unicodedata.normalize("NFKC", text).lower()
            tokens = _WORD_RE.findall(normalized)
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                raw_val = int.from_bytes(digest, byteorder="big")
                idx = raw_val % self.dim
                # Use 1 bit for sign: bit 0 of the first byte
                sign = 1.0 if (digest[0] & 1) == 0 else -1.0
                vec[idx] += sign
            res.append(_l2_normalize(vec))
        return res


def mmr(
    query_vec: Sequence[float],
    vecs: Sequence[Sequence[float]],
    *,
    k: int,
    lambda_: float,
) -> list[int]:
    """Rank candidate vector indices using Maximal Marginal Relevance (MMR)."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise SelectorError(f"k must be an integer >= 1, got {k!r}", code="invalid_k")
    if not (0.0 <= lambda_ <= 1.0):
        raise SelectorError(f"lambda_ must be in [0, 1], got {lambda_!r}", code="invalid_lambda")

    n = len(vecs)
    if n == 0:
        return []

    dim = len(query_vec)
    # Normalise query
    norm_q = _l2_normalize(query_vec)

    norm_vecs: list[list[float]] = []
    for i, v in enumerate(vecs):
        if len(v) != dim:
            raise SelectorError(
                f"Vector at index {i} has dimension {len(v)}, expected {dim}",
                code="dimension_mismatch",
            )
        norm_vecs.append(_l2_normalize(v))

    # Cosine with query for all docs. If either is 0 vector, cosine is 0.0.
    q_sims = [sum(q * d for q, d in zip(norm_q, d_vec, strict=True)) for d_vec in norm_vecs]

    remaining = set(range(n))
    selected: list[int] = []
    target_k = min(k, n)

    # Precompute or dynamically calculate pairwise similarity matrix between selected & remaining
    for step in range(target_k):
        best_idx = -1
        best_score = -float("inf")
        # Iterate in ascending order of candidate indices for tie breaking
        for cand in sorted(remaining):
            if step == 0:
                score = q_sims[cand]
            else:
                max_sim_to_sel = max(
                    sum(norm_vecs[cand][d] * norm_vecs[s][d] for d in range(dim)) for s in selected
                )
                score = lambda_ * q_sims[cand] - (1.0 - lambda_) * max_sim_to_sel

            # Ties go to lowest pool index (strictly greater replaces)
            if score > best_score:
                best_score = score
                best_idx = cand

        if best_idx != -1:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def _validate_vector(vec: Sequence[float], expected_dim: int) -> None:
    if len(vec) != expected_dim:
        raise SelectorError(
            f"Embedding dimension {len(vec)} does not match expected {expected_dim}",
            code="dimension_mismatch",
        )
    for val in vec:
        if math.isnan(val) or math.isinf(val):
            raise SelectorError(
                "Embedding vector contains NaN or Infinity",
                code="invalid_embedding",
            )


def select(
    query: str,
    pool: Sequence[Example],
    *,
    k: int = 4,
    lambda_: float = 0.7,
    embed: Embedder | None = None,
    max_tokens: int | None = None,
    counter: Callable[[str], int] = word_count,
    duplicate_threshold: float = 0.98,
    max_pool: int = 10_000,
    max_input_chars: int = 20_000,
) -> Selection:
    """Select few-shot examples using MMR, budget, and duplicate filtering."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise SelectorError(f"k must be an integer >= 1, got {k!r}", code="invalid_k")
    if not (0.0 <= lambda_ <= 1.0):
        raise SelectorError(f"lambda_ must be in [0, 1], got {lambda_!r}", code="invalid_lambda")
    if len(pool) > max_pool:
        raise SelectorError(
            f"Pool size {len(pool)} exceeds max_pool limit {max_pool}",
            code="pool_too_large",
        )

    if len(query) > max_input_chars:
        truncated = query[:200]
        raise SelectorError(
            f"Query length {len(query)} exceeds max_input_chars {max_input_chars}: {truncated}",
            code="input_too_large",
        )

    for ex in pool:
        if len(ex.input) > max_input_chars:
            truncated = ex.input[:200]
            raise SelectorError(
                f"Example {ex.id} length {len(ex.input)} exceeds {max_input_chars}: {truncated}",
                code="input_too_large",
            )

    if len(pool) == 0:
        return Selection(examples=(), scores=(), skipped_for_budget=())

    embedder = embed if embed is not None else HashingEmbedder()
    inputs_to_embed = [query] + [ex.input for ex in pool]
    embedded_raw = embedder(inputs_to_embed)

    if len(embedded_raw) != len(inputs_to_embed):
        raise SelectorError(
            f"Embedder returned {len(embedded_raw)} vectors for {len(inputs_to_embed)} texts",
            code="dimension_mismatch",
        )

    dim = len(embedded_raw[0])
    for v in embedded_raw:
        _validate_vector(v, dim)

    q_norm = _l2_normalize(embedded_raw[0])
    doc_norms = [_l2_normalize(v) for v in embedded_raw[1:]]
    q_sims = [sum(q * d for q, d in zip(q_norm, d_vec, strict=True)) for d_vec in doc_norms]

    # MMR selection loop with duplicate suppression and budget checking
    selected_examples: list[Example] = []
    selected_scores: list[float] = []
    selected_indices: list[int] = []
    skipped_for_budget: list[str] = []
    current_tokens = 0

    remaining = set(range(len(pool)))

    while remaining and len(selected_examples) < k:
        best_idx = -1
        best_score = -float("inf")

        for cand in sorted(remaining):
            if not selected_indices:
                score = q_sims[cand]
            else:
                max_sim = max(
                    sum(doc_norms[cand][d] * doc_norms[s][d] for d in range(dim))
                    for s in selected_indices
                )
                score = lambda_ * q_sims[cand] - (1.0 - lambda_) * max_sim

            if score > best_score:
                best_score = score
                best_idx = cand

        if best_idx == -1:
            break

        # Candidate is removed from MMR pool in this order
        remaining.remove(best_idx)
        cand_ex = pool[best_idx]
        cand_vec = doc_norms[best_idx]

        # 1. Duplicate check against already selected examples
        is_duplicate = False
        if selected_indices:
            max_sim_to_selected = max(
                sum(cand_vec[d] * doc_norms[s][d] for d in range(dim)) for s in selected_indices
            )
            if max_sim_to_selected >= duplicate_threshold:
                is_duplicate = True

        if is_duplicate:
            continue

        # 2. Budget check
        text_to_count = cand_ex.input + " " + cand_ex.output
        try:
            ex_tokens: Any = counter(text_to_count)
        except Exception as e:
            raise SelectorError(f"Counter raised an error: {e}", code="invalid_counter") from e

        if isinstance(ex_tokens, bool) or not isinstance(ex_tokens, int) or ex_tokens < 0:
            raise SelectorError(
                f"Counter returned invalid token count: {ex_tokens!r}",
                code="invalid_counter",
            )

        if max_tokens is not None and (current_tokens + ex_tokens > max_tokens):
            skipped_for_budget.append(cand_ex.id)
            continue

        # Candidate accepted
        selected_examples.append(cand_ex)
        selected_scores.append(best_score)
        selected_indices.append(best_idx)
        current_tokens += ex_tokens

    return Selection(
        examples=tuple(selected_examples),
        scores=tuple(selected_scores),
        skipped_for_budget=tuple(skipped_for_budget),
    )


def format_examples(
    examples: Sequence[Example],
    template: str = "Input: {input}\nOutput: {output}",
) -> str:
    """Format examples using simple string replacement for {input} and {output}."""
    formatted: list[str] = []
    for ex in examples:
        text = template.replace("{input}", ex.input).replace("{output}", ex.output)
        formatted.append(text)
    return "\n\n".join(formatted)
