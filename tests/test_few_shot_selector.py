"""Unit tests for few_shot_selector."""

from __future__ import annotations

import math
from typing import Any

import pytest

from few_shot_selector import (
    Example,
    HashingEmbedder,
    Selection,
    SelectorError,
    cosine,
    format_examples,
    mmr,
    select,
    word_count,
)


def test_acceptance_mmr_ties_and_lambda() -> None:
    """Verify acceptance criteria for mmr calculation and tie breaking."""
    assert mmr([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], k=2, lambda_=0.3) == [0, 2]
    assert mmr([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], k=2, lambda_=1.0) == [0, 1]


def test_acceptance_mmr_k_larger_and_validation() -> None:
    """Verify mmr with k > pool, k=0 and lambda_=1.5."""
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    assert mmr([1.0, 0.0], vecs, k=5, lambda_=0.5) == [0, 1]

    with pytest.raises(SelectorError) as exc_k:
        mmr([1.0, 0.0], vecs, k=0, lambda_=0.5)
    assert exc_k.value.code == "invalid_k"

    with pytest.raises(SelectorError) as exc_lambda:
        mmr([1.0, 0.0], vecs, k=2, lambda_=1.5)
    assert exc_lambda.value.code == "invalid_lambda"


def test_acceptance_hashing_embedder_deterministic_and_zero_norm() -> None:
    """Verify HashingEmbedder determinism, L2 norm, empty string, and cosine(zero, v)."""
    embedder1 = HashingEmbedder(dim=256)
    embedder2 = HashingEmbedder(dim=256)
    v1 = embedder1(["Hello world"])[0]
    v2 = embedder2(["Hello world"])[0]
    assert v1 == v2

    norm = math.sqrt(sum(x * x for x in v1))
    assert abs(norm - 1.0) < 1e-12

    empty_v = embedder1([""])[0]
    assert empty_v == [0.0] * 256
    assert cosine(empty_v, v1) == 0.0


def test_acceptance_select_calls_embedder_once_and_mmr_order() -> None:
    """select calls embedder once with [query, *inputs] and returns examples in MMR order."""
    calls: list[list[str]] = []

    def fake(texts: Any) -> list[list[float]]:
        calls.append(list(texts))
        # Generate distinct orthogonal-ish vectors based on text
        res: list[list[float]] = []
        for t in texts:
            if t == "q":
                res.append([1.0, 0.0, 0.0])
            elif t == "in1":
                res.append([0.9, 0.1, 0.0])
            elif t == "in2":
                res.append([0.1, 0.9, 0.0])
            else:
                res.append([0.0, 0.0, 1.0])
        return res

    pool = [
        Example(id="e1", input="in1", output="out1"),
        Example(id="e2", input="in2", output="out2"),
    ]
    sel = select("q", pool, k=2, embed=fake)
    assert len(calls) == 1
    assert calls[0] == ["q", "in1", "in2"]
    assert [e.id for e in sel.examples] == ["e1", "e2"]


def test_acceptance_token_budget_skipping() -> None:
    """With max_tokens=10 and counter=word_count, skip 12-word candidate."""
    # e1 has 12 words total: input + ' ' + output
    e1 = Example(id="e1", input="w1 w2 w3 w4 w5", output="w6 w7 w8 w9 w10 w11 w12")
    # e2 has 4 words
    e2 = Example(id="e2", input="hello world", output="nice answer")

    def fake_embed(texts: Any) -> list[list[float]]:
        # Query matches e1 better than e2
        res: list[list[float]] = []
        for t in texts:
            if t == "query":
                res.append([1.0, 0.0])
            elif t == e1.input:
                res.append([0.99, 0.01])
            else:
                res.append([0.5, 0.5])
        return res

    sel = select(
        "query",
        [e1, e2],
        k=2,
        embed=fake_embed,
        max_tokens=10,
        counter=word_count,
    )
    assert [e.id for e in sel.examples] == ["e2"]
    assert sel.skipped_for_budget == ("e1",)


def test_acceptance_duplicate_threshold_suppression() -> None:
    """Duplicate threshold suppresses identical input examples with lambda_=1.0."""
    pool = [
        Example(id="e1", input="same input", output="out1"),
        Example(id="e2", input="same input", output="out2"),
        Example(id="e3", input="different input", output="out3"),
    ]
    sel = select("same input", pool, k=2, lambda_=1.0, duplicate_threshold=0.98)
    assert [e.id for e in sel.examples] == ["e1", "e3"]


def test_acceptance_error_codes() -> None:
    """Check dimension_mismatch, invalid_embedding, pool_too_large, input_too_large."""

    def wrong_dim_embedder(texts: Any) -> list[list[float]]:
        return [[1.0, 0.0], [1.0]]  # second vector wrong dim

    pool = [Example(id="1", input="hello", output="world")]
    with pytest.raises(SelectorError) as exc_dim:
        select("q", pool, embed=wrong_dim_embedder)
    assert exc_dim.value.code == "dimension_mismatch"

    def nan_embedder(texts: Any) -> list[list[float]]:
        return [[1.0, 0.0], [float("nan"), 0.0]]

    with pytest.raises(SelectorError) as exc_nan:
        select("q", pool, embed=nan_embedder)
    assert exc_nan.value.code == "invalid_embedding"

    large_pool = [Example(id=str(i), input="a", output="b") for i in range(10_001)]
    with pytest.raises(SelectorError) as exc_pool:
        select("q", large_pool)
    assert exc_pool.value.code == "pool_too_large"

    long_input = "x" * 20_001
    with pytest.raises(SelectorError) as exc_input:
        select("q", [Example(id="1", input=long_input, output="b")])
    assert exc_input.value.code == "input_too_large"


def test_acceptance_format_examples() -> None:
    """Verify format_examples escapes braces and joins with double newline."""
    ex1 = Example("1", "a {x}", "b")
    assert format_examples([ex1]) == "Input: a {x}\nOutput: b"

    ex2 = Example("2", "c", "d {y}")
    expected = "Input: a {x}\nOutput: b\n\nInput: c\nOutput: d {y}"
    assert format_examples([ex1, ex2]) == expected


def test_edge_cases() -> None:
    """Test empty pool, budget smaller than every example, all zero embeddings, unicode input."""
    # Empty pool
    sel_empty = select("q", [])
    assert sel_empty == Selection(examples=(), scores=(), skipped_for_budget=())

    # Empty vecs mmr
    assert mmr([1.0, 0.0], [], k=3, lambda_=0.5) == []

    # Budget smaller than every example
    pool = [
        Example(id="e1", input="word one", output="word two"),
        Example(id="e2", input="word three", output="word four"),
    ]
    sel_budget = select("q", pool, k=2, max_tokens=1)
    assert sel_budget.examples == ()
    assert sel_budget.skipped_for_budget == ("e1", "e2")

    # All-zero embeddings: cosines are all 0.0, order by index
    def zero_embed(texts: Any) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]

    sel_zero = select("q", pool, k=2, embed=zero_embed)
    assert [e.id for e in sel_zero.examples] == ["e1", "e2"]

    # Unicode input
    embedder = HashingEmbedder(dim=32)
    v_unicode = embedder(["Café crème ñ ü 日本語"])[0]
    assert len(v_unicode) == 32
    norm = math.sqrt(sum(x * x for x in v_unicode))
    assert abs(norm - 1.0) < 1e-12


def test_additional_validation_and_branch_coverage() -> None:
    """Exercise error branches for k, lambda, counter, and cosine."""
    pool = [Example(id="1", input="in", output="out")]
    with pytest.raises(SelectorError) as exc_k:
        select("q", pool, k=-1)
    assert exc_k.value.code == "invalid_k"

    with pytest.raises(SelectorError) as exc_k_bool:
        select("q", pool, k=True)  # type: ignore[arg-type]
    assert exc_k_bool.value.code == "invalid_k"

    with pytest.raises(SelectorError) as exc_lam:
        select("q", pool, lambda_=-0.1)
    assert exc_lam.value.code == "invalid_lambda"

    # Query too large
    with pytest.raises(SelectorError) as exc_q:
        select("q" * 20_001, pool)
    assert exc_q.value.code == "input_too_large"

    # Invalid counter
    with pytest.raises(SelectorError) as exc_count:
        select("q", pool, counter=lambda _: -5)
    assert exc_count.value.code == "invalid_counter"

    with pytest.raises(SelectorError) as exc_count_type:
        select("q", pool, counter=lambda _: "10")  # type: ignore[arg-type]
    assert exc_count_type.value.code == "invalid_counter"

    with pytest.raises(SelectorError) as exc_count_err:

        def bad_counter(_: str) -> int:
            raise RuntimeError("boom")

        select("q", pool, counter=bad_counter)
    assert exc_count_err.value.code == "invalid_counter"

    # Embedder returning wrong number of vectors
    with pytest.raises(SelectorError) as exc_ret_len:
        select("q", pool, embed=lambda _: [[1.0, 0.0]])
    assert exc_ret_len.value.code == "dimension_mismatch"

    # Embedder dimension mismatch across query and pool
    with pytest.raises(SelectorError) as exc_v_len:
        mmr([1.0, 0.0], [[1.0, 0.0, 0.0]], k=1, lambda_=0.5)
    assert exc_v_len.value.code == "dimension_mismatch"

    # Cosine dimension mismatch
    with pytest.raises(SelectorError) as exc_cos:
        cosine([1.0], [1.0, 2.0])
    assert exc_cos.value.code == "dimension_mismatch"

    # Cosine clamping and zero norm
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0

    # HashingEmbedder invalid dim
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
