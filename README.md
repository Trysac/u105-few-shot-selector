# few-shot-selector

[![CI](https://github.com/Trysac/u105-few-shot-selector/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u105-few-shot-selector/actions/workflows/ci.yml)

Select few-shot examples by similarity and diversity with maximal marginal relevance, a deterministic hashing embedder, token budgets and duplicate suppression.

## Installation

```bash
pip install trysac-few-shot-selector
```

## Usage

```python
from few_shot_selector import Example, select, format_examples

pool = [Example("1", "Say hello in French", "Bonjour"), Example("2", "Say yes in French", "Oui")]
sel = select("Say thanks in French", pool, k=1)
output = format_examples(sel.examples)
```

## Features

- **Maximal Marginal Relevance (MMR)**: Balance relevance with diversity.
- **Deterministic Hashing Embedder**: Built-in blake2b feature hashing that guarantees reproducibility without external dependencies.
- **Token Budget & Duplicate Controls**: Enforce token limits and filter out redundant examples.
- **Safe Templating**: Preserves braces in example inputs.

## Public API Summary

- **Core Structures**: `Example(id, input, output)` and `Selection(examples, scores, skipped_for_budget)`.
- **Main functions**: `select()` (MMR selection with budget/dup control), `mmr()` (pure calculation), `cosine()` (similarity), `format_examples()` (safe string templating).
- **Embedder**: `HashingEmbedder(dim=256)` generates deterministic vectors.
- **Errors**: `SelectorError` raised with codes: `invalid_k`, `invalid_lambda`, `dimension_mismatch`, `invalid_embedding`, `pool_too_large`, `input_too_large`, `invalid_counter`.
- **Limits**: Configurable via `max_pool` (default 10,000) and `max_input_chars` (default 20,000).

## Why not <best-known alternative>?

Unlike frameworks such as LangChain's `MaxMarginalRelevanceExampleSelector`, this utility is built with zero external dependencies, features a deterministic default embedder, and offers granular, integrated token budget and duplicate suppression.

## Security

See [docs/SECURITY.md](docs/SECURITY.md) for details on threat modeling and input constraints.
