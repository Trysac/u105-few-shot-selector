# Architecture

## Overview

`few_shot_selector` enables deterministic few-shot prompt selection. It uses MMR to optimize for both relevance to the query and diversity of the selected set.

## Core Components

- **Dataclasses**: `Example` (candidate demonstration) and `Selection` (result containing chosen examples, scores, and skipped IDs).
- **Numeric Engine**: `cosine` computes L2-normalized similarity. `mmr` implements: $Score(d) = \lambda \cos(q, d) - (1 - \lambda) \max_{s \in S} \cos(d, s)$. Tie-breaks favor the lowest pool index.
- **`HashingEmbedder`**: Deterministic feature hashing using `hashlib.blake2b`. It processes NFKC-normalized tokens to generate feature indices and pseudo-random signs, guaranteeing identical outputs across environments.
- **Orchestrator**: `select` coordinates embedding, MMR ranking, duplicate filtering via `duplicate_threshold`, and budget management using a provided `counter` (default: `word_count`).
- **Templating**: `format_examples` performs safe replacement of `{input}` and `{output}` tokens, preserving braces in raw text.
