# Deterministic MMR selection with standard library feature hashing

Status: accepted (generated, pending human review)

## Context

Prompting LLMs with few-shot demonstrations requires balancing relevance and diversity without pulling in large vector databases or external ML frameworks. The embedder must be reproducible across worker processes.

## Decision

Implement MMR with a deterministic blake2b hashing embedder, token budget checks, and duplicate filtering in pure Python stdlib, using str.replace for prompt templating.

## Consequences

Zero external runtime dependencies. Deterministic cross-process embeddings without requiring pre-trained weights or network calls. Standard embedder protocol allows plugging in external models when desired.
