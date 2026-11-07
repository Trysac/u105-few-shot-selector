# Project Record

## Objective

Select few-shot examples by similarity and diversity with maximal marginal relevance.

## Scope

- In-memory MMR selection for few-shot prompt optimization.
- Python standard library implementation.
- Deterministic default feature hashing embedder.
- Configurable token budget and duplicate management.

## Out-of-Scope

- Integration with vector databases.
- Real-time LLM interaction (only prompt selection).
- Label balancing.
- Asynchronous processing.

## Factory run u105 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | gemini-3.8-flash | yes | 4185 | 9103 | 0 | 0 | 0.0186 |
| fix-1 | gemini-3.8-flash | no | 78650 | 1217 | 0 | 534566 | 0.1036 |
| audit | gemini-3.1-flash-lite | no | 40536 | 439 | 0 | 77432 | 0.0127 |
| docs | gemini-3.1-flash-lite | no | 192256 | 1866 | 0 | 0 | 0.0509 |

Total: $0.1859. Prices: Anthropic list prices.
