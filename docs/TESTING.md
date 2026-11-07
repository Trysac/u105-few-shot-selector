# Testing

## Coverage

The test suite in `tests/test_few_shot_selector.py` covers:

- **MMR Logic**: Selection order, tie-breaking behavior, and boundary lambda values (0.0 to 1.0).
- **Embedder**: Determinism of `HashingEmbedder` and L2 normalization correctness.
- **Contract**: Single-call embedder invocation in `select`.
- **Budgeting**: Token count exhaustion and `skipped_for_budget` tracking.
- **Duplicate Suppression**: Filtering based on `duplicate_threshold`.
- **Error Handling**: Verification of all `SelectorErrorCode` variants (`invalid_k`, `invalid_lambda`, `dimension_mismatch`, `invalid_embedding`, `pool_too_large`, `input_too_large`, `invalid_counter`).
- **Formatting**: Safe string templating with braces.
- **Edge Cases**: Empty pools, k=len(pool), zero vectors, unicode input, and budget smaller than any individual example.
