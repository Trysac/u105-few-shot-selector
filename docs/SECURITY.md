# Security

## Threat Model

`few_shot_selector` processes potentially untrusted user queries and candidate example pools.

### Defended

- **Denial of Service**: 
  - `max_input_chars` (default 20,000) prevents oversized strings from straining resources during regex and hashing.
  - `max_pool` (default 10,000) limits the size of candidate pools, preventing quadratic complexity memory/CPU usage.
- **Input Validation**: 
  - Embeddings are checked for `NaN`, `Infinity`, and dimension consistency.
  - `counter` functions are validated for non-negative integers.
- **Information Disclosure**: 
  - Error messages truncate inputs (max 200 chars) to prevent log-based leakage.

### Not Defended

- **Prompt Injection**: The library assumes inputs provided by the user are untrusted and does not sanitize content against prompt injection, as it is a pure selection utility.
