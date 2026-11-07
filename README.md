# few-shot-selector

[![PyPI version](https://img.shields.io/pypi/v/trysac-few-shot-selector)](https://pypi.org/project/trysac-few-shot-selector/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-few-shot-selector)](LICENSE)
[![CI](https://github.com/Trysac/u105-few-shot-selector/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u105-few-shot-selector/actions/workflows/ci.yml)

Select few-shot examples by similarity and diversity with maximal marginal relevance, a deterministic hashing embedder, token budgets and duplicate suppression

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-few-shot-selector
# or
uv add trysac-few-shot-selector
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import few_shot_selector

# normalize() trims surrounding whitespace and lower-cases the text.
print(few_shot_selector.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    few_shot_selector.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
