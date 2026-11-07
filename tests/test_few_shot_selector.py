"""Behavior tests for the public API of few_shot_selector."""

from __future__ import annotations

import pytest

import few_shot_selector


def test_normalize_trims_and_lowercases() -> None:
    assert few_shot_selector.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        few_shot_selector.normalize(value)


def test_version_is_semver() -> None:
    parts = few_shot_selector.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
