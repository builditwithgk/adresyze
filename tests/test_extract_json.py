"""The JSON extractor runs on every inference, so test it without importing Modal."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_modal_app", Path(__file__).resolve().parents[1] / "modal_app.py"
)
_mod = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_mod)
    extract_json = _mod.extract_json
except ModuleNotFoundError:  # modal not installed locally
    pytestmark = pytest.mark.skip(reason="modal not installed (pip install '.[remote]')")
    extract_json = None


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_prose_wrapped():
    assert extract_json('Sure! Here is the layout:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_nested_braces_are_balanced():
    out = extract_json('{"elements": [{"bbox": [0, 0, 1, 1]}], "n": {"x": 2}}')
    assert out["elements"][0]["bbox"] == [0, 0, 1, 1]
    assert out["n"] == {"x": 2}


def test_trailing_comma_is_repaired():
    assert extract_json('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}


def test_no_json_returns_none():
    assert extract_json("I cannot analyze this image.") is None


def test_unbalanced_returns_none():
    assert extract_json('{"a": 1') is None


def test_bare_array_is_rejected():
    assert extract_json("[1, 2, 3]") is None
