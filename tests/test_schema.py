"""Schema unit tests plus a conformance run over every real v1 record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adresyze import BBox, ElementType, RawLayout, normalize
from adresyze.schema import canonical_types

V1_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "v1"


# --------------------------------------------------------------------- units


def test_compound_label_splits():
    assert canonical_types("logo|product") == [ElementType.LOGO, ElementType.PRODUCT]


def test_unknown_label_falls_back_to_other():
    assert canonical_types("banana") == [ElementType.OTHER]


def test_alias_maps_to_canonical():
    assert canonical_types("call_to_action") == [ElementType.CTA]


def test_bbox_from_grid_normalizes_and_clamps():
    box = BBox.from_grid([0, 0, 588, 294], 588, 588)
    assert box.as_tuple() == (0.0, 0.0, 1.0, 0.5)
    assert BBox.from_grid([-10, 0, 9999, 588], 588, 588).as_tuple() == (0.0, 0.0, 1.0, 1.0)


def test_bbox_rejects_empty():
    with pytest.raises(ValueError):
        BBox(x1=0.5, y1=0.0, x2=0.5, y2=1.0)


def test_bbox_iou_and_pixels():
    a = BBox(x1=0, y1=0, x2=0.5, y2=1)
    assert a.iou(BBox(x1=0.5, y1=0, x2=1, y2=1)) == 0.0
    assert a.iou(a) == pytest.approx(1.0)
    assert a.to_pixels(1000, 500) == (0, 0, 500, 500)


def test_raw_element_reorders_flipped_bbox():
    layout = RawLayout(elements=[{"type": "logo", "bbox": [90, 80, 10, 20]}])
    assert layout.elements[0].bbox == [10, 20, 90, 80]


def test_normalize_requires_source_dimensions():
    raw = RawLayout(elements=[{"type": "logo", "bbox": [0, 0, 100, 100]}])
    with pytest.raises(ValueError, match="source dimensions"):
        normalize(raw)


def test_normalize_grounds_splits_and_synthesizes_background():
    raw = RawLayout(
        elements=[
            {"type": "logo|product", "bbox": [0, 0, 294, 294]},
            {"type": "cta", "bbox": [10, 10, 10, 50]},  # zero width -> dropped
        ],
        grid_width=588,
        grid_height=588,
    )
    layout, report = normalize(raw, source_width=1200, source_height=628)

    assert report.split_compound == 1
    assert report.dropped_empty == 1
    assert report.synthesized_background is True
    assert {e.type for e in layout.foreground} == {ElementType.LOGO, ElementType.PRODUCT}
    assert layout.background.bbox.as_tuple() == (0.0, 0.0, 1.0, 1.0)
    assert layout.aspect_ratio == pytest.approx(1200 / 628)
    # normalized, so it replays onto any canvas
    assert layout.foreground[0].bbox.to_pixels(1200, 628) == (0, 0, 600, 314)


def test_normalize_merges_duplicate_boxes():
    raw = RawLayout(
        elements=[
            {"type": "headline", "bbox": [0, 0, 300, 100]},
            {"type": "headline", "bbox": [0, 0, 302, 101]},
        ],
        grid_width=588,
        grid_height=588,
    )
    layout, report = normalize(raw, source_width=600, source_height=600)
    assert report.merged_duplicates == 1
    assert len(layout.of_type(ElementType.HEADLINE)) == 1


# --------------------------------------------------- conformance vs real data


def _v1_records():
    return sorted(V1_CACHE.glob("*.json")) if V1_CACHE.exists() else []


requires_v1 = pytest.mark.skipif(
    not _v1_records(), reason="run `python pipeline/fetch_v1_dataset.py` first"
)


@requires_v1
def test_every_v1_record_parses_as_raw():
    failures = []
    for path in _v1_records():
        try:
            RawLayout.model_validate(json.loads(path.read_text()))
        except Exception as exc:  # noqa: BLE001
            failures.append((path.name, str(exc)[:120]))
    assert not failures, f"{len(failures)} of {len(_v1_records())} failed: {failures[:3]}"


@requires_v1
def test_every_v1_record_normalizes_without_crashing():
    """v1 has no source dims, so we supply them -- this exercises the repair path."""
    for path in _v1_records():
        raw = RawLayout.model_validate(json.loads(path.read_text()))
        layout, report = normalize(raw, source_width=1200, source_height=628)
        assert report.ungrounded is True, "v1 must never claim to be grounded"
        assert layout.background is not None
        for el in layout.elements:
            assert 0.0 <= el.bbox.x1 < el.bbox.x2 <= 1.0
            assert 0.0 <= el.bbox.y1 < el.bbox.y2 <= 1.0


@requires_v1
def test_v1_vocabulary_is_fully_covered():
    """Any label in the wild must resolve; `other` is only allowed for real junk."""
    labels = set()
    for path in _v1_records():
        for el in json.loads(path.read_text()).get("elements", []):
            labels.add(el["type"])
    unresolved = {
        lbl
        for lbl in labels
        if canonical_types(lbl) == [ElementType.OTHER] and lbl.lower() != "other"
    }
    assert not unresolved, f"labels falling through to OTHER: {sorted(unresolved)}"
