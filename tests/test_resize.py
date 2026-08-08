"""Reflow engine tests. Pure geometry -- no model, no network."""

from __future__ import annotations

import pytest
from PIL import Image

from adresyze import BBox, Element, ElementType, Layout
from adresyze.resize import ATOMIC, ResizePlan, parse_ratio, plan, render, resize


def make_layout(*elements: Element, width: int = 1200, height: int = 628) -> Layout:
    return Layout(
        elements=list(elements),
        dominant_colors=["#ffffff", "#112233"],
        source_width=width,
        source_height=height,
    )


def el(kind: ElementType, x1, y1, x2, y2, priority: int = 1) -> Element:
    return Element(
        type=kind, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), priority=priority
    )


# ------------------------------------------------------------------ ratios


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1:1", 1.0), ("4:5", 0.8), ("1.91:1", 1.91), ("9:16", 0.5625), (0.75, 0.75), ("2:1", 2.0)],
)
def test_parse_ratio(value, expected):
    assert parse_ratio(value) == pytest.approx(expected)


def test_parse_ratio_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_ratio("0:1")


# ------------------------------------------------------------------- plan


def test_same_ratio_is_identity():
    layout = make_layout(el(ElementType.PRODUCT, 0.1, 0.1, 0.5, 0.5), width=600, height=600)
    p = plan(layout, "1:1")
    assert p.mode == "identity"
    assert p.retention == 1.0


def test_crop_keeps_everything_when_it_can():
    """Content sits in the left half, so a 1:1 crop of a wide ad can keep all of it."""
    layout = make_layout(
        el(ElementType.PRODUCT, 0.02, 0.1, 0.4, 0.9),
        el(ElementType.CTA, 0.05, 0.1, 0.3, 0.3),
        width=1200,
        height=600,
    )
    p = plan(layout, "1:1")
    assert p.mode == "crop"
    assert p.canvas_size == (600, 600)
    assert p.retention == 1.0
    assert not p.lost


def test_pads_rather_than_cutting_a_cta():
    """Elements at both edges cannot survive any crop -- padding must win."""
    layout = make_layout(
        el(ElementType.CTA, 0.01, 0.4, 0.12, 0.6),
        el(ElementType.LOGO, 0.88, 0.4, 0.99, 0.6),
        width=1200,
        height=600,
    )
    p = plan(layout, "1:1")
    assert p.mode == "pad"
    assert p.canvas_size == (1200, 1200)
    assert not p.lost, "padding must never lose an element"


def test_crop_prefers_the_side_with_more_value():
    """A high-priority CTA on the right should pull the crop window right."""
    layout = make_layout(
        el(ElementType.OTHER, 0.0, 0.0, 0.2, 1.0, priority=3),
        el(ElementType.CTA, 0.75, 0.3, 0.95, 0.7),
        width=1200,
        height=600,
    )
    p = plan(layout, "1:1")
    assert p.mode == "crop"
    assert p.crop_box[0] > 300, "window should shift toward the CTA"
    assert all(pl.kept for pl in p.placements if pl.type is ElementType.CTA)


def test_atomic_elements_are_all_or_nothing():
    layout = make_layout(el(ElementType.LOGO, 0.4, 0.4, 0.6, 0.6), width=1200, height=600)
    p = plan(layout, "1:1")
    logo = next(pl for pl in p.placements if pl.type is ElementType.LOGO)
    assert logo.visible_fraction == pytest.approx(1.0)
    assert logo.kept


def test_partial_atomic_counts_as_lost():
    from adresyze.resize import Placement

    half = Placement(ElementType.CTA, 1, 0.5, None)
    assert not half.kept
    assert ElementType.CTA in ATOMIC
    # a product surviving half is still useful
    assert Placement(ElementType.PRODUCT, 1, 0.5, None).kept


def test_background_extent_is_ignored():
    """The model's background box is unreliable, so it must not steer the crop."""
    layout = make_layout(
        el(ElementType.BACKGROUND, 0.0, 0.0, 0.3, 0.3),
        el(ElementType.PRODUCT, 0.6, 0.2, 0.95, 0.8),
        width=1200,
        height=600,
    )
    p = plan(layout, "1:1")
    assert p.crop_box[0] > 300, "background must not drag the window left"


def test_placements_are_in_canvas_coordinates():
    layout = make_layout(el(ElementType.PRODUCT, 0.4, 0.25, 0.6, 0.75), width=1200, height=600)
    p = plan(layout, "1:1")
    placed = next(pl for pl in p.placements if pl.type is ElementType.PRODUCT)
    assert placed.bbox is not None
    for v in placed.bbox.as_tuple():
        assert 0.0 <= v <= 1.0


def test_empty_layout_does_not_crash():
    p = plan(make_layout(width=1200, height=600), "1:1")
    assert p.retention == 1.0
    assert p.canvas_size == (600, 600)


# ----------------------------------------------------------------- render


def test_render_crop_produces_exact_canvas():
    layout = make_layout(el(ElementType.PRODUCT, 0.02, 0.1, 0.4, 0.9), width=1200, height=600)
    img = Image.new("RGB", (1200, 600), (10, 120, 200))
    out, p = resize(img, layout, "1:1")
    assert p.mode == "crop"
    assert out.size == (600, 600)
    assert out.getpixel((300, 300)) == (10, 120, 200)


def test_render_pad_fills_with_sampled_colour():
    layout = make_layout(
        el(ElementType.CTA, 0.01, 0.4, 0.12, 0.6),
        el(ElementType.LOGO, 0.88, 0.4, 0.99, 0.6),
        width=1200,
        height=600,
    )
    img = Image.new("RGB", (1200, 600), (200, 30, 40))
    out, p = resize(img, layout, "1:1")
    assert p.mode == "pad"
    assert out.size == (1200, 1200)
    # padding sampled from the (uniform) source, so top strip matches the ad
    assert out.getpixel((600, 5)) == (200, 30, 40)
    assert out.getpixel((600, 600)) == (200, 30, 40)


def test_render_honours_requested_width():
    layout = make_layout(el(ElementType.PRODUCT, 0.1, 0.1, 0.5, 0.5), width=1200, height=600)
    img = Image.new("RGB", (1200, 600), (0, 0, 0))
    out, _ = resize(img, layout, "1:1", width=256)
    assert out.size == (256, 256)


def test_all_presets_render():
    layout = make_layout(el(ElementType.PRODUCT, 0.3, 0.3, 0.7, 0.7), width=1000, height=1000)
    img = Image.new("RGB", (1000, 1000), (255, 255, 255))
    for target in ("1:1", "4:5", "1.91:1", "9:16", "16:9"):
        out, p = resize(img, layout, target, width=400)
        assert out.width == 400
        assert isinstance(p, ResizePlan)
        assert p.retention == pytest.approx(1.0)


def test_plan_describes_itself():
    layout = make_layout(el(ElementType.PRODUCT, 0.1, 0.1, 0.9, 0.9), width=1200, height=600)
    text = plan(layout, "4:5").describe()
    assert "1200x600" in text and "retention" in text
