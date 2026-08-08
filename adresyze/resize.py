"""Deterministic, layout-aware reflow. No generative model involved.

The engine never squashes: it either **crops** to the target ratio along the axis it
can afford to lose, or **pads** the canvas and extends the background. Which one it
picks is decided by whether a crop can keep every element that matters.

Elements are not equal. A call-to-action half cut off is worse than no call-to-action,
so atomic types score all-or-nothing; a product photo degrades gracefully and scores by
visible area. Weights combine the element's role with its `priority`, which became a
usable signal once the prompt defined it (see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .schema import BBox, Element, ElementType, Layout

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image

#: Standard placements. `parse_ratio` also accepts "W:H" or a float.
PRESETS: dict[str, float] = {
    "1:1": 1.0,
    "4:5": 0.8,
    "1.91:1": 1.91,
    "9:16": 0.5625,
    "16:9": 16 / 9,
}

#: How much each role matters when deciding what to keep.
TYPE_WEIGHT: dict[ElementType, float] = {
    ElementType.CTA: 1.5,
    ElementType.LOGO: 1.4,
    ElementType.PRICE: 1.3,
    ElementType.HEADLINE: 1.2,
    ElementType.PRODUCT: 1.0,
    ElementType.OTHER: 0.4,
    # The model's background box is usually a partial region rather than the canvas,
    # so its extent is ignored entirely -- background is fill, not content.
    ElementType.BACKGROUND: 0.0,
}

#: Types that are useless when partially visible: a clipped logo or a cut-off price
#: reads as a defect, where a cropped product photo still works.
ATOMIC: frozenset[ElementType] = frozenset(
    {ElementType.CTA, ElementType.LOGO, ElementType.PRICE}
)

#: Fraction of an atomic element that must survive for it to count as kept.
ATOMIC_THRESHOLD = 0.98

#: Priority 1 is critical, 3 is optional.
PRIORITY_WEIGHT: dict[int, float] = {1: 3.0, 2: 1.6, 3: 0.8}


def parse_ratio(value: str | float) -> float:
    """`"4:5"`, `"0.8"` or `0.8` -> 0.8 (width / height)."""
    if isinstance(value, (int, float)):
        ratio = float(value)
    elif value in PRESETS:
        ratio = PRESETS[value]
    elif ":" in value:
        w, _, h = value.partition(":")
        ratio = float(w) / float(h)
    else:
        ratio = float(value)
    if ratio <= 0:
        raise ValueError(f"aspect ratio must be positive: {value!r}")
    return ratio


def element_weight(el: Element) -> float:
    return TYPE_WEIGHT.get(el.type, 0.4) * PRIORITY_WEIGHT.get(el.priority, 1.0)


def is_critical(el: Element) -> bool:
    """Elements whose loss makes the creative wrong rather than merely worse.

    Only these force the pad fallback. Judging on *total* retention instead would pad
    almost every ad, because `other` accounts for ~14% of detected elements and one
    decorative box near an edge would veto every crop.
    """
    if el.type is ElementType.BACKGROUND:
        return False
    # Atomic roles are structurally essential, so the model's `must_preserve` guess does
    # not get a vote: it marked 21 of 201 CTAs disposable, and a creative without its
    # call to action has failed regardless of what the annotator thought.
    if el.type in ATOMIC:
        return True
    return el.must_preserve and el.priority == 1


@dataclass(frozen=True)
class Placement:
    """Where an element ended up, and how much of it survived."""

    type: ElementType
    priority: int
    visible_fraction: float
    bbox: BBox | None  # None when the element fell outside the canvas entirely

    @property
    def kept(self) -> bool:
        threshold = ATOMIC_THRESHOLD if self.type in ATOMIC else 0.5
        return self.visible_fraction >= threshold


@dataclass
class ResizePlan:
    """A fully-decided reflow. Inspect it before rendering -- it explains itself."""

    mode: str  # "identity" | "crop" | "pad"
    source_size: tuple[int, int]
    canvas_size: tuple[int, int]
    #: Region of the source that survives, in source pixels.
    crop_box: tuple[int, int, int, int]
    #: Where that region lands on the canvas, in canvas pixels.
    paste_box: tuple[int, int, int, int]
    fill: tuple[int, int, int]
    placements: list[Placement] = field(default_factory=list)
    score: float = 0.0
    max_score: float = 0.0

    @property
    def retention(self) -> float:
        """Weighted fraction of what mattered that survived. 1.0 is perfect."""
        return self.score / self.max_score if self.max_score else 1.0

    @property
    def lost(self) -> list[Placement]:
        return [p for p in self.placements if not p.kept and p.type is not ElementType.BACKGROUND]

    def describe(self) -> str:
        lost = ", ".join(f"{p.type.value}(p{p.priority})" for p in self.lost)
        return (
            f"{self.mode:<8} {self.source_size[0]}x{self.source_size[1]}"
            f" -> {self.canvas_size[0]}x{self.canvas_size[1]}"
            f"  retention {self.retention:.0%}"
            + (f"  LOST: {lost}" if lost else "")
        )


def plan(
    layout: Layout,
    target: str | float,
    *,
    steps: int = 128,
    min_retention: float = 0.999,
) -> ResizePlan:
    """Decide how to reach `target` from `layout`, without rendering anything.

    Tries a crop first, sliding the window along the axis being trimmed and scoring
    every position. If no position keeps everything that matters (`min_retention`),
    falls back to padding, which loses nothing but adds empty space.
    """
    ratio = parse_ratio(target)
    src_w, src_h = layout.source_width, layout.source_height
    src_ratio = src_w / src_h
    content = [e for e in layout.elements if TYPE_WEIGHT.get(e.type, 0.4) > 0]
    max_score = sum(element_weight(e) for e in content)

    if abs(ratio - src_ratio) < 1e-3:
        box = (0, 0, src_w, src_h)
        return ResizePlan(
            mode="identity",
            source_size=(src_w, src_h),
            canvas_size=(src_w, src_h),
            crop_box=box,
            paste_box=box,
            fill=(255, 255, 255),
            placements=_placements(content, box, (src_w, src_h), (0, 0, src_w, src_h), (src_w, src_h)),
            score=max_score,
            max_score=max_score,
        )

    # Widest crop of the requested shape that still fits inside the source.
    if ratio < src_ratio:
        crop_w, crop_h = int(round(src_h * ratio)), src_h
    else:
        crop_w, crop_h = src_w, int(round(src_w / ratio))
    crop_w, crop_h = min(crop_w, src_w), min(crop_h, src_h)

    critical = [e for e in content if is_critical(e)]
    max_critical = sum(element_weight(e) for e in critical)

    best_box, best_score, best_critical, best_raw = None, -1.0, 0.0, -1.0
    span = (src_w - crop_w) or (src_h - crop_h)
    for i in range(steps + 1):
        offset = round(span * i / steps) if span else 0
        box = (
            (offset, 0, offset + crop_w, crop_h)
            if src_w - crop_w
            else (0, offset, crop_w, offset + crop_h)
        )
        visible = {id(e): _visible_fraction(e, box, (src_w, src_h)) for e in content}
        kept = {
            id(e): element_weight(e) * _score_fraction(e, visible[id(e)]) for e in content
        }
        score = sum(kept.values())
        crit = sum(kept[id(e)] for e in critical)
        # Raw visible area breaks ties between windows that score identically. Without
        # it the scan stops at the first window clearing ATOMIC_THRESHOLD, leaving a
        # logo 2% clipped when a fully clean window was available.
        raw = sum(element_weight(e) * visible[id(e)] for e in content)
        if (crit, score, raw) > (best_critical, best_score, best_raw):
            best_box, best_score, best_critical, best_raw = box, score, crit, raw

    assert best_box is not None
    critical_ok = max_critical == 0 or best_critical / max_critical >= min_retention
    if critical_ok:
        return ResizePlan(
            mode="crop",
            source_size=(src_w, src_h),
            canvas_size=(crop_w, crop_h),
            crop_box=best_box,
            paste_box=(0, 0, crop_w, crop_h),
            fill=(255, 255, 255),
            placements=_placements(content, best_box, (crop_w, crop_h), (0, 0, crop_w, crop_h), (src_w, src_h)),
            score=best_score,
            max_score=max_score,
        )

    # Cropping would cost something that matters -- keep everything and pad instead.
    if ratio < src_ratio:
        canvas_w, canvas_h = src_w, int(round(src_w / ratio))
    else:
        canvas_w, canvas_h = int(round(src_h * ratio)), src_h
    canvas_w, canvas_h = max(canvas_w, src_w), max(canvas_h, src_h)
    off_x, off_y = (canvas_w - src_w) // 2, (canvas_h - src_h) // 2
    paste = (off_x, off_y, off_x + src_w, off_y + src_h)

    return ResizePlan(
        mode="pad",
        source_size=(src_w, src_h),
        canvas_size=(canvas_w, canvas_h),
        crop_box=(0, 0, src_w, src_h),
        paste_box=paste,
        fill=_fill_from_layout(layout),
        placements=_placements(
            content, (0, 0, src_w, src_h), (canvas_w, canvas_h), paste, (src_w, src_h)
        ),
        score=max_score,
        max_score=max_score,
    )


def _score_fraction(el: Element, visible: float) -> float:
    """Atomic elements are all-or-nothing; everything else degrades linearly."""
    if el.type in ATOMIC:
        return 1.0 if visible >= ATOMIC_THRESHOLD else 0.0
    return visible


def _visible_fraction(
    el: Element, crop: tuple[int, int, int, int], src: tuple[int, int]
) -> float:
    x1, y1, x2, y2 = el.bbox.to_pixels(*src)
    cx1, cy1, cx2, cy2 = crop
    iw = max(0, min(x2, cx2) - max(x1, cx1))
    ih = max(0, min(y2, cy2) - max(y1, cy1))
    area = (x2 - x1) * (y2 - y1)
    return (iw * ih) / area if area else 0.0


def _placements(
    elements: list[Element],
    crop: tuple[int, int, int, int],
    canvas: tuple[int, int],
    paste: tuple[int, int, int, int],
    source: tuple[int, int],
) -> list[Placement]:
    """Re-express each element in canvas coordinates.

    `source` must be the true source size: normalized boxes are relative to the whole
    image, not to the crop window.
    """
    out: list[Placement] = []

    for el in elements:
        visible = _visible_fraction(el, crop, source)
        x1, y1, x2, y2 = el.bbox.to_pixels(*source)
        # Clip into the crop, then shift into the pasted region.
        cx1, cy1 = max(x1, crop[0]) - crop[0], max(y1, crop[1]) - crop[1]
        cx2, cy2 = min(x2, crop[2]) - crop[0], min(y2, crop[3]) - crop[1]
        if cx2 <= cx1 or cy2 <= cy1:
            out.append(Placement(el.type, el.priority, 0.0, None))
            continue
        bbox = BBox(
            x1=(paste[0] + cx1) / canvas[0],
            y1=(paste[1] + cy1) / canvas[1],
            x2=min((paste[0] + cx2) / canvas[0], 1.0),
            y2=min((paste[1] + cy2) / canvas[1], 1.0),
        )
        out.append(Placement(el.type, el.priority, visible, bbox))
    return out


def _fill_from_layout(layout: Layout) -> tuple[int, int, int]:
    for hexcode in layout.dominant_colors:
        rgb = _hex_to_rgb(hexcode)
        if rgb and sum(rgb) > 90:  # skip near-black; ads pad better on light ground
            return rgb
    return (255, 255, 255)


def _hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    value = value.lstrip("#")
    if len(value) != 6:
        return None
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def render(image: "Image.Image", resize_plan: ResizePlan, *, width: int | None = None) -> "Image.Image":
    """Execute a plan against the real pixels."""
    from PIL import Image as PILImage

    fill = _sampled_fill(image, resize_plan) if resize_plan.mode == "pad" else resize_plan.fill
    canvas = PILImage.new("RGB", resize_plan.canvas_size, fill)
    region = image.convert("RGB").crop(resize_plan.crop_box)

    px, py, px2, py2 = resize_plan.paste_box
    if region.size != (px2 - px, py2 - py):
        region = region.resize((px2 - px, py2 - py), PILImage.LANCZOS)
    canvas.paste(region, (px, py))

    if width and width != canvas.width:
        height = max(1, round(canvas.height * width / canvas.width))
        canvas = canvas.resize((width, height), PILImage.LANCZOS)
    return canvas


def _sampled_fill(image: "Image.Image", resize_plan: ResizePlan) -> tuple[int, int, int]:
    """Median colour of the edges being extended, falling back to the layout's palette.

    Sampling the real border beats the model's dominant_colors most of the time -- the
    padding then reads as a continuation of the ad rather than a coloured bar.
    """
    from PIL import Image as PILImage

    rgb = image.convert("RGB")
    w, h = rgb.size
    vertical = resize_plan.canvas_size[1] > h
    strip_depth = max(1, (h if vertical else w) // 20)
    strips = (
        [rgb.crop((0, 0, w, strip_depth)), rgb.crop((0, h - strip_depth, w, h))]
        if vertical
        else [rgb.crop((0, 0, strip_depth, h)), rgb.crop((w - strip_depth, 0, w, h))]
    )
    pixels = [p for strip in strips for p in strip.resize((8, 8), PILImage.BILINEAR).getdata()]
    if not pixels:
        return resize_plan.fill
    return tuple(sorted(c[i] for c in pixels)[len(pixels) // 2] for i in range(3))  # type: ignore[return-value]


def resize(
    image: "Image.Image",
    layout: Layout,
    target: str | float,
    *,
    width: int | None = None,
) -> tuple["Image.Image", ResizePlan]:
    """Plan and render in one call."""
    resize_plan = plan(layout, target)
    return render(image, resize_plan, width=width), resize_plan

