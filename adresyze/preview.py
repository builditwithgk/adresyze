"""Before/after strips. The reflow engine is only as good as it looks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

from .resize import ResizePlan, plan, render
from .schema import ElementType, Layout

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image

DEFAULT_TARGETS: tuple[str, ...] = ("1:1", "4:5", "1.91:1", "9:16")

#: Overlay colours, chosen to stay legible on busy creatives.
TYPE_COLOR: dict[ElementType, tuple[int, int, int]] = {
    ElementType.LOGO: (255, 87, 34),
    ElementType.HEADLINE: (33, 150, 243),
    ElementType.CTA: (76, 175, 80),
    ElementType.PRODUCT: (156, 39, 176),
    ElementType.PRICE: (255, 193, 7),
    ElementType.BACKGROUND: (120, 120, 120),
    ElementType.OTHER: (158, 158, 158),
}

PANEL_H = 360
GAP = 14
LABEL_H = 22
BG = (24, 24, 27)
FG = (235, 235, 235)


def _font():
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, 13)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(
    image: "Image.Image", layout: Layout, *, show_index: bool = False
) -> "Image.Image":
    """Draw the detected layout onto a copy of the ad.

    `show_index` prefixes each label with its position in `layout.elements`. Reviewing
    needs it: an ad with two headlines otherwise shows two identical labels and no way
    to tell which one a given verdict row refers to.
    """
    from PIL import ImageDraw

    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font()

    for i, el in enumerate(layout.elements):
        if el.type is ElementType.BACKGROUND:
            continue
        box = el.bbox.to_pixels(out.width, out.height)
        colour = TYPE_COLOR.get(el.type, (200, 200, 200))
        draw.rectangle(box, outline=colour, width=max(2, out.width // 250))

        label = f"#{i} {el.type.value} p{el.priority}" if show_index else f"{el.type.value} p{el.priority}"
        tx, ty = box[0] + 3, box[1] + 2
        # Dark plate behind the text -- labels sit on busy creatives and vanish otherwise.
        left, top, right, bottom = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=colour, font=font)
    return out


def _fit(image: "Image.Image", height: int) -> "Image.Image":
    from PIL import Image as PILImage

    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), PILImage.LANCZOS)


def strip(
    image: "Image.Image",
    layout: Layout,
    targets: Sequence[str] = DEFAULT_TARGETS,
    *,
    show_boxes: bool = True,
) -> tuple["Image.Image", list[ResizePlan]]:
    """One row: the annotated original, then the reflow at each target ratio."""
    from PIL import Image as PILImage, ImageDraw

    panels: list[tuple[str, Image.Image]] = [
        ("source " + f"{image.width}x{image.height}", _fit(annotate(image, layout) if show_boxes else image, PANEL_H))
    ]
    plans: list[ResizePlan] = []

    for target in targets:
        p = plan(layout, target)
        plans.append(p)
        rendered = render(image, p)
        label = f"{target}  {p.mode}  {p.retention:.0%}"
        if p.lost:
            label += "  LOST " + ",".join(sorted({pl.type.value for pl in p.lost}))
        panels.append((label, _fit(rendered, PANEL_H)))

    total_w = sum(p.width for _, p in panels) + GAP * (len(panels) + 1)
    canvas = PILImage.new("RGB", (total_w, PANEL_H + LABEL_H + GAP * 2), BG)
    draw = ImageDraw.Draw(canvas)
    font = _font()

    x = GAP
    for label, panel in panels:
        canvas.paste(panel, (x, GAP + LABEL_H))
        draw.text((x, GAP // 2), label, fill=FG, font=font)
        x += panel.width + GAP
    return canvas, plans


def contact_sheet(rows: Iterable["Image.Image"]) -> "Image.Image":
    """Stack strips into one image."""
    from PIL import Image as PILImage

    rows = list(rows)
    if not rows:
        raise ValueError("no rows to stack")
    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    sheet = PILImage.new("RGB", (width, height), BG)
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet
