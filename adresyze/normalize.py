"""Turn a raw VLM response into a grounded :class:`Layout`.

Everything the model gets wrong is corrected here, so the resizer never has to
defend itself: compound labels are split, unknown labels fall back to ``other``,
empty boxes are dropped, duplicates are merged, and coordinates are normalized
out of the model's patch grid into 0..1 source-image space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import BBox, Element, ElementType, Layout, RawLayout, canonical_types


@dataclass
class NormalizeReport:
    """What had to be repaired. Worth logging -- it is the honest eval signal."""

    dropped_empty: int = 0
    split_compound: int = 0
    unknown_labels: list[str] = field(default_factory=list)
    merged_duplicates: int = 0
    synthesized_background: bool = False
    ungrounded: bool = False

    @property
    def clean(self) -> bool:
        return not (
            self.dropped_empty
            or self.split_compound
            or self.unknown_labels
            or self.merged_duplicates
            or self.ungrounded
        )


def normalize(
    raw: RawLayout,
    *,
    source_width: int | None = None,
    source_height: int | None = None,
    merge_duplicates: bool = True,
    ensure_background: bool = True,
) -> tuple[Layout, NormalizeReport]:
    """Ground ``raw`` against the source image.

    ``source_width``/``source_height`` override whatever is on ``raw``; pass them
    when you know the real image dimensions (you always do at inference time).
    """
    report = NormalizeReport()

    width = source_width or raw.source_width
    height = source_height or raw.source_height
    if width is None or height is None:
        raise ValueError(
            "source dimensions are required to ground a layout; v1 dataset "
            "records do not carry them (see RawLayout.is_groundable)"
        )

    grid_w, grid_h = raw.implied_grid()
    report.ungrounded = not raw.is_groundable

    elements: list[Element] = []
    for raw_el in raw.elements:
        if raw_el.is_degenerate:
            report.dropped_empty += 1
            continue

        types = canonical_types(raw_el.type)
        if len(types) > 1:
            report.split_compound += 1
        if ElementType.OTHER in types and raw_el.type.strip().lower() != "other":
            report.unknown_labels.append(raw_el.type)

        try:
            box = BBox.from_grid(raw_el.bbox, grid_w, grid_h)
        except ValueError:
            report.dropped_empty += 1
            continue

        priority = min(max(int(raw_el.priority or 1), 1), 3)
        for t in types:
            elements.append(
                Element(
                    type=t,
                    bbox=box,
                    priority=priority,
                    must_preserve=bool(raw_el.must_preserve),
                )
            )

    if merge_duplicates:
        elements, merged = _merge_overlapping(elements)
        report.merged_duplicates = merged

    if ensure_background and not any(
        e.type is ElementType.BACKGROUND for e in elements
    ):
        elements.insert(
            0,
            Element(
                type=ElementType.BACKGROUND,
                bbox=BBox(x1=0, y1=0, x2=1, y2=1),
                priority=3,
                must_preserve=False,
            ),
        )
        report.synthesized_background = True

    layout = Layout(
        elements=elements,
        dominant_colors=raw.dominant_colors,
        source_width=width,
        source_height=height,
        platform_guess=raw.platform_guess,
        image_file=raw.image_file,
    )
    return layout, report


def _merge_overlapping(
    elements: list[Element], iou_threshold: float = 0.85
) -> tuple[list[Element], int]:
    """Collapse same-type boxes the model emitted twice for one thing."""
    kept: list[Element] = []
    merged = 0
    for el in elements:
        for i, existing in enumerate(kept):
            if existing.type is el.type and existing.bbox.iou(el.bbox) >= iou_threshold:
                if el.bbox.area > existing.bbox.area:
                    kept[i] = el
                merged += 1
                break
        else:
            kept.append(el)
    return kept, merged
