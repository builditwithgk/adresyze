"""Layout contract for AdResyze.

Two shapes, deliberately:

``RawLayout``   what the VLM actually emits -- lenient, accepts the mess
                (compound labels, missing background, empty element lists,
                coordinates in the model's internal patch grid).

``Layout``      what the resizer consumes -- strict, coordinates normalized
                to 0..1 against the *source image*, so a layout is
                resolution-independent and can be replayed onto any canvas.

The v1 dataset (builditwithgk/adresyze-ad-layouts) only ever had the raw
shape, and without source dimensions its boxes cannot be grounded --
see ``RawLayout.is_groundable``.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ElementType(str, Enum):
    """Ad anatomy. `price` and `other` appear in the wild but not in the v1 card."""

    LOGO = "logo"
    HEADLINE = "headline"
    CTA = "cta"
    PRODUCT = "product"
    PRICE = "price"
    BACKGROUND = "background"
    OTHER = "other"


#: Labels the model emits that are not themselves valid types. A compound label
#: is split into its parts; the same box is kept for each.
COMPOUND_SEPARATORS = ("|", "/", "+", ",")

#: Free-text labels seen from the VLM mapped onto the canonical vocabulary.
TYPE_ALIASES: dict[str, ElementType] = {
    "call_to_action": ElementType.CTA,
    "call-to-action": ElementType.CTA,
    "button": ElementType.CTA,
    "title": ElementType.HEADLINE,
    "text": ElementType.HEADLINE,
    "brand": ElementType.LOGO,
    "product_image": ElementType.PRODUCT,
    "bg": ElementType.BACKGROUND,
    "offer": ElementType.PRICE,
    "discount": ElementType.PRICE,
}


def canonical_types(label: str) -> list[ElementType]:
    """Map a raw model label onto zero or more canonical element types.

    >>> canonical_types("logo|product")
    [<ElementType.LOGO: 'logo'>, <ElementType.PRODUCT: 'product'>]
    >>> canonical_types("banana")
    [<ElementType.OTHER: 'other'>]
    """
    parts = [label]
    for sep in COMPOUND_SEPARATORS:
        parts = [p for chunk in parts for p in chunk.split(sep)]

    out: list[ElementType] = []
    for part in parts:
        key = part.strip().lower().replace(" ", "_")
        if not key:
            continue
        try:
            resolved = ElementType(key)
        except ValueError:
            resolved = TYPE_ALIASES.get(key, ElementType.OTHER)
        if resolved not in out:
            out.append(resolved)
    return out or [ElementType.OTHER]


# --------------------------------------------------------------------------
# Raw side -- whatever the model gives us
# --------------------------------------------------------------------------


class RawElement(BaseModel):
    """One element as emitted by the VLM. No coordinate space is assumed."""

    model_config = ConfigDict(extra="allow")

    type: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    priority: int = 1
    must_preserve: bool = True

    @field_validator("bbox")
    @classmethod
    def _ordered(cls, v: list[float]) -> list[float]:
        x1, y1, x2, y2 = v
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    @property
    def is_degenerate(self) -> bool:
        x1, y1, x2, y2 = self.bbox
        return x2 - x1 <= 0 or y2 - y1 <= 0


class RawLayout(BaseModel):
    """A full VLM response, plus whatever grounding we managed to capture.

    ``source_width``/``source_height`` are the *original* image dimensions and
    ``grid_width``/``grid_height`` the dimensions of the space the model
    actually emitted coordinates in. v1 dataset records have neither.
    """

    model_config = ConfigDict(extra="allow")

    elements: list[RawElement] = Field(default_factory=list)
    dominant_colors: list[str] = Field(default_factory=list)
    aspect_ratio: str | None = None
    platform_guess: str | None = None
    image_file: str | None = None

    source_width: int | None = None
    source_height: int | None = None
    grid_width: int | None = None
    grid_height: int | None = None

    @field_validator("dominant_colors")
    @classmethod
    def _hex_only(cls, v: list[str]) -> list[str]:
        out = []
        for c in v:
            c = c.strip()
            if not c.startswith("#"):
                c = "#" + c
            if len(c) == 7:
                out.append(c.lower())
        return out

    @property
    def is_groundable(self) -> bool:
        """True when boxes can be placed on the source image."""
        return None not in (
            self.source_width,
            self.source_height,
            self.grid_width,
            self.grid_height,
        )

    def implied_grid(self) -> tuple[float, float]:
        """Best-effort grid extent when it was not recorded.

        Falls back to the largest coordinate seen. This is a guess and is only
        good enough for inspection -- never for eval.
        """
        if self.grid_width and self.grid_height:
            return float(self.grid_width), float(self.grid_height)
        if not self.elements:
            return 1.0, 1.0
        w = max(e.bbox[2] for e in self.elements)
        h = max(e.bbox[3] for e in self.elements)
        return max(w, 1.0), max(h, 1.0)


# --------------------------------------------------------------------------
# Strict side -- what the resizer is allowed to see
# --------------------------------------------------------------------------


class BBox(BaseModel):
    """Normalized box, 0..1, origin top-left."""

    model_config = ConfigDict(frozen=True)

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _non_empty(self) -> "BBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"empty bbox: {self.as_tuple()}")
        return self

    @classmethod
    def from_grid(cls, bbox: Iterable[float], grid_w: float, grid_h: float) -> "BBox":
        x1, y1, x2, y2 = (float(v) for v in bbox)
        clamp = lambda v, hi: min(max(v / hi, 0.0), 1.0)  # noqa: E731
        return cls(
            x1=clamp(x1, grid_w),
            y1=clamp(y1, grid_h),
            x2=clamp(x2, grid_w),
            y2=clamp(y2, grid_h),
        )

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x1 * width),
            round(self.y1 * height),
            round(self.x2 * width),
            round(self.y2 * height),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter / (self.area + other.area - inter)


class Element(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: ElementType
    bbox: BBox


class Layout(BaseModel):
    """A grounded, resolution-independent ad layout."""

    elements: list[Element] = Field(default_factory=list)
    dominant_colors: list[str] = Field(default_factory=list)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    platform_guess: str | None = None
    image_file: str | None = None

    @property
    def aspect_ratio(self) -> float:
        """Measured from the source image, never from the model's own guess."""
        return self.source_width / self.source_height

    def of_type(self, *types: ElementType) -> list[Element]:
        wanted = set(types)
        return [e for e in self.elements if e.type in wanted]

    @property
    def background(self) -> Element | None:
        found = self.of_type(ElementType.BACKGROUND)
        return found[0] if found else None

    @property
    def foreground(self) -> list[Element]:
        """Everything the resizer has to physically place."""
        return [e for e in self.elements if e.type is not ElementType.BACKGROUND]
