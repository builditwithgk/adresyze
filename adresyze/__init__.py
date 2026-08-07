"""AdResyze -- layout-aware ad creative resizing."""

from .normalize import NormalizeReport, normalize
from .schema import BBox, Element, ElementType, Layout, RawElement, RawLayout

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "Element",
    "ElementType",
    "Layout",
    "NormalizeReport",
    "RawElement",
    "RawLayout",
    "normalize",
]
