"""Command line interface.

    adresyze resize ad.jpg --layout ad.json --to 1:1,4:5 --out out/
    adresyze preview --count 8 --out docs/preview.png
    adresyze report

Layouts come from `modal run modal_app.py::main`. Resizing itself never touches a GPU
or the network -- it is Pillow and arithmetic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .normalize import normalize
from .preview import DEFAULT_TARGETS, contact_sheet, strip
from .resize import plan, render
from .schema import RawLayout


def _load_layout(path: Path):
    raw = RawLayout.model_validate(json.loads(path.read_text()))
    layout, _ = normalize(raw)
    return layout


def _layout_for(image: Path, explicit: str | None, layouts_dir: Path):
    candidate = Path(explicit) if explicit else layouts_dir / f"{image.stem}.json"
    if not candidate.exists():
        sys.exit(
            f"no layout for {image.name} at {candidate}\n"
            "generate one with: modal run modal_app.py::main"
        )
    return _load_layout(candidate)


def cmd_resize(args) -> int:
    from PIL import Image

    image_path = Path(args.image)
    layout = _layout_for(image_path, args.layout, Path(args.layouts))
    image = Image.open(image_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in [t.strip() for t in args.to.split(",") if t.strip()]:
        p = plan(layout, target)
        rendered = render(image, p, width=args.width)
        name = f"{image_path.stem}_{target.replace(':', 'x')}.jpg"
        rendered.save(out_dir / name, quality=92)
        print(f"{name:<40} {p.describe()}")
    return 0


def cmd_preview(args) -> int:
    from PIL import Image

    images_dir, layouts_dir = Path(args.images), Path(args.layouts)
    pairs = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        layout_path = layouts_dir / f"{image_path.stem}.json"
        if layout_path.exists():
            pairs.append((image_path, layout_path))
        if len(pairs) >= args.count:
            break

    if not pairs:
        sys.exit(f"no image/layout pairs in {images_dir} + {layouts_dir}")

    rows = []
    for image_path, layout_path in pairs:
        layout = _load_layout(layout_path)
        row, plans = strip(Image.open(image_path), layout, DEFAULT_TARGETS)
        rows.append(row)
        print(f"{image_path.name}")
        for target, p in zip(DEFAULT_TARGETS, plans):
            print(f"   {target:<8} {p.describe()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(rows).save(out)
    print(f"\nwrote {out}")
    return 0


def cmd_report(args) -> int:
    """How the engine behaves across the whole cached corpus."""
    from collections import Counter

    layouts = sorted(Path(args.layouts).glob("*.json"))
    if not layouts:
        sys.exit(f"no layouts in {args.layouts}")

    modes: Counter[str] = Counter()
    lost: Counter[str] = Counter()
    retention: list[float] = []
    perfect = 0
    total = 0

    for path in layouts:
        layout = _load_layout(path)
        for target in DEFAULT_TARGETS:
            p = plan(layout, target)
            modes[f"{target} {p.mode}"] += 1
            retention.append(p.retention)
            total += 1
            if p.lost:
                for placement in p.lost:
                    lost[placement.type.value] += 1
            else:
                perfect += 1

    print(f"{len(layouts)} ads x {len(DEFAULT_TARGETS)} targets = {total} reflows\n")
    print(f"no element lost : {perfect}/{total} ({100 * perfect / total:.1f}%)")
    print(f"mean retention  : {sum(retention) / len(retention):.1%}\n")
    print("mode by target:")
    for key, n in sorted(modes.items()):
        print(f"  {key:<20} {n}")
    if lost:
        print("\nelements lost (by type):")
        for name, n in lost.most_common():
            print(f"  {name:<12} {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="adresyze", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("resize", help="reflow one ad into target ratios")
    r.add_argument("image")
    r.add_argument("--layout", help="layout JSON (default: --layouts/<stem>.json)")
    r.add_argument("--layouts", default="samples/layouts")
    r.add_argument("--to", default="1:1,4:5,1.91:1")
    r.add_argument("--width", type=int, default=None)
    r.add_argument("--out", default="out")
    r.set_defaults(func=cmd_resize)

    p = sub.add_parser("preview", help="before/after contact sheet")
    p.add_argument("--images", default="samples/images")
    p.add_argument("--layouts", default="samples/layouts")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--out", default="docs/preview.png")
    p.set_defaults(func=cmd_preview)

    q = sub.add_parser("report", help="engine behaviour across the whole corpus")
    q.add_argument("--layouts", default="samples/layouts")
    q.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
