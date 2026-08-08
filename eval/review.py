"""Hand-verification harness. This is the only thing here that measures *accuracy*.

Everything else in this repo measures conformance -- does the output parse, validate,
use the agreed vocabulary. None of that says whether the boxes are in the right place.
Only a human looking at ads can say that.

Two steps:

    python -m eval.review sheets --count 50     # render numbered overlays to eval/sheets/
    #  ... look at them, fill in eval/verdicts.csv ...
    python -m eval.review score                 # per-type precision, and where it fails

`sheets` writes one annotated image per ad plus a pre-filled CSV. For each element you
mark `y` (box is right), `n` (wrong or badly placed), or `m` (element that exists in the
ad but the model missed entirely -- add a row). Precision comes from y/(y+n); recall
needs the `m` rows, so add them where you spot them.

50 ads is enough to catch a systematic failure and small enough to finish in a sitting.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from adresyze import RawLayout, normalize
from adresyze.preview import annotate

ROOT = Path(__file__).resolve().parents[1]
SHEETS = ROOT / "eval" / "sheets"
VERDICTS = ROOT / "eval" / "verdicts.csv"

FIELDS = ["image_file", "index", "type", "priority", "bbox", "verdict", "note"]


def cmd_sheets(args) -> int:
    from PIL import Image

    images, layouts = Path(args.images), Path(args.layouts)
    SHEETS.mkdir(parents=True, exist_ok=True)

    pairs = []
    for image_path in sorted(images.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        layout_path = layouts / f"{image_path.stem}.json"
        if layout_path.exists():
            pairs.append((image_path, layout_path))
        if len(pairs) >= args.count:
            break

    if not pairs:
        raise SystemExit(f"no image/layout pairs in {images} + {layouts}")

    if VERDICTS.exists() and not args.overwrite:
        raise SystemExit(f"{VERDICTS} exists -- pass --overwrite to regenerate")

    rows = []
    for image_path, layout_path in pairs:
        layout, _ = normalize(RawLayout.model_validate(json.loads(layout_path.read_text())))
        annotate(Image.open(image_path), layout).save(SHEETS / f"{image_path.stem}.png")
        for i, el in enumerate(layout.elements):
            rows.append(
                {
                    "image_file": image_path.name,
                    "index": i,
                    "type": el.type.value,
                    "priority": el.priority,
                    "bbox": ",".join(f"{v:.3f}" for v in el.bbox.as_tuple()),
                    "verdict": "",
                    "note": "",
                }
            )

    with VERDICTS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(pairs)} sheets -> {SHEETS}")
    print(f"{len(rows)} rows to review -> {VERDICTS}")
    print("\nmark each row y (correct) / n (wrong) / m (missed element, add the row yourself)")
    return 0


def cmd_score(args) -> int:
    if not VERDICTS.exists():
        raise SystemExit(f"{VERDICTS} not found -- run `python -m eval.review sheets` first")

    rows = list(csv.DictReader(VERDICTS.open(encoding="utf-8")))
    marked = [r for r in rows if r["verdict"].strip().lower() in {"y", "n", "m"}]
    if not marked:
        raise SystemExit(f"no verdicts filled in yet ({len(rows)} rows waiting)")

    by_type: dict[str, Counter] = defaultdict(Counter)
    for row in marked:
        by_type[row["type"]][row["verdict"].strip().lower()] += 1

    total = Counter()
    for counts in by_type.values():
        total.update(counts)

    reviewed_images = len({r["image_file"] for r in marked})
    print(f"{len(marked)}/{len(rows)} rows reviewed across {reviewed_images} ads\n")

    header = f"{'type':<12}{'correct':>9}{'wrong':>7}{'missed':>8}{'precision':>11}{'recall':>9}"
    print(header)
    print("-" * len(header))
    for name in sorted(by_type, key=lambda k: -sum(by_type[k].values())):
        c = by_type[name]
        precision = c["y"] / (c["y"] + c["n"]) if (c["y"] + c["n"]) else 0.0
        recall = c["y"] / (c["y"] + c["m"]) if (c["y"] + c["m"]) else 0.0
        print(f"{name:<12}{c['y']:>9}{c['n']:>7}{c['m']:>8}{precision:>10.0%}{recall:>9.0%}")

    precision = total["y"] / (total["y"] + total["n"]) if (total["y"] + total["n"]) else 0.0
    recall = total["y"] / (total["y"] + total["m"]) if (total["y"] + total["m"]) else 0.0
    print("-" * len(header))
    print(f"{'ALL':<12}{total['y']:>9}{total['n']:>7}{total['m']:>8}{precision:>10.0%}{recall:>9.0%}")

    notes = [r for r in marked if r["note"].strip()]
    if notes:
        print(f"\n{len(notes)} notes:")
        for row in notes[:15]:
            print(f"  {row['image_file']} [{row['type']}] {row['note']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sheets", help="render overlays and a blank verdict sheet")
    s.add_argument("--images", default="samples/images")
    s.add_argument("--layouts", default="samples/layouts")
    s.add_argument("--count", type=int, default=50)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_sheets)

    c = sub.add_parser("score", help="score the filled-in verdict sheet")
    c.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
