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

#: `sheet` is first on purpose: it is the file you actually open while reviewing.
#: `image_file` is kept because it is the key that ties a row back to the source ad
#: and to the published dataset.
FIELDS = ["sheet", "image_file", "index", "type", "priority", "bbox", "verdict", "note"]

#: `background` extent is ignored by the reflow engine (TYPE_WEIGHT 0.0) because the
#: model returns partial regions rather than the canvas. Labelling it would measure
#: something no code consumes, so it is left out of the review sheet.
SKIP_TYPES = {"background"}

#: The atomic roles drive every crop-vs-pad decision, so the sample is chosen to cover
#: them rather than to be alphabetically first.
SAMPLE_WEIGHT = {"price": 4.0, "logo": 3.0, "cta": 3.0, "product": 1.5, "headline": 1.0}


def select_ads(candidates: list[tuple[Path, list[str]]], count: int) -> list[Path]:
    """Greedily pick ads that balance coverage of the rarer, more important types.

    Taking the first N files alphabetically produced 25 logos and a single price row --
    no usable statistics on exactly the elements the engine depends on.
    """
    chosen: list[Path] = []
    covered: Counter[str] = Counter()
    pool = list(candidates)

    while pool and len(chosen) < count:
        def gain(item: tuple[Path, list[str]]) -> float:
            return sum(
                SAMPLE_WEIGHT.get(t, 0.5) / (1.0 + covered[t]) for t in set(item[1])
            )

        best = max(pool, key=gain)
        pool.remove(best)
        chosen.append(best[0])
        covered.update(best[1])
    return chosen


def cmd_sheets(args) -> int:
    from PIL import Image

    images, layouts = Path(args.images), Path(args.layouts)
    SHEETS.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[Path, list[str]]] = []
    for image_path in sorted(images.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        layout_path = layouts / f"{image_path.stem}.json"
        if not layout_path.exists():
            continue
        data = json.loads(layout_path.read_text())
        types = [
            e.get("type", "")
            for e in (data.get("elements") or [])
            if e.get("type") not in SKIP_TYPES
        ]
        if types:
            candidates.append((image_path, types))

    if not candidates:
        raise SystemExit(f"no image/layout pairs in {images} + {layouts}")

    if VERDICTS.exists() and not args.overwrite:
        raise SystemExit(f"{VERDICTS} exists -- pass --overwrite to regenerate")

    selected = select_ads(candidates, args.count)

    rows = []
    for image_path in selected:
        layout_path = layouts / f"{image_path.stem}.json"
        layout, _ = normalize(RawLayout.model_validate(json.loads(layout_path.read_text())))
        sheet_name = f"{image_path.stem}.png"
        annotate(Image.open(image_path), layout, show_index=True).save(SHEETS / sheet_name)
        for i, el in enumerate(layout.elements):
            if el.type.value in SKIP_TYPES:
                continue
            rows.append(
                {
                    "sheet": sheet_name,
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

    print(f"{len(selected)} sheets -> {SHEETS}")
    print(f"{len(rows)} rows to review -> {VERDICTS}")
    spread = Counter(r["type"] for r in rows)
    print("\ncoverage:")
    for name, n in spread.most_common():
        print(f"  {name:<12} {n}")
    print(f"\nskipped types: {', '.join(sorted(SKIP_TYPES))} (extent unused by the engine)")
    print("mark each row y (correct) / n (wrong) / m (missed element, add the row yourself)")
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

