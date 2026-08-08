"""Build the v2 annotation set for publication from the cached layouts.

    python -m pipeline.build_v2_dataset --out dist/adresyze-ad-layouts-v2

v2 fixes the two defects v1 shipped with:

* **Grounded coordinates.** v1 stored boxes in the model's internal patch grid with no
  record of the source dimensions, so they could not be placed on the image. v2 stores
  boxes normalized 0..1 against the source image and records `source_width`/`source_height`.
* **Fields that carry signal.** v1's `priority` was 1 for 1121 of 1126 elements and
  `must_preserve` true for 1064 -- both artefacts of a prompt that never defined them.

`aspect_ratio` is computed from the source dimensions, never taken from the model: asked
directly, it answered "1.91:1" for 301 of 302 ads when 219 were actually 1:1.

Images are not redistributed -- they are brand-owned. Annotations only, same as v1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from adresyze import RawLayout, normalize
from adresyze.resize import PRESETS

GENERATOR = {
    # Deliberately explicit: v2 was NOT produced by the adresyze-lora adapter. The
    # adapter lost a measured bake-off against the stock model and is not in the path.
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "adapter": None,
    "prompt": "STRICT_PROMPT (see modal_app.py)",
    "decoding": "greedy",
}


#: How far a real ratio may sit from a preset and still be labelled with it.
RATIO_TOLERANCE = 0.03


def nearest_ratio(width: int, height: int) -> str:
    """Label with a preset only when the ad genuinely is that shape.

    Snapping unconditionally invents labels -- a 600x400 ad (1.50) is not "16:9"
    (1.78), it is simply not a standard placement. `aspect_ratio_value` is the truth;
    this field is a convenience.
    """
    ratio = width / height
    name, value = min(PRESETS.items(), key=lambda kv: abs(kv[1] - ratio))
    return name if abs(value - ratio) / ratio <= RATIO_TOLERANCE else "other"


def build(src: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("*.json"))
    if not paths:
        raise SystemExit(f"no layouts in {src} -- run `modal run modal_app.py::main` first")

    types: Counter[str] = Counter()
    ratios: Counter[str] = Counter()
    written = skipped = 0

    for path in paths:
        data = json.loads(path.read_text())
        if "error" in data:
            skipped += 1
            continue

        layout, report = normalize(RawLayout.model_validate(data))
        width, height = layout.source_width, layout.source_height

        record = {
            "image_file": layout.image_file,
            "source_width": width,
            "source_height": height,
            "aspect_ratio": nearest_ratio(width, height),
            "aspect_ratio_value": round(width / height, 4),
            "platform_guess": layout.platform_guess,
            "dominant_colors": layout.dominant_colors,
            "elements": [
                {
                    "type": el.type.value,
                    # normalized 0..1 against the source image, origin top-left
                    "bbox": [round(v, 4) for v in el.bbox.as_tuple()],
                    "priority": el.priority,
                    "must_preserve": el.must_preserve,
                }
                for el in layout.elements
            ],
            "generator": GENERATOR,
            "repairs": {
                "synthesized_background": report.synthesized_background,
                "dropped_empty": report.dropped_empty,
                "split_compound": report.split_compound,
                "merged_duplicates": report.merged_duplicates,
            },
        }
        (out / path.name).write_text(json.dumps(record, indent=2))
        written += 1
        ratios[record["aspect_ratio"]] += 1
        types.update(el["type"] for el in record["elements"])

    print(f"wrote {written} records to {out}  (skipped {skipped})")
    print("\naspect ratios (computed from source dimensions):")
    for name, n in ratios.most_common():
        print(f"  {name:<10} {n}")
    print("\nelement types:")
    for name, n in types.most_common():
        print(f"  {name:<12} {n}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="samples/layouts")
    ap.add_argument("--out", default="dist/adresyze-ad-layouts-v2")
    args = ap.parse_args()
    build(Path(args.src), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
