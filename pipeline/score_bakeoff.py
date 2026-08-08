"""Score a LoRA-vs-base bake-off. Local, free.

    python -m pipeline.score_bakeoff docs/evidence/bakeoff.json

Measures what actually matters downstream -- can the reflow engine consume this?
There is no hand-labelled ground truth, so this scores *conformance*, not accuracy:
does it parse, does it validate, does it use the agreed vocabulary and hex colours,
and does it find the elements an ad actually has.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from adresyze import RawLayout, normalize
from adresyze.schema import ElementType, canonical_types

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
RATIOS = {"1:1", "4:5", "1.91:1", "9:16"}
PLATFORMS = {"instagram", "facebook", "linkedin", "google_display", "other"}


def score_one(text: str, meta: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modal_app import extract_json  # noqa: PLC0415

    parsed = extract_json(text)
    if parsed is None:
        return {"parsed": False}

    raw_types = [e.get("type", "") for e in parsed.get("elements") or []]
    colors = parsed.get("dominant_colors") or []

    out = {
        "parsed": True,
        "n_elements": len(raw_types),
        "in_vocabulary": sum(
            canonical_types(t) != [ElementType.OTHER] or t.lower() == "other"
            for t in raw_types
        ),
        "n_types": len(raw_types),
        "hex_colors": sum(bool(HEX.match(str(c))) for c in colors),
        "n_colors": len(colors),
        "ratio_ok": parsed.get("aspect_ratio") in RATIOS,
        "platform_ok": parsed.get("platform_guess") in PLATFORMS,
        "has_background": "background" in raw_types,
    }

    try:
        raw = RawLayout.model_validate({**parsed, **meta})
        layout, report = normalize(raw)
        out["validates"] = True
        out["repairs"] = (
            report.dropped_empty + report.split_compound + report.merged_duplicates
        )
        out["final_elements"] = len(layout.elements)
    except Exception:  # noqa: BLE001
        out["validates"] = False
    return out


def main(path: str = "docs/evidence/bakeoff.json") -> int:
    rows = json.loads(Path(path).read_text())
    totals = {"lora": [], "base": []}

    for row in rows:
        meta = {k: row[k] for k in ("image_file", "source_width", "source_height", "grid_width", "grid_height")}
        for arm in ("lora", "base"):
            totals[arm].append(score_one(row[arm], meta))

    n = len(rows)
    print(f"{n} images, strict prompt, greedy decoding\n")
    header = f"{'metric':<26}{'LoRA':>10}{'base':>10}   winner"
    print(header)
    print("-" * len(header))

    def pct(arm, key):
        vals = [s.get(key) for s in totals[arm]]
        return 100 * sum(bool(v) for v in vals) / n

    def ratio(arm, num, den):
        a = sum(s.get(num, 0) for s in totals[arm])
        b = sum(s.get(den, 0) for s in totals[arm])
        return 100 * a / b if b else 0.0

    def mean(arm, key):
        vals = [s[key] for s in totals[arm] if key in s]
        return sum(vals) / len(vals) if vals else 0.0

    rowspec = [
        ("parse rate %", lambda a: pct(a, "parsed"), "high"),
        ("schema-valid %", lambda a: pct(a, "validates"), "high"),
        ("vocabulary conformance %", lambda a: ratio(a, "in_vocabulary", "n_types"), "high"),
        ("hex colour %", lambda a: ratio(a, "hex_colors", "n_colors"), "high"),
        ("aspect ratio in enum %", lambda a: pct(a, "ratio_ok"), "high"),
        ("platform in enum %", lambda a: pct(a, "platform_ok"), "high"),
        ("has background %", lambda a: pct(a, "has_background"), "high"),
        ("mean elements found", lambda a: mean(a, "n_elements"), "high"),
        ("mean repairs needed", lambda a: mean(a, "repairs"), "low"),
    ]

    for label, fn, direction in rowspec:
        lora, base = fn("lora"), fn("base")
        if abs(lora - base) < 1e-9:
            winner = "tie"
        elif (lora > base) == (direction == "high"):
            winner = "LoRA"
        else:
            winner = "base"
        print(f"{label:<26}{lora:>10.1f}{base:>10.1f}   {winner}")

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))

