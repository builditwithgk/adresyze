"""Check cached layouts against the schema. Local, free, no GPU.

    python -m pipeline.validate_layouts [samples/layouts]

Reports what normalize() had to repair. A high repair rate means the prompt or the
adapter is drifting -- it is the closest thing to an eval signal that does not need
hand-labelled ground truth.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from adresyze import RawLayout, normalize


def main(directory: str = "samples/layouts") -> int:
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        print(f"no layouts in {directory}/ -- run `modal run modal_app.py` first")
        return 1

    repairs = Counter()
    types = Counter()
    ungrounded, errored, failed = [], [], []

    for path in paths:
        data = json.loads(path.read_text())
        if "error" in data:
            errored.append(path.name)
            continue
        try:
            raw = RawLayout.model_validate(data)
            layout, report = normalize(raw)
        except Exception as exc:  # noqa: BLE001
            failed.append((path.name, str(exc)[:100]))
            continue

        if not raw.is_groundable:
            ungrounded.append(path.name)
        for field in ("dropped_empty", "split_compound", "merged_duplicates"):
            repairs[field] += getattr(report, field)
        repairs["synthesized_background"] += report.synthesized_background
        repairs["unknown_labels"] += len(report.unknown_labels)
        types.update(e.type.value for e in layout.elements)

    ok = len(paths) - len(errored) - len(failed)
    print(f"{ok}/{len(paths)} layouts valid")
    if errored:
        print(f"  model errors     : {len(errored)}  {errored[:3]}")
    if failed:
        print(f"  schema failures  : {len(failed)}  {failed[:3]}")
    if ungrounded:
        print(f"  UNGROUNDED       : {len(ungrounded)} -- inference did not record dimensions")

    print("\nrepairs applied:")
    for name, count in repairs.most_common():
        print(f"  {name:<24} {count}")
    print("\nelement types:")
    for name, count in types.most_common():
        print(f"  {name:<24} {count}")

    return 1 if (failed or ungrounded) else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
