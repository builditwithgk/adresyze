"""Download the v1 annotation set from the Hugging Face Hub into .cache/.

    python pipeline/fetch_v1_dataset.py

v1 is annotations only -- the ad images themselves are brand-owned and are not
redistributed. The boxes in v1 are *not* groundable (no source dimensions were
recorded); this fetch exists so the schema can be conformance-tested against
real model output, not so the boxes can be trusted.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = "builditwithgk/adresyze-ad-layouts"
CACHE = Path(__file__).resolve().parents[1] / ".cache" / "v1"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "adresyze"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    tree = json.loads(_get(f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=1"))
    paths = [e["path"] for e in tree if e["type"] == "file" and e["path"].endswith(".json")]
    print(f"{len(paths)} annotation files in {REPO}")

    for i, path in enumerate(paths, 1):
        dest = CACHE / path.replace("/", "__")
        if not dest.exists():
            dest.write_bytes(_get(f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"))
        if i % 50 == 0:
            print(f"  {i}/{len(paths)}")

    print(f"cached -> {CACHE}")
    return CACHE


if __name__ == "__main__":
    sys.exit(0 if fetch() else 1)
