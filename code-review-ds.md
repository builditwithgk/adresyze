# AdResyze — Code Review

**Reviewer:** Cline + DeepSeek
**Date:** 2026-08-15
**Scope:** `adresyze/` (schema, normalize, resize, cli, preview), `modal_app.py`, `pipeline/`, `eval/`, `training/`, `tests/`, CI workflow, docs.
**Method:** full source read + live verification (tests, corpus validation, report regeneration).

## Rating: 8.5 / 10

| Dimension | Score | Notes |
|---|---|---|
| Architecture & design | 9.0 | Two-shape layout contract is exactly the right call |
| Code quality & correctness | 7.5 | Clean, typed, but real edge cases (below) |
| Testing | 8.5 | 42/42 pass, CI on 3 Python versions — but no accuracy eval yet |
| Documentation & honesty | 10.0 | Best-in-class for a portfolio project |
| Reproducibility & ops | 7.0 | Headline numbers not reproducible from a fresh clone |

## Verification performed (no code changed)

- `pytest`: **42/42 passed** (4.4 s)
- `python -m pipeline.validate_layouts`: **302/302 layouts valid**, 14 synthesized backgrounds, 2 unknown labels, 0 schema failures
- `adresyze report`: **94.1% reflows lose nothing, 99.0% mean retention** — matches the README table exactly

## Top 5 observations

### 1. (Biggest gap) The hand-labelled accuracy eval exists but has never been run — 0 of 263 verdict rows marked

Everything in this repo measures *conformance* (parses, validates, uses the vocabulary) — nothing yet measures *accuracy* (are the boxes in the right place?). The harness in `eval/review.py` is genuinely well-designed (stratified sampling weighted toward the atomic roles, numbered overlays, per-type precision/recall), but `eval/verdicts.csv` has 263 generated rows and **not one verdict**. The README status list confirms it. This is the single highest-value next step and the one thing standing between the project and a verifiable accuracy claim.

### 2. (Standout strength) Evidence-driven engineering — the LoRA was removed on a *measured* bake-off

This is the credibility anchor of the whole repo. The author trained a LoRA, ran a controlled bake-off (same ads, same strict prompt, one variable: adapter on/off), found the stock model won (100% vs 96% parse rate; 4.5 vs 3.9 elements found), **published the negative result**, and dropped the adapter from the pipeline. The v1 dataset defect (boxes in an ungroundable patch grid, no source dims) was also diagnosed and fixed properly — inference now records both source and grid dimensions. Rare and very credible.

### 3. (Correctness) Edge cases in `resize.plan()` and the web API

- `plan()` has no lower bound on crop dimensions: a very small target ratio (e.g. `0.0001:1`) makes `crop_w`/`crop_h` round to **0**, yielding a zero-sized canvas that flows into `render()`.
- `modal_app.py` `web()` does `base64.b64decode(payload["image_b64"])` with no key/type validation — a malformed POST raises an unhandled exception (500) instead of a clean error.
- Cosmetic: the Gradio summary f-string prints `background synthesized_` (trailing underscore, line ~391).
- Minor: `score_bakeoff.py` mutates `sys.path` and re-imports `modal_app` inside `score_one()` per row — fragile.

### 4. (Reproducibility) The headline numbers cannot be reproduced from a fresh clone

The 302-layout corpus is gitignored (`samples/layouts/`, `samples/images/`), so `adresyze report`, `validate_layouts`, and `preview` all require the local cache — a fresh clone must either re-run GPU inference (Modal, ~$ and credentials) or fetch the published v2 dataset, whose format is normalized/rounded and therefore won't reproduce the exact report numbers. Checking the report/validate output in as a committed artifact (as is already done for `bakeoff.json` in `docs/evidence/`) would close this loop.

### 5. (Hygiene) Small but real issues

- `test_extract_json.py` imports the **entire** `modal_app.py` (import-time `modal.App(...)` construction); if `modal` isn't installed the whole file silently skips — CI only runs it because `[remote]` is installed.
- No lint/type-check config in `pyproject.toml`; the CI workflow runs pytest only (no `ruff`/`mypy` guardrails) despite the codebase otherwise being disciplined about types.
- `RawLayout`/`RawElement` use `extra="allow"` with implicit field pass-through — fine for model tolerance, but it means typos in cached records are silently absorbed rather than flagged.
- A few important magic numbers are documented in prose but not centralized (`ATOMIC_THRESHOLD`, `RATIO_TOLERANCE`, weights) — minor, they're at least annotated.

## What is genuinely good

- **Two-shape schema** (`RawLayout` → `normalize()` → `Layout`): the resizer never defends itself against model noise. Compound labels split, unknown labels fall back, empty boxes dropped, coordinates grounded to 0..1.
- **Deterministic, self-explaining reflow**: `ResizePlan.describe()`, atomic all-or-nothing scoring for CTA/logo/price, background weight 0.0, crop-vs-pad decision with `min_retention`, and a tie-breaker that prefers the cleanest window.
- **Meaningful tests**: pure-geometry, no model mocking; a conformance suite over the *real* v1 dataset; `extract_json` tested independently of Modal.
- **Honest docs**: the README and dataset card explicitly state that conformance ≠ accuracy, that data is India-market discount creatives, that `other` is 14% of elements, and that the LoRA is provenance only.

## Verdict

A strong **8.5/10**. For a portfolio project this is unusually honest and well-engineered — the architecture is right, the claims are measured and verifiable, and the tests actually pass. The one thing separating it from a 9+ is completing the hand-labelled eval and publishing real precision/recall numbers.
