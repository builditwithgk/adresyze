# AdResyze

Give it one ad creative, get it back correctly laid out in every platform aspect ratio.

A Qwen2.5-VL LoRA reads the anatomy of an ad — logo, headline, CTA, product, price,
background — and a deterministic CV reflow engine rebuilds it at 1:1, 4:5 and 1.91:1
without squashing the logo or cropping the call to action.

| | |
|---|---|
| Adapter | [`builditwithgk/adresyze-lora`](https://huggingface.co/builditwithgk/adresyze-lora) |
| Dataset | [`builditwithgk/adresyze-ad-layouts`](https://huggingface.co/datasets/builditwithgk/adresyze-ad-layouts) |
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| License | Apache-2.0 |

## Status

Early. The layout contract is in place and conformance-tested against all 289 real
annotations; inference and the reflow engine are next.

- [x] Layout schema + normalizer (`adresyze/schema.py`, `adresyze/normalize.py`)
- [ ] Modal inference (`modal_app.py`)
- [ ] Reflow engine (`adresyze/resize.py`)
- [ ] CLI, eval, demo

## How it fits together

```
ad image ──▶ Qwen2.5-VL + LoRA ──▶ RawLayout ──▶ normalize() ──▶ Layout ──▶ resize() ──▶ creatives
             (Modal, GPU)          lenient        repairs        strict,     Pillow,     1:1 4:5
                                                                 0..1 coords deterministic 1.91:1
```

Two layout shapes on purpose. `RawLayout` accepts what the model actually emits —
compound labels like `logo|product`, missing backgrounds, empty boxes. `normalize()`
repairs that and grounds the coordinates. `Layout` is what the reflow engine sees, so
it never has to defend itself against model noise.

Coordinates in a `Layout` are normalized 0..1 against the source image, which makes a
layout resolution-independent: capture once, replay onto any canvas.

## Development

```bash
pip install -e ".[dev]"
python pipeline/fetch_v1_dataset.py   # caches the 289 annotations to .cache/
pytest
```

Ad creatives are brand-owned and are never committed — `samples/images/` is gitignored.

## Notes on the v1 dataset

The v1 annotations were auto-labelled by base Qwen2.5-VL, and the LoRA was then trained
on them. That is self-distillation: it conditions output format and domain vocabulary,
it does not add capability the base model lacked.

Two properties of v1 shape the design here:

- **Boxes are not groundable.** They sit in the model's internal patch grid (588 = 21×28)
  and do not preserve the source aspect ratio — median max-X/max-Y is 0.99 even for
  1.91:1 banners, and the most common `background` box is a square `(0,0,588,588)`.
  No source dimensions were recorded, so the mapping cannot be recovered. `infer.py`
  therefore captures the true image dimensions and normalizes at inference time.
- **`priority` and `must_preserve` carry no signal.** 1121 of 1126 elements have
  `priority: 1`, and 1064 of 1126 are `must_preserve: true`. Reflow rules key off
  element `type` instead.
