# AdResyze

Give it one ad creative, get it back correctly laid out in every platform aspect ratio.

Qwen2.5-VL reads the anatomy of an ad - logo, headline, CTA, product, price,
background - and a deterministic CV reflow engine rebuilds it at 1:1, 4:5 and 1.91:1
without squashing the logo or cropping the call to action.

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Adapter | [`builditwithgk/adresyze-lora`](https://huggingface.co/builditwithgk/adresyze-lora) (published, not used - see below) |
| Dataset | [`builditwithgk/adresyze-ad-layouts`](https://huggingface.co/datasets/builditwithgk/adresyze-ad-layouts) |
| Inference | Modal, L4 GPU |
| License | Apache-2.0 |

## Status

- [x] Layout schema + normalizer (`adresyze/schema.py`, `adresyze/normalize.py`)
- [x] Modal inference (`modal_app.py`)
- [x] Reflow engine + CLI (`adresyze/resize.py`, `adresyze/cli.py`)
- [x] v2 annotation set (`pipeline/build_v2_dataset.py`)
- [ ] Hand-labelled eval
- [ ] Live demo

## Results

Across the full corpus — 302 ads x 4 target ratios = 1208 reflows:

| | |
|---|---|
| reflows losing nothing | **94.1%** |
| mean retention | **99.0%** |
| logos, CTAs or prices lost | **0** |

```bash
adresyze report                              # the table above
adresyze preview --count 6                   # before/after contact sheet
adresyze resize ad.jpg --to 1:1,4:5,9:16     # one ad
```

The engine crops when a crop window can keep everything structurally essential, and
pads otherwise. Atomic roles — logo, CTA, price — score all-or-nothing, because a
clipped call to action is worse than none, and they override the model's
`must_preserve` guess, which marked 21 of 201 CTAs disposable.

Padding dominates the extreme ratios: 212 of 302 ads pad for 9:16. That is correct —
a square ad cannot become 9:16 without either losing content or adding space — but it
is the honest ceiling of a CV-only resizer, and the point where generative outpainting
would start to earn its cost.

## The LoRA lost to its own prompt

The adapter is published and it works, in the narrow sense that it is correctly
attached and does change the output. It is not what AdResyze runs on.

The model card's prompt never states the element vocabulary, so Qwen2.5-VL answers with
the objects it can see - `car`, `building` - and with colour *names* instead of hex.
The v1 dataset's clean `product` / `#ffffff` labels came from post-processing in the
annotation script, not from the model, so the LoRA was trained to produce a vocabulary
the prompt never showed it. It learned the syntax (it strips code fences and renames
`bbox_2d` to `bbox`) and none of the semantics. It cannot reproduce the training label
for an image in its own training set.

Writing the contract into the prompt fixes that - for the *base* model. Measured over
25 ads, greedy decoding, the same `STRICT_PROMPT` in both arms:

| metric | LoRA | base | winner |
|---|---|---|---|
| parse rate % | 96.0 | **100.0** | base |
| schema-valid % | 96.0 | **100.0** | base |
| vocabulary conformance % | 100.0 | 100.0 | tie |
| hex colour % | 100.0 | 100.0 | tie |
| aspect ratio in enum % | 96.0 | **100.0** | base |
| platform in enum % | 96.0 | **100.0** | base |
| has background % | 80.0 | **88.0** | base |
| mean elements found | 3.9 | **4.5** | base |
| mean repairs needed | 0.0 | 0.0 | tie |

The two metrics the fine-tune existed to fix - vocabulary and hex colours - reach 100%
on the stock model once the prompt says what is wanted. The adapter wins nothing and
finds 13% fewer elements, which for a resizer means a missed CTA.

So `USE_ADAPTER = False`. Reproduce the table with:

```bash
# set USE_ADAPTER = True in modal_app.py first
modal run modal_app.py::bakeoff --count 25
python -m pipeline.score_bakeoff
```

This scores *conformance*, not accuracy - neither arm is checked against hand-labelled
ground truth. Conformance is simply where the adapter was supposed to help.

## How it fits together

```
ad image --> Qwen2.5-VL --> RawLayout --> normalize() --> Layout --> resize() --> creatives
             Modal, GPU     lenient       repairs        strict,     Pillow,      1:1 4:5
                                                         0..1 coords deterministic 1.91:1
```

Two layout shapes on purpose. `RawLayout` accepts what the model actually emits -
compound labels like `logo|product`, missing backgrounds, empty boxes. `normalize()`
repairs that and grounds the coordinates. `Layout` is what the reflow engine sees, so
it never has to defend itself against model noise.

Coordinates in a `Layout` are normalized 0..1 against the source image, which makes a
layout resolution-independent: capture once, replay onto any canvas.

## Usage

```bash
pip install -e ".[dev,remote]"
modal setup

modal run modal_app.py::warmup            # once - pull weights into a Modal volume
modal run modal_app.py::main              # samples/images/ -> samples/layouts/
python -m pipeline.validate_layouts       # local, free: what needed repairing
```

Batch once, then develop the reflow engine offline against the cached layouts.

Ad creatives are brand-owned and are never committed - `samples/images/` is gitignored.

## Notes on the v1 dataset

The v1 annotations were auto-labelled by base Qwen2.5-VL, and the LoRA was then trained
on them. That is self-distillation: it conditions output format, it does not add
capability the base model lacked.

Two properties of v1 shape the design here:

- **Boxes are not groundable.** They sit in the model's internal patch grid (588 = 21x28)
  and do not preserve the source aspect ratio - median max-X/max-Y is 0.99 even for
  1.91:1 banners, and the most common `background` box is a square `(0,0,588,588)`.
  No source dimensions were recorded, so the mapping cannot be recovered. Inference
  now captures the true image dimensions and the emitted grid, and normalizes against
  them.
- **`priority` and `must_preserve` carry no signal.** 1121 of 1126 elements have
  `priority: 1`, and 1064 of 1126 are `must_preserve: true`. Reflow rules key off
  element `type` instead.
