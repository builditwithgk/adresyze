---
license: cc-by-4.0
task_categories:
  - object-detection
  - image-to-text
language:
  - en
tags:
  - advertising
  - layout-analysis
  - creative-automation
  - qwen2-vl
size_categories:
  - n<1K
---

# AdResyze Ad Layout Annotations (v2)

Layout annotations for 302 Indian brand advertisements: which region of an ad is the
logo, the headline, the call to action, the product, the price, the background — and
where each one sits.

Built for [AdResyze](https://github.com/builditwithgk/adresyze), which uses these
layouts to reflow a single creative into every platform aspect ratio without squashing
the logo or cropping the call to action.

**Try it:** [builditwithgk--adresyze-ui.modal.run](https://builditwithgk--adresyze-ui.modal.run)
&nbsp;·&nbsp; **Code:** [github.com/builditwithgk/adresyze](https://github.com/builditwithgk/adresyze)

**Annotations only.** The advertisements themselves are brand-owned and are not
redistributed.

## Format

One JSON file per ad.

```json
{
  "image_file": "121101721_337678557303047_605074869721584267_n.jpg",
  "source_width": 600,
  "source_height": 400,
  "aspect_ratio": "other",
  "aspect_ratio_value": 1.5,
  "platform_guess": "other",
  "dominant_colors": ["#ffffff", "#000000"],
  "elements": [
    {
      "type": "product",
      "bbox": [0.25, 0.3878, 0.6599, 0.7398],
      "priority": 1,
      "must_preserve": true
    }
  ],
  "generator": {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "adapter": null}
}
```

| field | meaning |
|---|---|
| `bbox` | `[x1, y1, x2, y2]`, **normalized 0..1** against the source image, origin top-left |
| `type` | `logo`, `headline`, `cta`, `product`, `price`, `background`, `other` |
| `priority` | 1 critical, 2 important, 3 optional |
| `must_preserve` | whether the element should survive a crop |
| `aspect_ratio` | nearest standard placement, or `other` when the ad is not within 3% of one |
| `aspect_ratio_value` | the true `width / height` — this is the authoritative value |

Because boxes are normalized against `source_width` × `source_height`, they can be
placed on the image at any resolution.

## Contents

302 ads, 1493 elements.

| type | count | | ratio | count |
|---|---|---|---|---|
| headline | 358 | | 1:1 | 198 |
| background | 306 | | other | 33 |
| other | 210 | | 4:5 | 30 |
| product | 202 | | 9:16 | 27 |
| cta | 201 | | 1.91:1 | 11 |
| logo | 176 | | 16:9 | 3 |
| price | 54 | | | |

## How it was made

Ads collected from the Meta Ad Library (India) on discount and offer search terms.
Annotated with **stock `Qwen/Qwen2.5-VL-7B-Instruct`**, greedy decoding, using a prompt
that states the element vocabulary and value formats explicitly. Every record was then
validated against a schema; 302 of 302 passed.

Note the `generator` block on every record: v2 was **not** produced by the
[adresyze-lora](https://huggingface.co/builditwithgk/adresyze-lora) adapter. Measured
over 25 ads with an identical prompt, the adapter lost or tied the stock model on every
conformance metric — 96% vs 100% parse rate, 3.9 vs 4.5 elements found per ad — so it
is not in the pipeline. The adapter remains published as provenance.

## Changes from v1

- **Boxes are grounded.** v1 stored coordinates in the model's internal patch grid and
  recorded no source dimensions, so they could not be mapped onto the image. v2 stores
  normalized coordinates plus `source_width` / `source_height`.
- **`priority` and `must_preserve` vary.** In v1 they were effectively constant (1121
  of 1126 elements had `priority: 1`) because nothing defined them. Defining them in
  the prompt made them informative.
- **`aspect_ratio` is computed from the image**, not guessed by the model.
- 302 records instead of 289, and 1493 elements instead of 1126.

## Limitations

- **Machine-generated, not human-verified.** These are model outputs validated for
  schema conformance, not ground truth. No hand-labelled accuracy figure is published.
- **~3% of raw boxes extended past the model's own coordinate grid** and were clamped.
- **`other` is 14% of elements** — regions the model declined to classify.
- Indian market, discount and offer creatives. Do not assume it generalizes.

## Licence

CC BY 4.0. Attribution requested.
