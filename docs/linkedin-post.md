# LinkedIn - AdResyze

Attach `architecture.png`. Demo + repo links in the first comment.

---

Every component in a production pipeline should have to defend its place.

AdResyze shipped in April with a fine-tuned Qwen2.5-VL reading ad layouts - logo, headline, CTA, product, price - so that resizing a creative becomes composition rather than cropping.

This quarter I put that adapter through a controlled bake-off against the stock model. Same ads, same decoding, one variable.

Stock won. 100% vs 96% parse rate. 4.5 vs 3.9 elements found per ad.

So the fine-tune came out of the pipeline. A prompt that states the output contract does the job, and the system got smaller.

What ships today: the model decides what matters, arithmetic decides where it goes. No generative fill. No warped logos. No hallucinated backgrounds.

Across 302 ads and 4 aspect ratios - 94.1% of reflows lose nothing, and it has never dropped a logo, a CTA or a price.

The adapter and the dataset stay public. A measured negative result is worth more than the artefact it retired.

Demo is live and takes one upload. Bring a creative you think will break it - the failure cases are the interesting ones.

---

*Every number is conformance: does the output parse, validate, and keep what it found. Not hand-verified accuracy. Say so if asked.*
