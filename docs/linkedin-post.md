# LinkedIn - AdResyze, four months on

Attach `architecture.png`. Links in the first comment, not the body.

---

Four months ago I published a fine-tuned vision model for reading ad layouts.

Last week I finally measured it properly.

It lost. To a prompt.

**The problem:** an ad creative has to exist at 1:1, 4:5, 1.91:1, 9:16. Crop blindly and you cut the call to action. Scale blindly and you warp the logo. So something has to read the ad first - logo here, headline there, CTA bottom right - before anything moves a pixel.

I fine-tuned Qwen2.5-VL on 289 annotated Indian ads for that reading step. LoRA, rank 16, 3 epochs. Loss went down. Shipped it.

**The test I should have run on day one:** same 25 ads, greedy decoding, adapter on vs adapter off, with a prompt that actually states the output contract.

→ parse rate: 96% adapter, 100% stock
→ elements found per ad: 3.9 adapter, 4.5 stock
→ vocabulary and colour format - the two things the fine-tune existed to fix - 100% on BOTH

The adapter won nothing. It found 13% fewer elements, which for a resizer means a missed call to action.

**Then the root cause, which is the part I enjoyed.**

My training labels said `product` and `#ffffff`. But the base model, given the same prompt that generated those labels, says `car` and `white`.

Those clean labels never came from the model. They came from post-processing in my annotation script.

So I had trained an adapter to produce a vocabulary the prompt never showed it. It learned to strip code fences and rename a JSON key. It never learned what the words meant - it could not reproduce the label for an image inside its own training set.

260 samples can teach syntax. They cannot teach semantics.

**What runs now:**

Stock Qwen2.5-VL reads the ad and returns structured JSON. A deterministic Pillow engine rebuilds it - cropping only when nothing critical is lost, padding when it cannot.

Across 302 ads x 4 ratios: 94.1% of reflows lose nothing. It has never lost a logo, a CTA, or a price.

No generative fill. No warped logos. No hallucinated backgrounds. The model decides *what matters*; arithmetic decides *where it goes*.

Fewer moving parts than the version I posted in April, and better output. That trade keeps showing up.

**The adapter is still published.**

Deleting it would hide the most useful thing I learned this year: an unmeasured component is a liability, however good it felt to build.

Next experiment - retrain with more target modules on the corrected format, and find out whether a fine-tune can beat a good prompt here at all. My guess is no at this data scale.

But that is a guess until it is measured. Which is roughly the whole point.

---

*Note: every number above is conformance - does the output parse, validate, and keep the elements it found. None of it is hand-verified against ground truth. Say so if anyone asks.*
