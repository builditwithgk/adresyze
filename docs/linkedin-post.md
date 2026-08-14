# LinkedIn - AdResyze, four months on

Attach `architecture.png`. Links in the first comment, not the body.

---

Four months ago I posted the architecture behind AdResyze.

Six moving parts: Qwen2.5-VL + LoRA, FastAPI, n8n, Supabase, a VPS, Modal.

Today it runs on two. It works better.

Here's what got deleted, and why each deletion was a measurement rather than a preference.

**The LoRA went first.**

I'd fine-tuned Qwen2.5-VL on 289 annotated Indian ads to read ad anatomy - logo, headline, CTA, product, price. Rank 16, 3 epochs, loss went down, shipped it.

Last week I finally ran the test I should have run on day one. Same 25 ads, greedy decoding, adapter on vs adapter off, and a prompt that actually states the output contract:

→ parse rate: 96% adapter, 100% stock
→ elements found per ad: 3.9 adapter, 4.5 stock
→ vocabulary and colour format - the two things the fine-tune existed to fix - 100% on BOTH

The adapter won nothing. It found 13% fewer elements, which for a resizer means a missed call to action.

**Then the root cause, which is the part I enjoyed.**

My training labels said `product` and `#ffffff`. But the base model, given the same prompt that generated those labels, says `car` and `white`.

Those clean labels never came from the model. They came from post-processing in my annotation script.

So I had trained an adapter to produce a vocabulary the prompt never showed it. It learned to strip code fences and rename a JSON key. It never learned what the words meant - it couldn't reproduce the label for an image inside its own training set.

260 samples can teach syntax. They cannot teach semantics.

**Then the infrastructure went.**

n8n was orchestrating a flow with one branch. Supabase was storing outputs nobody queried. FastAPI and the VPS were doing what a single serverless function already does.

Every one of them was solving a problem I did not have yet.

**What's left:**

Stock Qwen2.5-VL reads the ad and returns structured JSON. A deterministic Pillow engine rebuilds it - cropping only when nothing critical is lost, padding when it can't.

Across 302 ads x 4 ratios: 94.1% of reflows lose nothing. It has never lost a logo, a CTA, or a price.

No generative fill. No warped logos. No hallucinated backgrounds. The model decides *what matters*; arithmetic decides *where it goes*.

**The adapter is still published.**

Deleting it would hide the most useful thing I learned this year: an unmeasured component is a liability, however good it felt to build.

Next experiment - retrain with more target modules on the corrected format, and find out whether a fine-tune can beat a good prompt here at all. My guess is no at this data scale.

But that's a guess until it's measured. Which is roughly the whole point.

---

*Note: every number above is conformance - does the output parse, validate, and keep the elements it found. None of it is hand-verified against ground truth. Say so if anyone asks.*
