"""Qwen2.5-VL + AdResyze LoRA on Modal.

One function, three ways to call it:

    modal run modal_app.py::warmup          # once -- pull ~17 GB into the volume
    modal run modal_app.py                  # batch: samples/images/ -> samples/layouts/
    modal deploy modal_app.py               # live HTTPS endpoint for the demo

The container captures the *true* source dimensions and the resized grid the model
actually emitted coordinates in, and returns both. That is the fix for the v1 defect:
without them a bbox cannot be placed on the image (see README).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER = "builditwithgk/adresyze-lora"

# Verbatim from the model card. The LoRA was trained against this exact wording --
# paraphrasing it measurably degrades a rank-16 adapter, so do not "improve" it.
PROMPT = (
    "Analyze this advertisement image. Return ONLY a JSON object with elements array "
    "containing type, bbox [x1, y1, x2, y2], priority, and must_preserve fields, plus "
    "dominant_colors, aspect_ratio, and platform_guess."
)

# Caps the visual token count. Qwen2.5-VL resizes to multiples of 28; ~1 MP keeps the
# sequence short enough to fit a 24 GB card alongside fp16 weights.
MAX_PIXELS = 1024 * 1024
MIN_PIXELS = 256 * 28 * 28

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.56.1",
        "accelerate",
        "peft",
        "qwen-vl-utils",
        "pillow",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/weights"})
)

weights = modal.Volume.from_name("adresyze-weights", create_if_missing=True)
app = modal.App("adresyze", image=image)


@app.function(volumes={"/weights": weights}, timeout=3600)
def warmup() -> dict:
    """Pull base model + adapter into the volume. Run once; costs a few minutes."""
    from huggingface_hub import snapshot_download

    base = snapshot_download(BASE_MODEL)
    adapter = snapshot_download(ADAPTER)
    weights.commit()
    return {"base": base, "adapter": adapter}


@app.cls(
    gpu="L4",  # 24 GB; bump to "A10G" or "L40S" if you hit OOM on large creatives
    volumes={"/weights": weights},
    timeout=900,
    scaledown_window=120,
)
class LayoutModel:
    """Loaded once per container, reused across images in a batch."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            BASE_MODEL, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER).eval()

    def _generate(self, pil_image, max_new_tokens: int, temperature: float) -> tuple[str, tuple[int, int]]:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_image], return_tensors="pt"
        ).to("cuda")

        # The coordinate space the model emits in: patch grid -> pixels.
        # This is exactly what v1 failed to record.
        t, gh, gw = (int(v) for v in inputs["image_grid_thw"][0])
        patch = getattr(self.processor.image_processor, "patch_size", 14)
        grid = (gw * patch, gh * patch)

        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature or None,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return decoded, grid

    @modal.method()
    def infer(self, image_bytes: bytes, image_file: str | None = None) -> dict:
        """Return a RawLayout-shaped dict, grounded with source and grid dimensions."""
        import io

        from PIL import Image

        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        source_w, source_h = pil.size

        # One greedy pass; one warmer retry if the model emits unparseable JSON.
        last_error = None
        for attempt, temperature in enumerate((0.0, 0.4)):
            decoded, grid = self._generate(pil, max_new_tokens=1024, temperature=temperature)
            parsed = extract_json(decoded)
            if parsed is not None:
                parsed.update(
                    image_file=image_file,
                    source_width=source_w,
                    source_height=source_h,
                    grid_width=grid[0],
                    grid_height=grid[1],
                    _attempts=attempt + 1,
                )
                return parsed
            last_error = decoded[:400]

        return {
            "error": "unparseable model output",
            "raw_output": last_error,
            "image_file": image_file,
            "source_width": source_w,
            "source_height": source_h,
        }

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(self, payload: dict) -> dict:
        """POST {"image_b64": "..."} -- used by the gradio demo."""
        return self.infer.local(base64.b64decode(payload["image_b64"]), payload.get("image_file"))


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of model output.

    VLMs wrap JSON in prose or fences and occasionally trail a comma; be forgiving
    here so a good layout is not thrown away over punctuation.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            candidate = text[start : i + 1]
            for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
                try:
                    parsed = json.loads(attempt)
                except json.JSONDecodeError:
                    continue
                return parsed if isinstance(parsed, dict) else None
            return None
    return None


@app.local_entrypoint()
def main(images: str = "samples/images", out: str = "samples/layouts", overwrite: bool = False):
    """Batch every local image through the model and cache the layouts.

    Run this once, then develop the reflow engine offline against the cache.
    """
    src, dst = Path(images), Path(out)
    dst.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ) if src.exists() else []

    if not paths:
        print(f"no images in {src}/ -- drop 10-15 ad creatives there first")
        return

    todo = [p for p in paths if overwrite or not (dst / f"{p.stem}.json").exists()]
    print(f"{len(paths)} images, {len(todo)} to infer")
    if not todo:
        return

    model = LayoutModel()
    payloads = [(p.read_bytes(), p.name) for p in todo]
    failures = 0

    for path, result in zip(todo, model.infer.starmap(payloads)):
        (dst / f"{path.stem}.json").write_text(json.dumps(result, indent=2))
        if "error" in result:
            failures += 1
            print(f"  FAIL {path.name}: {result['error']}")
        else:
            n = len(result.get("elements") or [])
            print(f"  ok   {path.name}: {n} elements, grid {result['grid_width']}x{result['grid_height']}")

    print(f"\n{len(todo) - failures}/{len(todo)} succeeded -> {dst}/")
    print("validate with: python -m pipeline.validate_layouts")
