"""Render the AdResyze architecture diagram.

    python docs/make_architecture.py

Deliberately mirrors the shipped pipeline rather than an aspirational one: the LoRA
is drawn off the main path because it is not in it.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1800, 1000
BG = (13, 14, 22)
PANEL = (22, 24, 36)
FG = (236, 238, 245)
MUTED = (138, 143, 165)
FAINT = (86, 90, 110)

AI = (124, 108, 240)       # purple  - model inference
EXEC = (46, 190, 130)      # green   - deterministic execution
SERVE = (58, 140, 245)     # blue    - serving / IO
PUB = (232, 145, 60)       # orange  - published artefacts
DROP = (95, 99, 120)       # grey    - not in the pipeline

FONTS = ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf")


def font(size: int, bold: bool = False):
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(40, True)
F_SUB = font(19)
F_BOX = font(21, True)
F_LINE = font(15)
F_TAG = font(13)
F_LEG = font(15)


def centre(draw, text, box, y, fnt, fill):
    x0, _, x1, _ = box
    w = draw.textlength(text, font=fnt)
    draw.text(((x0 + x1 - w) / 2, y), text, font=fnt, fill=fill)


def node(draw, box, title, lines, colour, tag=None, dashed=False):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=colour, width=3 if not dashed else 2)
    if dashed:  # overdraw a dashed edge to read as "not part of the flow"
        draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=None)
        for x in range(int(x0), int(x1), 14):
            draw.line([(x, y0), (min(x + 7, x1), y0)], fill=colour, width=2)
            draw.line([(x, y1), (min(x + 7, x1), y1)], fill=colour, width=2)
        for y in range(int(y0), int(y1), 14):
            draw.line([(x0, y), (x0, min(y + 7, y1))], fill=colour, width=2)
            draw.line([(x1, y), (x1, min(y + 7, y1))], fill=colour, width=2)

    y = y0 + 18
    centre(draw, title, box, y, F_BOX, FG if not dashed else MUTED)
    y += 34
    for line in lines:
        centre(draw, line, box, y, F_LINE, MUTED)
        y += 21
    if tag:
        centre(draw, tag, box, y1 - 26, F_TAG, colour)


def arrow(draw, start, end, colour, width=3, dashed=False, head=11):
    x0, y0 = start
    x1, y1 = end
    if dashed:
        total = max(abs(x1 - x0), abs(y1 - y0))
        steps = max(int(total / 13), 1)
        for i in range(steps):
            if i % 2:
                continue
            a = (x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps)
            b = (x0 + (x1 - x0) * (i + 1) / steps, y0 + (y1 - y0) * (i + 1) / steps)
            draw.line([a, b], fill=colour, width=2)
    else:
        draw.line([start, end], fill=colour, width=width)

    if x1 == x0:  # vertical
        d = 1 if y1 > y0 else -1
        draw.polygon([(x1, y1), (x1 - head * 0.6, y1 - head * d), (x1 + head * 0.6, y1 - head * d)], fill=colour)
    else:
        d = 1 if x1 > x0 else -1
        draw.polygon([(x1, y1), (x1 - head * d, y1 - head * 0.6), (x1 - head * d, y1 + head * 0.6)], fill=colour)


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    centre(d, "AdResyze - The Pipeline", (0, 0, W, 0), 44, F_TITLE, FG)
    centre(d, "AI for understanding  -  deterministic math for execution", (0, 0, W, 0), 98, F_SUB, MUTED)

    row_y0, row_y1 = 380, 560
    boxes = {}

    boxes["input"] = (70, row_y0 + 20, 300, row_y1 - 20)
    node(d, boxes["input"], "Ad Image", ["upload or batch", "any size, any ratio"], SERVE)

    boxes["model"] = (370, row_y0 - 30, 660, row_y1 + 30)
    node(d, boxes["model"], "Qwen2.5-VL 7B", [
        "STRICT_PROMPT states", "the vocabulary + formats",
        "records source AND grid dims",
    ], AI, tag="stock model - Modal L4")

    boxes["norm"] = (730, row_y0, 990, row_y1)
    node(d, boxes["norm"], "normalize()", [
        "splits compound labels", "drops empty boxes",
        "grounds to 0..1 coords",
    ], EXEC, tag="repairs model noise")

    boxes["reflow"] = (1060, row_y0 - 30, 1350, row_y1 + 30)
    node(d, boxes["reflow"], "Reflow engine", [
        "crop when nothing", "critical is lost",
        "otherwise pad + fill",
    ], EXEC, tag="pure Pillow - deterministic")

    boxes["out"] = (1420, row_y0 - 45, 1730, row_y1 + 45)
    node(d, boxes["out"], "4 Platform Ratios", [
        "1:1     Feed",
        "4:5     Portrait",
        "1.91:1  Banner",
        "9:16    Story",
    ], SERVE, tag="94.1% lose nothing - 0 logos/CTAs lost")

    mid = (row_y0 + row_y1) / 2
    arrow(d, (300, mid), (365, mid), SERVE)
    arrow(d, (660, mid), (725, mid), AI)
    arrow(d, (990, mid), (1055, mid), EXEC)
    arrow(d, (1350, mid), (1415, mid), EXEC)

    # --- the adapter, deliberately off the path -----------------------------
    lora = (370, 170, 660, 300)
    node(d, lora, "adresyze-lora", [
        "96% vs 100% parse rate",
        "3.9 vs 4.5 elements found",
    ], DROP, tag="NOT in the pipeline - lost the bake-off", dashed=True)
    arrow(d, (515, 300), (515, row_y0 - 34), DROP, dashed=True)
    d.text((672, 226), "measured, then", font=F_TAG, fill=DROP)
    d.text((672, 244), "dropped", font=F_TAG, fill=DROP)

    # --- what the v1 design carried and this one does not -------------------
    removed = (70, 700, 300, 820)
    node(d, removed, "Removed", [
        "LoRA  -  n8n  -  Supabase",
        "FastAPI  -  VPS",
    ], DROP, tag="none of it earned its place", dashed=True)

    # --- serving ------------------------------------------------------------
    ui = (370, 700, 990, 820)
    node(d, ui, "Gradio UI", [
        "CPU container calls the GPU class remotely,",
        "so the L4 scales to zero between requests",
    ], SERVE, tag="Modal - one HTTPS URL")
    arrow(d, (515, 700), (515, row_y1 + 34), SERVE)

    # --- published (artefacts, not a runtime step -- so no arrow) ------------
    pub = (1060, 700, 1730, 820)
    node(d, pub, "Published", [
        "302 grounded layouts  -  hf.co/datasets/builditwithgk/adresyze-ad-layouts",
        "adapter kept as provenance  -  code + bake-off on github.com/builditwithgk/adresyze",
    ], PUB, tag="artefacts, not a runtime step")

    # --- legend -------------------------------------------------------------
    legend = [("AI inference", AI), ("Deterministic", EXEC), ("Serving / IO", SERVE),
              ("Published", PUB), ("Dropped", DROP)]
    x = 70
    for label, colour in legend:
        d.ellipse((x, 906, x + 13, 919), fill=colour)
        d.text((x + 22, 903), label, font=F_LEG, fill=MUTED)
        x += int(d.textlength(label, font=F_LEG)) + 62

    tagline = "#BuildItWithGK"
    d.text((W - 70 - d.textlength(tagline, font=F_LEG), 903), tagline, font=F_LEG, fill=FAINT)

    out = Path(__file__).resolve().parent / "architecture.png"
    img.save(out, quality=95)
    print("wrote", out, img.size)


if __name__ == "__main__":
    main()
