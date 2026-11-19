"""Black-and-white README art from real results: banner, plots and B&W conversions of dashboard captures."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lucid.eval import studies  # noqa: E402

SHOTS, ART = ROOT / "docs/screenshots", ROOT / "docs/art"
PLEX = str(ROOT / "app/fonts/ibmplexmono-IBMPlexMono-Regular.ttf")


def plate(path, box, w, h, left):
    """A crop of a Doré plate, faded toward the centre, then 1-bit Floyd-Steinberg."""
    im = Image.open(path).convert("L")
    W, H = im.size
    im = ImageOps.autocontrast(im.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H))), cutoff=1)
    im = ImageOps.fit(im, (w, h), Image.LANCZOS)
    fade = np.clip(np.linspace(1.25, -0.1, w), 0, 1) ** 1.2
    g = np.array(im, float) * 0.85 * (fade if left else fade[::-1])
    return np.array(Image.fromarray(g.astype(np.uint8)).convert("1"), np.uint8) * 200


def banner():
    W, H, w = 1280, 400, 380
    canvas = np.zeros((H, W), np.uint8)
    canvas[:, :w] = plate(ART / "dore-albatross.jpg", (0.35, 0.3, 0.95, 0.85), w, H, True)
    canvas[:, -w:] = plate(ART / "dore-water-snakes.jpg", (0.12, 0.18, 0.62, 0.8), w, H, False)
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype(str(ART / "Jacquard12-Regular.ttf"), 168)
    d.text((W / 2, 168), "lucid", font=title, fill=240, anchor="mm")
    d.text((W / 2, 290), "a dream you can steer.", font=ImageFont.truetype(PLEX, 24), fill=170, anchor="mm")
    SHOTS.mkdir(parents=True, exist_ok=True)
    img.save(SHOTS / "banner.png", optimize=True)


def bw(src, dst, width=1280):
    """Dashboard capture to grayscale (README is black and white only)."""
    im = ImageOps.grayscale(Image.open(src))
    im = ImageOps.autocontrast(im, cutoff=0.5)
    im.thumbnail((width, width))
    im.save(dst, optimize=True)


def chart(w, h, xlim, ylim, xlabel, ylabel, title):
    """A black canvas with white axes; returns the image, a drawer and a data-to-pixel map."""
    img = Image.new("L", (w, h), 0)
    d, f = ImageDraw.Draw(img), ImageFont.truetype(PLEX, 15)
    L, R, T, B = 90, w - 30, 50, h - 60
    d.line([(L, T), (L, B), (R, B)], fill=255)
    d.text((L, 16), title, font=f, fill=200)
    d.text(((L + R) / 2, h - 22), xlabel, font=f, fill=200, anchor="mm")
    d.text((14, T - 18), ylabel, font=f, fill=200, anchor="lm")
    xy = lambda x, y: (L + (x - xlim[0]) / (xlim[1] - xlim[0]) * (R - L), B - (y - ylim[0]) / (ylim[1] - ylim[0]) * (B - T))
    for k in range(5):
        y = ylim[0] + k * (ylim[1] - ylim[0]) / 4
        d.text((L - 8, xy(0, y)[1] if xlim[0] == 0 else xy(xlim[0], y)[1]), f"{y:.3g}", font=f, fill=200, anchor="rm")
    return img, d, f, xy


def dot(d, p, filled, r=5):
    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255 if filled else 0, outline=255)


def plots():
    prefix, runs = studies()
    if not runs:
        return
    es = [r["arms"]["E"] for r in runs if "E" in r["arms"]]
    top = max(1 - e["depths"]["1"]["legal"][0] for e in es) * 1.15
    img, d, f, xy = chart(1280, 560, (0.5, 8.5), (0, top), "mean loops per frame",
                          "illegal", "open: fixed depth 1-8 (arm D)   filled: energy halting, tau sweep (arm E)   one row per seed")
    for x in range(1, 9):
        d.text(xy(x, 0), str(x), font=f, fill=200, anchor="mt")
    for e in es:
        for k in range(1, 9):
            dot(d, xy(k, 1 - e["depths"][str(k)]["legal"][0]), False)
        for sw in e["tau_sweep"]:
            dot(d, xy(sw["mean_loops"], sw["illegal"]), True, 4)
    img.save(SHOTS / "halt.png", optimize=True)

    names = [("A", "1", "A one-pass"), ("B", "1", "B nearest-16"), ("C", "8", "C std x8"), ("E", "1", "D prox x1"),
             ("E", "8", "D prox x8"), ("E", "halt", "E lucid")]
    img, d, f, xy = chart(1280, 560, (-0.5, len(names) - 0.5), (0, 1), "", "legal",
                          "one-step legal frames, 3 seeds   open: quiet steps   filled: random steps   bars: 95% bootstrap CI")
    for i, (arm, depth, label) in enumerate(names):
        d.text((xy(i, 0)[0], xy(i, 0)[1] + 12), label, font=f, fill=200, anchor="mt")
        for j, r in enumerate(runs):
            a = r["arms"].get(arm)
            if not a:
                continue
            v = a["halt"] if depth == "halt" else a["depths"][depth]
            for k, key in enumerate(("legal_quiet", "legal_spawn")):
                m, lo, hi = v[key]
                x = i + (k - 0.5) * 0.3 + (j - 1) * 0.07
                d.line([xy(x, lo), xy(x, hi)], fill=255)
                dot(d, xy(x, m), bool(k), 4)
    img.save(SHOTS / "legal.png", optimize=True)


if __name__ == "__main__":
    banner()
    plots()
    for name in sys.argv[1:]:  # dashboard captures to convert
        bw(name, SHOTS / Path(name).name)
