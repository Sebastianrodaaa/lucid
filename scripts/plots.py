"""Black-and-white README art from real results: banner, plots and B&W conversions of dashboard captures."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lucid.eval import TAU, studies  # noqa: E402

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


def dots(ax_w, ax_h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "monospace", "font.size": 9, "axes.edgecolor": "white", "axes.labelcolor": "white",
                         "xtick.color": "white", "ytick.color": "white", "text.color": "white", "axes.facecolor": "black",
                         "figure.facecolor": "black", "axes.spines.top": False, "axes.spines.right": False})
    return plt.subplots(figsize=(ax_w, ax_h), dpi=160)


def plots():
    prefix, runs = studies()
    if not runs:
        return
    fig, ax = dots(6.4, 3.4)
    for r in runs:
        e = r["arms"].get("E")
        if not e:
            continue
        sw = e["tau_sweep"]
        fixed = [1 - e["depths"][str(k)]["legal"][0] for k in range(1, 9)]
        ax.plot(range(1, 9), fixed, "o", color="white", ms=4, mfc="none")
        ax.plot([x["mean_loops"] for x in sw], [x["illegal"] for x in sw], "o", color="white", ms=4)
        h = next(x for x in sw if x["tau"] == TAU)
        ax.annotate(f"seed {r['seed']} τ {TAU}", (h["mean_loops"], h["illegal"]), xytext=(6, 6), textcoords="offset points", fontsize=7)
    ax.set_xlabel("mean loops per frame")
    ax.set_ylabel("illegal one-step frames")
    ax.set_title("open: fixed depth 1-8   filled: energy halting, τ sweep", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(SHOTS / "halt.png")

    fig, ax = dots(6.4, 3.0)
    names = [("A", "1", "A one-pass"), ("B", "1", "B nearest-16"), ("C", "8", "C std x8"), ("E", "8", "D prox x8"), ("E", "halt", "E lucid")]
    for i, (arm, depth, label) in enumerate(names):
        for j, r in enumerate(runs):
            a = r["arms"].get(arm)
            if not a:
                continue
            v = a["halt"] if depth == "halt" else a["depths"][depth]
            for k, key in enumerate(("legal_quiet", "legal_spawn")):
                m, lo, hi = v[key]
                x = i + (k - 0.5) * 0.3 + (j - 1) * 0.06
                ax.plot([x, x], [lo, hi], color="white", lw=1)
                ax.plot(x, m, "o", color="white", ms=4, mfc="white" if k else "none")
    ax.set_xticks(range(len(names)), [n[2] for n in names])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("one-step legal frames")
    ax.set_title("open: quiet steps   filled: random steps (spawn, respawn)   bars: 95% bootstrap CI per seed", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(SHOTS / "legal.png")


if __name__ == "__main__":
    banner()
    plots()
    for name in sys.argv[1:]:  # dashboard captures to convert
        bw(name, SHOTS / Path(name).name)
