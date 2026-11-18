"""Record real 1080p footage of the dashboard replaying a recorded real run at real speed, 16:9 and 9:16.
Frames are rendered deterministically from the replay log (60 fps video = 4 frames per 15 Hz tick), then a B&W README GIF."""
import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs/media"
CHROME = Path.home() / ".local/bin/google-chrome"
FPS = 60


def ffmpeg(path, fps=FPS):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "png", "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def film(page, out, ticks, slow=None, card_seconds=4):
    """Real speed: each tick spans FPS/hz frames. `slow` = (start, end, factor) renders that span slowed and labelled."""
    hz = page.evaluate("window.fps")
    enc = ffmpeg(out)
    per = FPS // hz
    for t in range(ticks):
        factor = slow[2] if slow and slow[0] <= t < slow[1] else 1
        subs = int(per / factor)
        for k in range(subs):
            page.evaluate(f"captureAt({t}, {k}, {subs}, {factor})")
            enc.stdin.write(page.screenshot(type="png"))
    page.evaluate("captureAt('card')")
    shot = page.screenshot(type="png")
    for _ in range(FPS * card_seconds):
        enc.stdin.write(shot)
    enc.stdin.close()
    enc.wait()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", default="demo")
    cli.add_argument("--port", type=int, default=8123)
    cli.add_argument("--seconds", type=float, default=26, help="real-speed seconds of replay before the end card")
    a = cli.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([sys.executable, "-m", "app.server", "--port", str(a.port)], cwd=ROOT)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/api/results")
                break
            except OSError:
                time.sleep(0.2)
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=str(CHROME) if CHROME.exists() else None)
            for name, size in (("16x9", (1920, 1080)), ("9x16", (1080, 1920))):
                page = browser.new_page(viewport=dict(width=size[0], height=size[1]))
                page.goto(f"http://127.0.0.1:{a.port}/?replay={a.replay}&film=1")
                page.wait_for_function("window.captureReady === true")
                page.evaluate("document.fonts.ready")
                ticks = min(page.evaluate("window.ticks"), int(a.seconds * page.evaluate("window.fps")))
                film(page, MEDIA / f"lucid-{name}.mp4", ticks)
                print("wrote", MEDIA / f"lucid-{name}.mp4", ticks, "ticks", flush=True)
                if name == "16x9":
                    page.evaluate("captureAt(40, 3, 4)")
                    page.screenshot(path=str(MEDIA / "hero-color.png"))
                page.close()
            page = browser.new_page(viewport=dict(width=1920, height=1080))
            page.goto(f"http://127.0.0.1:{a.port}/?view=results")
            page.wait_for_timeout(800)
            page.screenshot(path=str(MEDIA / "results-color.png"))
            browser.close()
    finally:
        server.terminate()
    # README GIF: first 12 s of the 16:9 film, grayscale, 960 px, 15 fps (one frame per tick: still real speed)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", "12", "-i", str(MEDIA / "lucid-16x9.mp4"), "-vf",
                    "fps=15,scale=960:-1:flags=lanczos,format=gray,split[a][b];[a]palettegen=max_colors=32[p];[b][p]paletteuse=dither=bayer",
                    str(ROOT / "docs/screenshots/demo.gif")], check=True)


if __name__ == "__main__":
    main()
