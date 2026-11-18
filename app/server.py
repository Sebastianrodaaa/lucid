"""Live play over a websocket from the released checkpoints, and replay of recorded logs at real speed."""
import argparse
import asyncio
import base64
import gzip
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from lucid import game
from lucid.eval import TAU, load_model, starts, summary
from lucid.model import dream, noise_draft, onehot

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent
HZ = 15  # game ticks per second, live and replay
PALETTES = {  # bg, wall, paddle, ball, brick A, brick B, bonus A, bonus B
    "A": ["#0b0610", "#3a2530", "#e58fa6", "#fff4f7", "#6e3a4b", "#9a5368", "#f7c6d3", "#c96f88"],
    "E": ["#06090a", "#1d3329", "#7cffb2", "#f4fff8", "#24603f", "#348a5b", "#d6ffe6", "#52d68a"],
}
RGB = {k: np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in v], np.float32) for k, v in PALETTES.items()}
KEYS = {"left": game.LEFT, "none": game.STAY, "right": game.RIGHT}


def picture(logits, arm):
    """Expected colour of every cell under the model's distribution: a smear stays a smear."""
    rgb = np.einsum("chw,cx->hwx", logits.exp().numpy(), RGB[arm])
    return base64.b64encode(rgb.clip(0, 255).astype(np.uint8).tobytes()).decode()


class Side:
    """One dreaming model plus the simulator that follows its dream to judge every frame."""

    def __init__(self, arm, model, prev, state, seed):
        self.arm, self.model, self.gen = arm, model, torch.Generator().manual_seed(seed)
        self.prev, self.cur = torch.from_numpy(prev)[None], torch.from_numpy(game.render(state))[None]
        self.state, self.lamps = state, deque(maxlen=100)

    @torch.no_grad()
    def tick(self, action):
        draft = torch.zeros(1, game.COLORS, game.N, game.N) if self.arm == "A" else noise_draft(1, self.gen)
        t = time.perf_counter()
        outs, energies, used = dream(self.model, onehot(self.prev), onehot(self.cur), torch.tensor([action]), draft,
                                     loops=1, tau=None if self.arm == "A" else TAU)
        ms = (time.perf_counter() - t) * 1000
        frame = outs[int(used) - 1].argmax(1)
        futures = game.legal_futures(self.state, action)  # the lamp asks the simulator, never the model
        k, frames = game.nearest(frame[0].numpy(), futures)
        lamp = bool((frames[k] == frame[0].numpy()).all())
        spawn = k // 2 if len(futures) == 8 and lamp else None  # outcomes are column-major: (column, colour)
        self.state, self.prev, self.cur = futures[k], self.cur, frame.to(torch.uint8)
        self.lamps.append(lamp)
        return dict(loops=[picture(o[0], self.arm) for o in outs[:int(used)]], used=int(used), lamp=lamp,
                    legal100=sum(self.lamps), window=len(self.lamps), futures=len(futures), spawn=spawn,
                    energy=[round(float(e), 4) for e in energies[:int(used)]], ms=round(ms, 2))


class Session:
    def __init__(self, models, seed):
        self.models, self.seed, self.t = models, seed, 0
        self.reset()

    def reset(self):
        prev, state = starts(1, self.seed)[0]
        self.sides = {arm: Side(arm, m, prev, state, self.seed) for arm, m in self.models.items()}
        self.seed += 1

    def tick(self, action):
        self.t += 1
        return dict(t=self.t, action=action, **{arm: s.tick(action) for arm, s in self.sides.items()})


def http(connection, request):
    path = request.path.split("?")[0]
    if path == "/ws":
        return None
    if path == "/api/results":
        body, kind = json.dumps(summary()).encode(), "application/json"
    elif path.startswith("/api/replay/"):
        log = ROOT / "results" / (Path(path).name + ".json.gz")
        if not log.is_file():
            return connection.respond(404, "no such replay\n")
        body, kind = gzip.decompress(log.read_bytes()), "application/json"
    elif path == "/":
        body, kind = (APP / "index.html").read_bytes(), "text/html; charset=utf-8"
    elif path.startswith("/fonts/") and (APP / "fonts" / Path(path).name).is_file():
        body, kind = (APP / "fonts" / Path(path).name).read_bytes(), "font/ttf"
    else:
        return connection.respond(404, "not found\n")
    return Response(200, "OK", Headers({"Content-Type": kind, "Content-Length": str(len(body)), "Cache-Control": "no-store"}), body)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--lucid", default=str(ROOT / "checkpoints/lucid-E.pt"))
    cli.add_argument("--onepass", default=str(ROOT / "checkpoints/onepass-A.pt"))
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--record", help="write a scripted-player session to results/<name>.json.gz instead of serving")
    cli.add_argument("--ticks", type=int, default=450)
    a = cli.parse_args()
    torch.set_num_threads(4)
    models = {"A": load_model(a.onepass), "E": load_model(a.lucid)}
    if a.record:
        session, rng, ticks = Session(models, a.seed), np.random.default_rng(a.seed), []
        for _ in range(a.ticks):  # the player steers by the simulator state that follows lucid's dream
            ticks.append(session.tick(game.scripted(session.sides["E"].state, rng)))
        meta = dict(fps=HZ, seed=a.seed, tau=TAU, player="scripted, eps 0.1", palettes=PALETTES,
                    checkpoints={"A": Path(a.onepass).name, "E": Path(a.lucid).name}, ticks=ticks)
        out = ROOT / "results" / f"{a.record}.json.gz"
        out.write_bytes(gzip.compress(json.dumps(meta).encode(), mtime=0))
        print(f"{out}: {len(ticks)} ticks at {HZ} Hz")
        return

    async def play(ws):
        session, key, closed = Session(models, a.seed), {"key": "none"}, asyncio.Event()
        await ws.send(json.dumps(dict(hello=True, fps=HZ, tau=TAU, palettes=PALETTES)))

        async def listen():
            async for msg in ws:
                m = json.loads(msg)
                if m.get("reset"):
                    session.reset()
                elif m.get("key") in KEYS:
                    key["key"] = m["key"]
            closed.set()

        listener = asyncio.create_task(listen())
        nxt = time.perf_counter()
        while not closed.is_set():
            await ws.send(json.dumps(session.tick(KEYS[key["key"]])))
            nxt += 1 / HZ
            await asyncio.sleep(max(0, nxt - time.perf_counter()))
        listener.cancel()

    async def run():
        async with serve(play, "127.0.0.1", a.port, process_request=http, max_size=2 ** 20) as server:
            print(f"lucid: http://127.0.0.1:{a.port}/  (live)   replay: http://127.0.0.1:{a.port}/?replay=demo", flush=True)
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
