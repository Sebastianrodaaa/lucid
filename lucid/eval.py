"""Score trained arms against the simulator and write results/<prefix>-<seed>.json. Real runs only."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lucid import game
from lucid.model import MAX_LOOPS, Refiner, dream, final, noise_draft, onehot
from lucid.train import candidates, legal, load

ROOT = Path(__file__).resolve().parents[1]
TAU = 0.5  # energy halting threshold, fixed in advance (not tuned on test)
TEST_SEED, TEST_N = 7777, 40_000
ARMS = {"A": "one-pass CE", "B": "nearest-of-16", "C": "looped, standard target", "E": "lucid (D = its trunk at fixed depth)"}


def ci(x, reps=1000, seed=0):
    """Mean and 95% bootstrap interval of a 0/1 or real vector."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)]


def spearman(a, b):
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    def avg(v):  # average ranks for ties
        r, out = rank(v), np.empty(len(v))
        for u in np.unique(v):
            out[v == u] = r[v == u].mean()
        return out
    ra, rb = avg(np.asarray(a)), avg(np.asarray(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def load_model(path):
    m = Refiner()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def first_draft(arm, b, gen):
    return torch.zeros(b, game.COLORS, game.N, game.N) if arm == "A" else noise_draft(b, gen)


@torch.no_grad()
def trace(model, arm, data, idx, gen, loops):
    """Argmax frame and energy after every loop for a batch; [loops, b, N, N] and [loops, b]."""
    prev, cur, act = onehot(data["prev"][idx]), onehot(data["cur"][idx]), data["act"][idx]
    outs, energies, _ = dream(model, prev, cur, act, first_draft(arm, len(idx), gen), loops=loops)
    return torch.stack([o.argmax(1) for o in outs]), torch.stack(energies)


def one_step(model, arm, data, seed, batch=500):
    """Legal-frame flags after each loop on the test set (the simulator decides legality)."""
    gen = torch.Generator().manual_seed(seed)
    loops = 1 if arm in "AB" else MAX_LOOPS
    ok, energy = [], []
    for s in range(0, len(data["act"]), batch):
        idx = torch.arange(s, min(s + batch, len(data["act"])))
        frames, e = trace(model, arm, data, idx, gen, loops)
        cand = candidates(data, idx)
        ok.append(torch.stack([legal(f, cand) for f in frames])), energy.append(e)
    return torch.cat(ok, 1).numpy(), torch.cat(energy, 1).numpy()


def halt(ok, energy, tau):
    """Loops used and legality when each frame stops at the first loop with sigmoid(energy) < tau."""
    below = energy < tau
    used = np.where(below.any(0), below.argmax(0), len(energy) - 1)
    return used + 1, ok[used, np.arange(ok.shape[1])]


def fixed_at(curve, mean_loops):
    """Illegal rate of a fixed-depth schedule mixing floor/ceil depths to hit mean_loops exactly."""
    depths = np.arange(1, len(curve) + 1)
    return float(np.interp(mean_loops, depths, curve))


def spawn_tv(model, arm, data, count, states=20, draws=100, seed=0):
    """Push `draws` noise seeds through each of `states` real random steps; compare outcome frequencies to the simulator's."""
    pick = torch.nonzero(data["owner"] >= 0).squeeze(1)
    pick = pick[data["count"][data["owner"][pick]] == count][:states]
    gen = torch.Generator().manual_seed(seed)
    hist = np.zeros(count + 1)
    for i in pick:
        idx = i.repeat(draws)
        if arm in "AB":
            frames = trace(model, arm, data, idx, gen, 1)[0][-1]
        else:
            frames, e = trace(model, arm, data, idx, gen, MAX_LOOPS)
            used = halt(np.ones(e.shape, bool), e.numpy(), TAU)[0]
            frames = frames[used - 1, torch.arange(draws)]
        cand = data["futures"][data["owner"][i]][:count]
        match = (frames[:, None] == cand).flatten(2).all(2)
        hist[:count] += match.sum(0).numpy()
        hist[count] += (~match.any(1)).sum().item()
    p = hist / hist.sum()
    tv = 0.5 * (np.abs(p[:count] - 1 / count).sum() + p[count])
    out = dict(rollouts=int(hist.sum()), tv=round(float(tv), 4), hist=hist.astype(int).tolist())
    if count == 8:  # bonus: outcome k = column k // 2, colour k % 2
        cols = p[:count].reshape(4, 2).sum(1)
        out["column_tv"] = round(float(0.5 * (np.abs(cols - 0.25).sum() + p[count])), 4)
    return out


def starts(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = game.random_state(rng)
        for _ in range(int(rng.integers(2, 20))):
            prev, s = s, game.step(s, game.scripted(s, rng), rng)
        out.append((game.render(prev), s))
    return out


@torch.no_grad()
def closed_loop(model, arm, depth, n=50, steps=100, seed=0):
    """Dream `steps` frames from real starts with a scripted player, feeding the model its own frames.
    The oracle follows the nearest legal future of each dreamed frame and lights the lamp only on exact matches."""
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    begin = starts(n, seed)
    prev = torch.from_numpy(np.stack([b[0] for b in begin]))
    states = [b[1] for b in begin]
    cur = torch.from_numpy(np.stack([game.render(s) for s in states]))
    lamps, loops = np.zeros((steps, n), bool), np.zeros((steps, n))
    for t in range(steps):
        acts = [game.scripted(s, rng) for s in states]
        a = torch.tensor(acts)
        tau = TAU if depth == "halt" else None
        outs, _, used = dream(model, onehot(prev), onehot(cur), a, first_draft(arm, n, gen),
                              loops=1 if depth == "halt" else depth, tau=tau)
        nxt = final(outs, used).argmax(1)
        for i, s in enumerate(states):
            futures = game.legal_futures(s, acts[i])
            k, frames = game.nearest(nxt[i].numpy(), futures)
            lamps[t, i] = (frames[k] == nxt[i].numpy()).all()
            states[i] = futures[k]
        loops[t] = used.numpy()
        prev, cur = cur, nxt.to(torch.uint8)
    return lamps, loops


@torch.no_grad()
def latency(model, arm, data, depth, frames=300, threads=4):
    """Median and mean ms per dreamed frame at batch 1."""
    torch.set_num_threads(threads)
    gen, ms = torch.Generator().manual_seed(0), []
    for i in range(frames):
        idx = torch.tensor([i])
        prev, cur, act = onehot(data["prev"][idx]), onehot(data["cur"][idx]), data["act"][idx]
        t = time.perf_counter()
        dream(model, prev, cur, act, first_draft(arm, 1, gen), loops=1 if depth == "halt" else depth,
              tau=TAU if depth == "halt" else None)
        ms.append((time.perf_counter() - t) * 1000)
    ms = np.array(ms[20:])
    return dict(threads=threads, median_ms=round(float(np.median(ms)), 3), mean_ms=round(float(ms.mean()), 3),
                fps=round(float(1000 / ms.mean()), 1))


def oracle_latency(data, frames=300):
    """Arm F: the simulator itself. Legal by construction; timed for one step plus render."""
    rng, ms = np.random.default_rng(0), []
    for _, s in starts(frames, 123):
        t = time.perf_counter()
        game.render(game.step(s, game.scripted(s, rng), rng))
        ms.append((time.perf_counter() - t) * 1000)
    ms = np.array(ms)
    return dict(name="simulator oracle", legal=1.0, median_ms=round(float(np.median(ms)), 4), fps=round(float(1000 / ms.mean()), 1))


def evaluate(prefix, seed, arms, rollouts=50):
    data = load(TEST_N, TEST_SEED)
    count = np.ones(len(data["act"]), int)
    hit = data["owner"].numpy() >= 0
    count[hit] = data["count"].numpy()[data["owner"].numpy()[hit]]
    spawn = count > 1
    res = dict(prefix=prefix, seed=seed, tau=TAU, test=dict(seed=TEST_SEED, transitions=TEST_N, random_steps=int(spawn.sum())), arms={})
    for arm in arms:
        path = ROOT / f"runs/{prefix}-{seed}-{arm}.pt"
        if not path.exists():
            continue
        model, meta = load_model(path), json.loads(path.with_suffix(".json").read_text())
        torch.set_num_threads(8)
        ok, energy = one_step(model, arm, data, seed)
        r = dict(name=ARMS[arm], steps=meta["steps"], train_seconds=meta["seconds"], parameters=meta["parameters"])
        r["depths"] = {str(k + 1): dict(legal=ci(ok[k]), legal_spawn=ci(ok[k][spawn]), legal_quiet=ci(ok[k][~spawn]))
                       for k in range(len(ok))}
        if arm == "E":
            used, ok_h = halt(ok, energy, TAU)
            r["halt"] = dict(legal=ci(ok_h), legal_spawn=ci(ok_h[spawn]), legal_quiet=ci(ok_h[~spawn]),
                             mean_loops=round(float(used.mean()), 4), mean_loops_spawn=round(float(used[spawn].mean()), 4),
                             mean_loops_quiet=round(float(used[~spawn].mean()), 4), spearman_loops_futures=spearman(used, count))
            curve = [1 - ok[k].mean() for k in range(len(ok))]
            r["tau_sweep"] = []
            for tau in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
                u, o = halt(ok, energy, tau)
                r["tau_sweep"].append(dict(tau=tau, mean_loops=round(float(u.mean()), 4), illegal=round(float(1 - o.mean()), 5),
                                           fixed_same_loops=round(fixed_at(curve, u.mean()), 5)))
        for count_k in (8, 16):
            r[f"spawn_{count_k}"] = spawn_tv(model, arm, data, count_k, seed=seed)
        depths = ["halt"] if arm == "E" else [1] if arm in "AB" else [1, 8]
        r["closed_loop"], r["latency"] = {}, {}
        for depth in depths + ([1, 2, 4, 8] if arm == "E" else []):
            torch.set_num_threads(8)
            lamps, loops = closed_loop(model, arm, depth, n=rollouts, seed=seed)
            r["closed_loop"][str(depth)] = dict(legal=ci(lamps.ravel()), per_rollout=ci(lamps.mean(0)), mean_loops=round(float(loops.mean()), 4))
            r["latency"][str(depth)] = latency(model, arm, data, depth)
        res["arms"][arm] = r
        print(arm, json.dumps({k: v for k, v in r.items() if k in ("halt", "closed_loop", "latency", "spawn_8")}), flush=True)
    res["oracle"] = oracle_latency(data)
    out = ROOT / f"results/{prefix}-{seed}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def studies(prefix=None):
    """Every results/<prefix>-<seed>.json; the full study if present, else the smoke run."""
    for p in [prefix] if prefix else ["study", "smoke"]:
        runs = [json.loads(f.read_text()) for f in sorted((ROOT / "results").glob(f"{p}-[0-9]*.json"))]
        if runs:
            return p, runs
    return None, []


def _cell(vals, digits=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = np.mean(vals)
    return f"{m:.{digits}f}" if len(vals) == 1 else f"{m:.{digits}f} ({min(vals):.{digits}f}-{max(vals):.{digits}f})"


def rows(runs):
    """One row per configuration: mean over seeds, (min-max over seeds) when there is more than one seed."""
    configs = [("A", "A one-pass", "1"), ("B", "B nearest-of-16", "1"), ("C", "C looped std, depth 1", "1"),
               ("C", "C looped std, depth 8", "8")] + [("E", f"D proximal, fixed depth {d}", str(d)) for d in (1, 2, 4, 8)] + \
              [("E", "E lucid, energy halt", "halt")]
    out = []
    for arm, label, depth in configs:
        rs = [r["arms"][arm] for r in runs if arm in r["arms"]]
        if not rs:
            continue
        one = [r["halt"] if depth == "halt" else r["depths"].get(depth) for r in rs]
        cl = [r["closed_loop"].get(depth) for r in rs]
        lat = [r["latency"].get(depth) for r in rs]
        loops = [r["halt"]["mean_loops"] if depth == "halt" else int(depth) for r in rs]
        out.append([label, _cell([o["legal"][0] for o in one if o]), _cell([o["legal_spawn"][0] for o in one if o]),
                    _cell([c["legal"][0] for c in cl if c]), _cell([r["spawn_8"]["column_tv"] for r in rs]) if depth in ("1", "halt") or arm == "C" else "-",
                    _cell(loops, 2), _cell([x["median_ms"] for x in lat if x], 1), _cell([x["fps"] for x in lat if x], 0)])
    o = [r["oracle"] for r in runs if "oracle" in r]
    if o:
        out.append(["F simulator oracle", "1.000", "1.000", "1.000", "0.000", "-", _cell([x["median_ms"] for x in o], 3), _cell([x["fps"] for x in o], 0)])
    return out


HEAD = ["arm", "one-step legal", "legal on random steps", "closed-loop legal", "bonus column TV", "mean loops", "ms/frame", "FPS"]


def tables(prefix=None):
    """Markdown tables for README and REPORT, regenerated from results/*.json only."""
    p, runs = studies(prefix)
    if not runs:
        return "_no results yet_\n"
    seeds = ", ".join(str(r["seed"]) for r in runs)
    md = [f"Run `{p}`, seeds {seeds}; test set {runs[0]['test']['transitions']:,} real transitions "
          f"({runs[0]['test']['random_steps']:,} random steps). Mean over seeds, (min-max) across seeds.", "",
          "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    md += ["| " + " | ".join(r) + " |" for r in rows(runs)]
    halt = [(r["seed"], r["arms"]["E"]) for r in runs if "E" in r["arms"]]
    if halt:
        md += ["", "| seed | lucid illegal at halt | best fixed depth illegal at same mean loops | mean loops | Spearman loops vs futures |", "|---|---|---|---|---|"]
        for seed, e in halt:
            sw = next(x for x in e["tau_sweep"] if x["tau"] == TAU)
            md.append(f"| {seed} | {sw['illegal']:.4f} | {sw['fixed_same_loops']:.4f} | {sw['mean_loops']:.2f} | {e['halt']['spearman_loops_futures']} |")
    return "\n".join(md) + "\n"


def summary():
    """What the dashboard shows: the table, the 2,000-rollout bonus histogram and the end-card numbers."""
    p, runs = studies()
    if not runs:
        return {}
    r0 = runs[0]["arms"]
    out = dict(note=f"run {p} · seeds {', '.join(str(r['seed']) for r in runs)} · mean over seeds (min-max)", head=HEAD, rows=rows(runs))
    if "A" in r0 and "E" in r0:
        col = lambda h: np.array(h[:8]).reshape(4, 2).sum(1).tolist() + [h[8]]
        out["hist"] = dict(rollouts=r0["E"]["spawn_8"]["rollouts"], E=col(r0["E"]["spawn_8"]["hist"]), A=col(r0["A"]["spawn_8"]["hist"]),
                           tvE=r0["E"]["spawn_8"]["column_tv"], tvA=r0["A"]["spawn_8"]["column_tv"], seed=runs[0]["seed"])
        es = [r["arms"]["E"] for r in runs if "E" in r["arms"]]
        out["end"] = dict(nums=[["one-step legal", _cell([e["halt"]["legal"][0] for e in es])],
                                ["closed-loop legal", _cell([e["closed_loop"]["halt"]["legal"][0] for e in es])],
                                ["fps on cpu, 4 threads", _cell([e["latency"]["halt"]["fps"] for e in es], 0)],
                                ["mean loops", _cell([e["halt"]["mean_loops"] for e in es], 2)]],
                          note=f"from results/{p}-*.json · lucid, energy halt at tau {TAU} · one-pass baseline one-step legal "
                               f"{_cell([r['arms']['A']['depths']['1']['legal'][0] for r in runs if 'A' in r['arms']])}")
    return out


def readme_block(text, block):
    """Replace the text between the results markers."""
    a, b = "<!-- results:start -->\n", "<!-- results:end -->"
    i, j = text.index(a) + len(a), text.index(b)
    return text[:i] + block + text[j:]


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--tables"]:
        for f in (ROOT / "README.md", ROOT / "docs/REPORT.md"):
            f.write_text(readme_block(f.read_text(), tables()))
        raise SystemExit
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prefix", default="study")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--arms", default="ABCE")
    cli.add_argument("--rollouts", type=int, default=50)
    a = cli.parse_args()
    evaluate(a.prefix, a.seed, a.arms, a.rollouts)
