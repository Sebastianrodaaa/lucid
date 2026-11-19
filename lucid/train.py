"""Train one arm on real simulator transitions. A one-pass CE, B nearest-of-16, C looped standard target,
E looped proximal target plus energy head (its trunk at fixed depth is arm D)."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lucid.game import generate
from lucid.model import MAX_LOOPS, Refiner, noise_draft, onehot

ROOT = Path(__file__).resolve().parents[1]
K = 16  # most legal futures of any step (ball respawn columns)


def load(n, seed):
    """Real transitions, cached; legal futures padded to K by repeating the first."""
    path = ROOT / f".cache/data-{n}-{seed}.npz"
    if not path.exists():
        d = generate(n, seed)
        fut = np.stack([np.concatenate([f, np.repeat(f[:1], K - len(f), 0)]) for f in d["futures"]])
        count = np.array([len(f) for f in d["futures"]])
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.npz")  # publish atomically: two training lanes may build the same cache
        np.savez(tmp, prev=d["prev"], cur=d["cur"], act=d["act"], next=d["next"], owner=d["owner"], futures=fut, count=count)
        os.replace(tmp, path)
    return {k: torch.from_numpy(v) for k, v in np.load(path).items()}


def candidates(data, idx):
    """[b, K, N, N] legal futures per sample (the real next frame repeated when the step is deterministic)."""
    own = data["owner"][idx]
    cand = data["next"][idx][:, None].repeat(1, K, 1, 1)
    hit = own >= 0
    cand[hit] = data["futures"][own[hit]]
    return cand


def proximal(logits, cand):
    """The legal future the prediction already likes most: argmax over candidates of summed log-likelihood."""
    b = logits.shape[0]
    ll = logits.flatten(2).gather(1, cand.flatten(2).long()).sum(2)
    return cand[torch.arange(b), ll.argmax(1)]


def legal(frame, cand):
    return (frame[:, None] == cand).flatten(2).all(2).any(1)


def batch_loss(model, arm, data, idx, gen):
    prev, cur, act = onehot(data["prev"][idx]), onehot(data["cur"][idx]), data["act"][idx]
    real = data["next"][idx].long()
    b, stats = len(idx), {}
    if arm == "A":
        out, _ = model.loop(prev, cur, act, torch.zeros(b, *prev.shape[1:]))
        return F.nll_loss(out, real), stats
    draft = noise_draft(b, gen)
    if arm == "B":
        m, rnd = 16, torch.nonzero(data["owner"][idx] >= 0).squeeze(1)
        if len(rnd):  # ponytail: the 16-draft search runs only on random steps; deterministic steps have one target anyway
            with torch.no_grad():
                rep = lambda t: t[rnd].repeat(m, *[1] * (t.dim() - 1))
                drafts = noise_draft(len(rnd) * m, gen)
                out, _ = model.loop(rep(prev), rep(cur), rep(act), drafts)
                nll = F.nll_loss(out, rep(real), reduction="none").flatten(1).sum(1).view(m, len(rnd))
                draft[rnd] = drafts.view(m, len(rnd), *drafts.shape[1:])[nll.argmin(0), torch.arange(len(rnd))]
        out, _ = model.loop(prev, cur, act, draft)
        return F.nll_loss(out, real), stats
    depth = int(torch.randint(1, MAX_LOOPS + 1, (1,), generator=gen))
    with torch.no_grad():
        for _ in range(depth - 1):
            draft, _ = model.loop(prev, cur, act, draft)
    out, energy = model.loop(prev, cur, act, draft)
    if arm == "C":
        return F.nll_loss(out, real), stats
    cand = candidates(data, idx)
    target = proximal(out.detach(), cand).long()
    illegal = (~legal(out.argmax(1), cand)).float()
    stats["illegal"] = illegal.mean().item()
    return F.nll_loss(out, target) + 0.1 * F.binary_cross_entropy_with_logits(energy, illegal), stats


def train(arm, seed, n, out, minutes=None, steps=None, batch=128, lr=1e-3, log_every=50):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    data = load(n, seed=1000 + seed)
    rnd = torch.nonzero(data["owner"] >= 0).squeeze(1)
    model = Refiner()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    t0, step, log = time.time(), 0, []
    budget = minutes * 60 if minutes else float("inf")
    while time.time() - t0 < budget and step < (steps or float("inf")):
        frac = max((time.time() - t0) / budget, step / (steps or float("inf")))
        for g in opt.param_groups:
            g["lr"] = lr * min(1, step / 200) * (0.5 + 0.5 * np.cos(np.pi * frac))
        idx = torch.cat([torch.randint(0, n, (batch - batch // 4,), generator=gen),  # a quarter of every batch is random steps
                         rnd[torch.randint(0, len(rnd), (batch // 4,), generator=gen)]])
        loss, stats = batch_loss(model, arm, data, idx, gen)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % log_every == 0:
            log.append(dict(step=step, seconds=round(time.time() - t0, 1), loss=round(loss.item(), 5), **stats))
            print(arm, seed, log[-1], flush=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    meta = dict(arm=arm, seed=seed, transitions=n, minutes=minutes, step_budget=steps, batch=batch, lr=lr, steps=step,
                seconds=round(time.time() - t0, 1), threads=torch.get_num_threads(),
                parameters=sum(p.numel() for p in model.parameters()), log=log)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return model, meta


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--arm", choices="ABCE", required=True)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--transitions", type=int, default=500_000)
    cli.add_argument("--minutes", type=float, help="wall-clock budget")
    cli.add_argument("--steps", type=int, help="optimizer-step budget")
    cli.add_argument("--out", required=True)
    a = cli.parse_args()
    train(a.arm, a.seed, a.transitions, a.out, a.minutes, a.steps)
