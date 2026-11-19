"""One runnable check: python3 test_lucid.py"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from lucid import game
from lucid.eval import ROOT, readme_block, tables
from lucid.train import legal, proximal, train


def test_legal_futures_contain_real_next(steps=10_000):
    """Draw each random event directly (column, colour, respawn) and check the enumeration contains the result."""
    rng, s, seen = np.random.default_rng(0), game.random_state(np.random.default_rng(1)), set()
    for t in range(steps):
        a = int(rng.integers(3)) if t % 2 else game.scripted(s, rng)
        det, event = game._advance(s, a)
        if event is None:
            real = det
        elif event[0] == "bonus":
            j, color = int(rng.integers(game.BRICK_W)), int(rng.choice([game.BONUS_A, game.BONUS_B]))
            real = det._replace(bonuses=det.bonuses + ((event[1], event[2] + j, color),))
        else:
            col = int(rng.choice(game.RESPAWN_COLS))
            real = det._replace(ball=(game.RESPAWN_ROW, col, -1, 1 if col < 15 else -1))
        futures = game.legal_futures(s, a)
        frames = {game.render(f).tobytes() for f in futures}
        assert game.render(real).tobytes() in frames, t
        assert len(frames) == len(futures), "futures must be distinct frames"
        seen.add(len(futures))
        s = real
    assert seen == {1, 8, 16}, seen


class Liar(torch.nn.Module):
    """Tests only: always dreams the same fixed frame and swears it is legal (energy -inf)."""

    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def loop(self, prev, cur, action, draft):
        logits = torch.full((1, game.COLORS, game.N, game.N), -12.0)
        logits.scatter_(1, torch.from_numpy(self.frame).long()[None, None], 0.0)
        return logits, torch.tensor([-1e9])


def test_lamp_uses_simulator_not_model():
    from app.server import Side
    rng = np.random.default_rng(3)
    s = game.random_state(rng)
    prev = game.render(s)
    truth = game.render(game.legal_futures(s, game.STAY)[0])
    assert Side("E", Liar(truth), prev, s, 0).tick(game.STAY)["lamp"] is True
    wrong = truth.copy()
    wrong[15, 15] = game.BONUS_A if wrong[15, 15] != game.BONUS_A else game.BG
    assert Side("E", Liar(wrong), prev, s, 0).tick(game.STAY)["lamp"] is False, "a confident model must not light the lamp"


def test_proximal_picks_nearest_legal_future():
    cand = torch.zeros(1, 3, game.N, game.N, dtype=torch.uint8)
    cand[0, 0, 5, 5], cand[0, 1, 5, 6], cand[0, 2, 5, 7] = game.BONUS_A, game.BONUS_A, game.BONUS_A
    pred = cand[0, 1].clone()
    pred[20, 20] = game.BRICK_A  # one stray cell; still nearest to candidate 1
    logits = torch.nn.functional.one_hot(pred.long(), game.COLORS).permute(2, 0, 1)[None].float().mul(5).log_softmax(1)
    assert torch.equal(proximal(logits, cand)[0], cand[0, 1])
    assert legal(cand[:, 2], cand).item() and not legal(pred[None], cand).item()


def test_tiny_training_runs():
    with tempfile.TemporaryDirectory() as d:
        for arm in "ABCE":
            _, meta = train(arm, 0, 2000, Path(d) / f"{arm}.pt", steps=2, batch=8, log_every=1)
            assert meta["steps"] == 2 and 1.7e6 < meta["parameters"] < 2.1e6


def test_results_regenerate_readme_tables():
    for f in (ROOT / "README.md", ROOT / "docs/REPORT.md"):
        text = f.read_text()
        assert readme_block(text, tables()) == text, f"{f.name} tables are stale: run python -m lucid.eval --tables"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
