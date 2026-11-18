"""One looped conv-transformer refiner for every learned arm; only training and depth differ."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lucid.game import COLORS, N

MAX_LOOPS = 8


class Refiner(nn.Module):
    def __init__(self, d=256, layers=2, patch=4, heads=8, conv=32):
        super().__init__()
        self.p, g = patch, N // patch
        self.stem = nn.Sequential(nn.Conv2d(3 * COLORS, conv, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(conv, conv, 3, padding=1), nn.GELU())
        self.embed = nn.Linear(conv * patch * patch, d)
        self.pos = nn.Parameter(torch.randn(g * g, d) * 0.02)
        self.act = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.block = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)  # shared by every loop
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, patch * patch * COLORS)
        self.energy = nn.Sequential(nn.Linear(d + patch * patch * COLORS, 64), nn.GELU(), nn.Linear(64, 1))

    def _tokens(self, x):
        b, c = x.shape[:2]
        return x.unfold(2, self.p, self.p).unfold(3, self.p, self.p).permute(0, 2, 3, 1, 4, 5).reshape(b, -1, c * self.p ** 2)

    def loop(self, prev, cur, action, draft):
        """One pass: (prev, cur one-hot, action, draft logits) -> refined logits, illegal-frame energy logit."""
        x = torch.cat([prev, cur, draft.softmax(1)], 1)
        h = self.embed(self._tokens(self.stem(x))) + self.pos + self.act(action)[:, None]
        h = self.norm(self.block(h))
        b, g = h.shape[0], N // self.p
        delta = self.head(h).view(b, g, g, COLORS, self.p, self.p).permute(0, 3, 1, 4, 2, 5).reshape(b, COLORS, N, N)
        out = delta.log_softmax(1).clamp(min=-12)
        e = self.energy(torch.cat([h.detach(), self._tokens(out.detach().exp())], -1)).squeeze(-1)
        return out, e.logsumexp(-1)


def onehot(frames):
    return F.one_hot(frames.long(), COLORS).permute(0, 3, 1, 2).float()


def noise_draft(b, generator=None, scale=1.0):
    return torch.randn(b, COLORS, N, N, generator=generator) * scale


@torch.no_grad()
def dream(model, prev, cur, action, draft, loops=1, tau=None):
    """Run fixed `loops`, or halt per frame once sigmoid(energy) < tau (at most MAX_LOOPS).
    Returns per-loop logits list, energies list, and loops used per sample."""
    outs, energies = [], []
    used = torch.full((prev.shape[0],), loops if tau is None else MAX_LOOPS)
    done = torch.zeros(prev.shape[0], dtype=torch.bool)
    for k in range(loops if tau is None else MAX_LOOPS):
        draft, e = model.loop(prev, cur, action, draft)
        outs.append(draft), energies.append(e.sigmoid())
        if tau is not None:
            stop = ~done & (energies[-1] < tau)
            used[stop], done = k + 1, done | stop
            if done.all():
                break
    return outs, energies, used


def final(outs, used):
    """Logits of the loop each sample halted at."""
    stack = torch.stack(outs)
    return stack[(used - 1).clamp(max=len(outs) - 1), torch.arange(stack.shape[1])]


if __name__ == "__main__":
    m = Refiner()
    print(sum(p.numel() for p in m.parameters()), "parameters")
