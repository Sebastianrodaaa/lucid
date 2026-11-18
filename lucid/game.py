"""A 32x32 brick-and-paddle game whose random draws can be enumerated exactly."""
from collections import namedtuple

import numpy as np

N, COLORS = 32, 8
BG, WALL, PADDLE, BALL, BRICK_A, BRICK_B, BONUS_A, BONUS_B = range(COLORS)
LEFT, STAY, RIGHT = range(3)
BRICK_ROWS, BRICK_COLS, BRICK_W, BRICK_X = range(4, 8), 7, 4, 3
PADDLE_ROW, PADDLE_W, RESPAWN_ROW, RESPAWN_COLS = 30, 5, 20, range(7, 23)
FULL = (1 << (len(BRICK_ROWS) * BRICK_COLS)) - 1

# bricks: bitmask, paddle: left column, ball: (row, col, d_row, d_col), bonuses: ((row, col, color), ...)
State = namedtuple("State", "bricks paddle ball bonuses")


def brick_at(bricks, r, c):
    if r not in BRICK_ROWS or not BRICK_X <= c < BRICK_X + BRICK_COLS * BRICK_W:
        return None
    i = (r - BRICK_ROWS[0]) * BRICK_COLS + (c - BRICK_X) // BRICK_W
    return i if bricks >> i & 1 else None


def render(s):
    f = np.zeros((N, N), np.uint8)
    f[0], f[:, 0], f[:, -1] = WALL, WALL, WALL
    for i in range(len(BRICK_ROWS) * BRICK_COLS):
        if s.bricks >> i & 1:
            r, x = BRICK_ROWS[0] + i // BRICK_COLS, BRICK_X + i % BRICK_COLS * BRICK_W
            f[r, x:x + BRICK_W] = BRICK_A + r % 2
    f[PADDLE_ROW, s.paddle:s.paddle + PADDLE_W] = PADDLE
    for r, c, color in s.bonuses:
        f[r, c] = color
    r, c = s.ball[:2]
    f[r, c] = BALL
    return f


def _advance(s, action):
    """Deterministic part of a step, plus the pending random event or None."""
    paddle = min(max(s.paddle + action - 1, 1), N - 1 - PADDLE_W)
    bonuses = []
    for r, c, color in s.bonuses:
        r += 1
        if r < PADDLE_ROW or (r == PADDLE_ROW and not paddle <= c < paddle + PADDLE_W):
            bonuses.append((r, c, color))
    r, c, dr, dc = s.ball
    if c + dc in (0, N - 1):
        dc = -dc
    if r + dr == 0:
        dr = -dr
    if r + dr == PADDLE_ROW and paddle <= c + dc < paddle + PADDLE_W:
        dr = -dr
    bricks, event = s.bricks, None
    for hr, hc, fr, fc in ((r + dr, c, -1, 1), (r, c + dc, 1, -1), (r + dr, c + dc, -1, -1)):
        hit = brick_at(bricks, hr, hc)
        if hit is not None:
            bricks &= ~(1 << hit)
            dr, dc = dr * fr, dc * fc
            event = ("bonus", hr + 1, BRICK_X + hit % BRICK_COLS * BRICK_W)
            break
    tr, tc = r + dr, c + dc
    blocked = tc in (0, N - 1) or tr == 0 or brick_at(bricks, tr, tc) is not None or (
        tr == PADDLE_ROW and paddle <= tc < paddle + PADDLE_W)
    if not blocked:
        r, c = tr, tc
    if r == N - 1:
        event = ("respawn",)
    return State(bricks or FULL, paddle, (r, c, dr, dc), tuple(bonuses)), event


def _outcomes(event):
    if event is None:
        return [None]
    if event[0] == "bonus":
        return [(j, color) for j in range(BRICK_W) for color in (BONUS_A, BONUS_B)]
    return list(RESPAWN_COLS)


def _apply(s, event, outcome):
    if event is None:
        return s
    if event[0] == "bonus":
        j, color = outcome
        return s._replace(bonuses=s.bonuses + ((event[1], event[2] + j, color),))
    return s._replace(ball=(RESPAWN_ROW, outcome, -1, 1 if outcome < 15 else -1))


def legal_futures(s, action):
    """Every possible next state, each equally likely. The simulator is the only oracle."""
    det, event = _advance(s, action)
    return [_apply(det, event, o) for o in _outcomes(event)]


def step(s, action, rng):
    futures = legal_futures(s, action)
    return futures[rng.integers(len(futures))]


def random_state(rng):
    mask = sum(1 << i for i in range(len(BRICK_ROWS) * BRICK_COLS) if rng.random() < rng.uniform(0.4, 1))
    ball = (int(rng.integers(10, 26)), int(rng.integers(2, N - 2)), int(rng.choice([-1, 1])), int(rng.choice([-1, 1])))
    return State(mask or FULL, int(rng.integers(1, N - PADDLE_W)), ball, ())


def scripted(s, rng, eps=0.1):
    """Track the ball with the paddle centre; act randomly with probability eps."""
    if rng.random() < eps:
        return int(rng.integers(3))
    centre = s.paddle + PADDLE_W // 2
    return LEFT if s.ball[1] < centre else RIGHT if s.ball[1] > centre else STAY


def nearest(frame, futures):
    """Index of the legal future closest to a frame in cell (Hamming) distance."""
    frames = np.stack([render(f) for f in futures])
    return int((frames != frame).reshape(len(futures), -1).sum(1).argmin()), frames


def generate(n, seed, scripted_share=0.5, episode=200):
    """n real transitions: prev, cur, action, next, and the legal-future frames for random steps."""
    rng = np.random.default_rng(seed)
    prev, cur, nxt = (np.empty((n, N, N), np.uint8) for _ in range(3))
    act = np.empty(n, np.int64)
    futures, owner = [], np.full(n, -1, np.int64)
    s, t, eps = None, episode, 0.1
    for i in range(n):
        if t == episode:
            s, t = random_state(rng), 0
            eps = 0.1 if rng.random() < scripted_share else 1.0
            last, s = render(s), step(s, scripted(s, rng, eps), rng)
        a = scripted(s, rng, eps)
        legal = legal_futures(s, a)
        nxt_s = legal[rng.integers(len(legal))]
        prev[i], cur[i], act[i], nxt[i] = last, render(s), a, render(nxt_s)
        if len(legal) > 1:
            owner[i] = len(futures)
            futures.append(np.stack([render(f) for f in legal]))
        last, s, t = cur[i], nxt_s, t + 1
    return dict(prev=prev, cur=cur, act=act, next=nxt, owner=owner, futures=futures)


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = generate(20000, 0)
    k = np.array([len(f) for f in d["futures"]])
    print(f"{time.time() - t0:.1f}s; random steps {len(k) / 20000:.3f}; counts", np.unique(k, return_counts=True))
