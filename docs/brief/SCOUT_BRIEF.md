# Scout brief: lucid (2026-11-18 morning)

Self-contained copy of the brief that started this build. The original scout files
(`/workspace/ai-frontier/projects.md`, `reports/2026-11-18-morning.md`, `2026-11-18-morning-post.md`)
were not on the build machine. Go given by Sebastian on 2026-11-18, 13:25 Europe/Paris.

## Idea

A game that exists only inside a looped neural net on a CPU. lucid

1. is a small conv-transformer world model (about 2M parameters) that predicts the next 32x32 arcade frame by looping a shared block;
2. trains each loop toward the nearest legal future, not one stored target: the real simulator enumerates the exact set of possible next frames from the true state (spawn column x type), and the target is the nearest of those;
3. seeds pass 1 with a noise draft so different seeds converge to different legal futures (legal frames become fixed points of the loop);
4. uses a learned energy over the predicted frame to decide when to stop looping, per frame (1 to 8 loops).

The playable artifact is the world model itself. The player steers a paddle with the arrow keys and sees the
model's dream, not a game engine. A one-pass baseline of the same size smears stochastic events.

Status: a new combination; searches on 2026-11-18 for "proximal target world model", "energy halting world model"
and "looped stochastic world model" found no exact match (weak evidence). The report must test the published claim
that adaptive halting does not help world models (https://arxiv.org/abs/2607.10203). A null or a reversal is a result.

## Demo spec (targets, not results)

- Two screens, same seed, same inputs: one-pass model left, lucid right. Caption: "neither of these is a game engine. both are neural nets."
- 0-3 s: a brick breaks; left shows the bonus as a grey smear across four columns, right flickers for a few loops and snaps to one crisp bonus.
- 3-12 s: arrow keys steer; a ring around lucid's screen fills one segment per loop (1-8); a filmstrip colors past frames by loops used.
- 12-22 s: a truth lamp under each screen is white when the dreamed frame equals a future the simulator says is possible, grey otherwise; counters of legal frames in the last 100 steps.
- 22-30 s: spawn histogram vs the simulator's frequencies; end card with real numbers from `results/`. "lucid. a dream you can steer."
- Live play over a websocket from the released checkpoint (`make play`), and replay-from-logs at real speed from committed logs.

## Method

1. Game: NumPy paddle/bricks/falling bonuses with random column and type, 32x32, 8 colors, actions left/stay/right; `legal_futures(state, action)` enumerates the random draws. 500k transitions from scripted and random players.
2. Arms: A one-pass deterministic (cross-entropy per cell); B nearest-of-m one-step (m=16); C looped fixed depth 1/2/4/8, standard target; D looped fixed depth, proximal target; E lucid (proximal target plus energy halting); F the simulator as oracle for the truth lamp.
3. Model: conv-transformer refiner over frame plus action plus noise draft, shared block per loop, energy head, about 2M parameters matched across learned arms, PyTorch on CPU.
4. Metrics: one-step and closed-loop legal-frame rate, spawn total variation, loops vs legal-future count (Spearman), ms per frame and FPS on CPU, all seeds, 95% bootstrap CIs.
5. Compute: 8 vCPU, no GPU. Time a smoke config first (1 seed, 50k transitions, arms A and E, 30 minutes). If the full plan projects over 48 hours, cut to 2 seeds and a 24x24 grid and say so.

## Hypothesis (falsifiable)

At about 2M parameters over 3 seeds: (1) lucid's frames exactly match a legal future in at least 90% of one-step
predictions and at least 80% of steps in 100-step closed-loop play with a scripted player, while a one-pass
deterministic model stays below 40% on spawn frames; (2) lucid matches or beats nearest-of-16 on legal-frame rate at
equal or lower CPU latency per frame; (3) lucid's spawn distribution over 2,000 rollouts is within total variation
0.10 of the simulator's; (4) energy halting puts more loops on frames with more legal futures (Spearman at least 0.5)
and at equal mean loops has at least 10% fewer illegal frames than the best fixed depth. Fails if (1) or (4) misses in
2 of 3 seeds. Mode collapse in (3) or a null in (4) is reported, not hidden.

## Closest prior work (checked 2026-11-18)

- Looped World Models / LoopWM, https://arxiv.org/abs/2606.18208
- Nearest-of-m one-step world models, https://arxiv.org/abs/2611.11887, github.com/thddydgnl/nearest-of-m-world-models
- Proximal Targets, https://arxiv.org/abs/2611.22737
- Energy-Based Reasoners, https://arxiv.org/abs/2611.08977, github.com/wwillchen/ebr
- Adaptive compute in latent world models, https://arxiv.org/abs/2607.10203
- GameNGen; DIAMOND, https://arxiv.org/abs/2405.12399; Neural Game Engine, https://arxiv.org/abs/2003.10520
- Related: FastFrames, Data-Efficient Real-Time World Models (https://arxiv.org/abs/2611.22497), DreamFAR, InfuseLoop
- Own prior work: shoshin (2026-11-10) learned game rules with a tiny world model under surprise gating.

## Rules

CPU only, no paid compute or APIs, no pretrained weights. Real data only; mock fixtures only under `tests/mock_data/`.
MIT plus NOTICE.md. Black-and-white athanor-format README; night-arcade dashboard. Tests plus CI. PR to main, not merged.
