# Shot list

Real footage of the dashboard replaying `results/demo.json.gz` at **1x real speed** (15 ticks per second, as recorded). There is no time-lapse or slow motion. The session uses the released seed-0 checkpoints (`checkpoints/`), a scripted player, and a fresh real start every 100 ticks. Its seed (1) is the first one from 0 where lucid dreams a legal bonus within ticks 8 to 45, so it is a chosen example. Rates over all rollouts are in `docs/REPORT.md`. Video: 60 fps, 4 identical video frames per tick. `scripts/record_demo.py` renders it deterministically from the log.

| file | size |
|---|---|
| `docs/media/lucid-16x9.mp4` | 1920x1080, 30 s (26 s replay + 4 s end card) |
| `docs/media/lucid-9x16.mp4` | 1080x1920, same content |
| `docs/screenshots/demo.gif` | first 12 s of the 16:9 film, grayscale, 960 px, 15 fps (one frame per tick, real speed) |
| `docs/media/hero.png`, `smear.png`, `results.png` | colour stills (B&W copies in `docs/screenshots/`) |

| time | tick | what happens |
|---|---|---|
| 0:00 | 0 | Two screens, same start, same inputs. Caption: "neither of these is a game engine. both are neural nets." |
| 0:01.4 | 21 | A brick breaks: 8 possible bonuses. Left (one-pass): smear, lamp grey. Right (lucid): one bonus in one column, lamp white. |
| 0:04.3 | 64 | Second bonus: one-pass grey, lucid white. |
| 0:06.7 | 100 | Restart from a new real frame. |
| 0:07.0 | 105 | Bonus: one-pass grey, lucid white. |
| 0:08.5 | 127 | Ball respawn, 16 possible columns: one-pass grey, lucid white. |
| 0:09.4 | 141 | Bonus: both lamps grey (lucid also misses). |
| 0:12.3, 0:15.9, 0:18.8 | 185, 239, 282 | Bonuses: both lamps grey. |
| 0:13.3, 0:20.0 | 200, 300 | Restarts. |
| 0:21.2 | 318 | Respawn: one-pass grey, lucid white. |
| 0:22.1, 0:25.0 | 332, 375 | Bonuses: both grey. |
| 0:26 to 0:30 | card | End card with the numbers from `results/`: legal-frame rate, CPU FPS, mean loops. "lucid. a dream you can steer." |

Over the 390 replayed ticks, lucid's lamp is lit on 225 and one-pass's on 136. The loop ring shows 1 of 8 on every frame, because energy halting stopped after one loop throughout this session (see The Halt in the README). The filmstrip and the lamp rows fill from left to right. The bottom bar shows the bonus-column histogram from 2,000 rollouts in `results/study-0.json`, the legal counters for the last 100 steps, the loops per frame, and the measured ms per frame.

The ring never jumps to 6 to 8 loops on spawns, as the original script hoped. That is the result, and the footage shows it as it is.
