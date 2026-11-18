# Inspirations

lucid is a small combination of published ideas. None of these authors are affiliated with it. Links were checked on 2026-11-18.

- **Looped World Models (LoopWM)**, https://arxiv.org/abs/2606.18208. Looped latent refinement with adaptive depth for world models. Its targets are deterministic; lucid adds many-futures targets.
- **Nearest-of-m one-step world models**, https://arxiv.org/abs/2611.11887 (code: https://github.com/thddydgnl/nearest-of-m-world-models). Avoids mode averaging with one-step IMLE. lucid's arm B is our own re-implementation of the idea at m = 16, not their code.
- **Proximal Targets**, https://arxiv.org/abs/2611.22737. Train a looped refiner toward the nearest correct answer instead of one stored answer, on puzzles. lucid moves this to dynamics, with the simulator enumerating the correct answers.
- **Energy-Based Reasoners**, https://arxiv.org/abs/2611.08977 (code: https://github.com/wwillchen/ebr, MIT). Energy-guided loops and stopping for reasoning. lucid's halting head is a small learned energy over the predicted frame.
- **Adaptive compute in latent world models**, https://arxiv.org/abs/2607.10203. Reports that no trained router beat fixed depth schedules. lucid's halting result (`docs/REPORT.md`, The Halt) is a direct test of that claim on one small game.
- **DIAMOND**, https://arxiv.org/abs/2405.12399, **GameNGen**, and **Neural Game Engine**, https://arxiv.org/abs/2003.10520. Playable neural games: diffusion on GPUs, or deterministic games.
- Related signals: FastFrames; Data-Efficient Real-Time World Models, https://arxiv.org/abs/2611.22497; DreamFAR; InfuseLoop.
- **shoshin** (own earlier work, 2026-11-10): a tiny world model that learned a game's rules under surprise gating. lucid is different: multimodal next-frame prediction with looped proximal targets and energy halting, and you can play it.

Design references for the dashboard are listed in `docs/brief/ui-refs/REFS.md`.
