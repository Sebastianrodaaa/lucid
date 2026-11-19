PY ?= .venv/bin/python
PORT ?= 8000

.PHONY: setup test play replay results record study
setup:
	python3 -m venv .venv
	$(PY) -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -q numpy pillow websockets
test:
	$(PY) test_lucid.py
play:
	(cd checkpoints && sha256sum -c SHA256SUMS)
	(sleep 2; python3 -m webbrowser http://127.0.0.1:$(PORT)/) &
	$(PY) -m app.server --port $(PORT)
replay:
	(sleep 2; python3 -m webbrowser "http://127.0.0.1:$(PORT)/?replay=demo") &
	$(PY) -m app.server --port $(PORT)
results:
	$(PY) -m lucid.eval --tables
record:
	$(PY) scripts/record_demo.py

STEPS ?= 5000
TRAIN = OMP_NUM_THREADS=4 $(PY) -m lucid.train --steps $(STEPS) --out runs/study-$$s-$$a.pt --seed $$s --arm $$a > runs/study-$$s-$$a.log 2>&1
# The full study, 3 seeds: two 4-thread lanes on 8 vCPU (B, the slowest arm, in its own lane), then evaluation alone.
study:
	mkdir -p runs
	(a=B; for s in 0 1 2; do $(TRAIN); done) & (for s in 0 1 2; do for a in A C E; do $(TRAIN); done; done) & wait
	for s in 0 1 2; do $(PY) -m lucid.eval --prefix study --seed $$s > runs/eval-$$s.log 2>&1; done
