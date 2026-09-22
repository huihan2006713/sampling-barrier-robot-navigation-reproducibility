# Sampling and Disturbance Effects in Barrier-Filtered Mobile Robot Navigation
## Reproduction package

Companion code for the paper "Sampling and disturbance effects in
barrier-filtered mobile robot navigation".

### Contents
- `simulate.py` — the declared 1,600-run benchmark (five filters, two obstacle
  offsets, two disturbance regimes, four sampling periods, seeds 0–19).
  Run `python simulate.py`; it rebuilds `results/trials.csv`,
  `results/summary.json`, Figures 1–2 and the trace files.
- `sensitivity.py` — the two auxiliary experiments: the disturbance-bound
  sensitivity test and the gain–period sweep. Run `python sensitivity.py`; it
  rebuilds `results/sensitivity_*.csv/json` and `results/boundary_*.csv/json`
  and Figures 3–4.
- `regen_figs.py` — regenerates all four figures from the result files.
- `results/` — per-trial records, aggregate summaries, traces and figures.

### Environment
Python 3.12.10, NumPy 2.3.5, Matplotlib (Agg backend). The 1,600-run benchmark
runs in about a minute; the two auxiliary experiments (24,480 runs) take a few
minutes.

### Data
All experiments are synthetic model-based simulations; no external data files
are required. Random seeds are fixed at 0–19 and no failure is discarded.

### Reproduction
```
python simulate.py
python sensitivity.py
python regen_figs.py
```
The printed summaries reproduce Tables 3–4 and Figures 1–4 of the paper.
