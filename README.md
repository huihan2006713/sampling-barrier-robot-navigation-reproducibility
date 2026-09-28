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
- `reviewer_experiments.py` — additional, separate reviewer-response experiments:
  three-step MPC-CBF-style baseline, bounded observation/latency, two circular
  obstacles, and an acceleration-limited velocity servo. Run
  `python reviewer_experiments.py`. Outputs go to `results/reviewer/`.
- `reviewer_figs.py` — rebuilds `results/reviewer/stress_tests.png` from the
  recorded reviewer-experiment summaries.
- `results/` — per-trial records, aggregate summaries, traces and figures.

### Environment
Python 3.12.10, NumPy 2.3.5, Matplotlib (Agg backend). The 1,600-run benchmark
runs in about a minute; the two auxiliary experiments (24,480 runs) take a few
minutes. The reviewer experiments additionally need SciPy. Per-call wall
timings depend on the machine and exclude sensor/communication latency.

### Data
All experiments are synthetic model-based simulations; no external data files
are required. Random seeds are fixed at 0–19 and no failure is discarded.

### Reproduction
```
python simulate.py
python sensitivity.py
python regen_figs.py
python reviewer_experiments.py
python reviewer_figs.py
```
The printed summaries reproduce Tables 3–4 and Figures 1–4 of the paper.

### Reviewer experiment protocol and limits

The original benchmark and its outputs are unchanged. The matched comparison
uses the same two obstacle offsets, two disturbance rules, 20 seeds, and
sampling periods 0.20, 0.50, 0.80 s (80 cases per filter-period). The adapted
`MPC-CBF-3` optimizes three predicted zero-disturbance steps with SLSQP (40
iterations maximum), weighting nominal-command deviation by 0.4, input
smoothness by 0.3 and terminal squared goal distance by 0.4. It uses the same
speed disk and robust SRCBF half-space at every
predicted step. Its *first executed* command is checked against that robust
half-space, and an infeasible numerical solution falls back to SRCBF; fallback
calls are counted. This is a specified local MPC-CBF-style baseline, **not** a
replication of another paper's published controller or an equal-runtime
comparison. All controller times, goal timeouts, and failures are retained.

The uncertainty sweep pairs uncompensated and conservatively compensated
SRCBF/CAP at position-error radii 0 and 0.02 m and actuation delays 0 and
0.04 s. In a delayed run the preceding command persists during the delay.
The correction uses `e = epsilon + (V + D) * latency` and the same barrier
gain, with worst clearance checked on both straight subsegments. The bias
angle, start state and position-error angle are fixed by seed. The one-obstacle
zero-error and zero-delay cases reproduce the original benchmark exactly.

The two-obstacle test uses obstacle-center separation 1.7 or 2.4 m and a
controller-independent bounded bias disturbance, with 20 seeds and periods
0.50/0.80 s. It projects onto the *intersection* of both half-spaces and the
speed disk. At the midpoint between centers for the 1.7 m layout, each
clearance is 0.05 m; the opposed positive half-space requirements are
incompatible at both tested periods. These simulations test a local deadlock
case and do not provide a global deadlock-avoidance method.

The velocity-servo stress test replaces instantaneous velocity execution by
`p_dot = v + d`, `v_dot = (u-v)/tau_v`, capped at 2 m/s², with
`tau_v = 0.1 or 0.3 s`. For each constant-command hold, the saturated and
exponential phases have exact state updates. Clearance is minimized over the
quadratic saturated phase and over all bracketed stationary points of the
exponential phase. The paper's first-order safety proof does not apply to
this model.

These experiments are wholly synthetic. They do not implement a full ZOCBF or
SACBF baseline, a differential-drive plant, hardware execution, online bound
learning, or a safe fallback for infeasible multi-obstacle constraints.
