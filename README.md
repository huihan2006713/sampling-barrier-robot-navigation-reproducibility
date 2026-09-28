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
- `reviewer_timing.py` — paired synthetic update-jitter and missed-update
  stress test with three explicitly defined fallback actions. Run
  `python reviewer_timing.py`; per-trial and aggregate records are stored in
  `results/reviewer/update_jitter_loss_*`.
- `reviewer_multifield.py` — static 3- and 5-obstacle random fields with a
  declared tangent escape heuristic. Run `python reviewer_multifield.py`;
  trial records and summaries go to `results/reviewer/random_static_field_*`.
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
python reviewer_timing.py
python reviewer_multifield.py
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
With a fixed actuation delay at every update, a command remains effective
until the following command takes effect, so its hold duration is one full
sampling period (`H = delta`), not `delta - latency`. The displaced state at
the first effective instant is bounded by `e` above. Thus CAP's `a*H = 1`
at `delta = 0.5 s` satisfies the non-strict sampled-data condition; its
observed safety is also checked directly in the trial records. This argument
requires the declared bounds and feasibility at each update.

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

The update-jitter experiment uses a nominal 0.50 s schedule with independent
uniform jitter of ±0.10 s and a 0.10 probability of a missed controller
update. Both robust filters design for a 0.60 s maximum successful-update
interval. Upon a miss, `hold_previous` keeps the command, `zero_command`
sets it to zero, and `outward_escape` issues a maximum-speed radial outward
command. The latter assumes an exact fresh local position and instantaneously
realizable velocity even when the upstream controller update was missed;
there is no communication or hardware implementation. Trial schedules are
paired between methods and fallbacks. A missed update can extend a held
command beyond the 0.60 s design horizon; a zero command is not necessarily
safe under inward disturbance. The outward action has nondecreasing signed
clearance under the single-obstacle first-order model when `V >= D`, but it
does not address actuator lag, localization error or multi-obstacle feasibility.

The random static-field experiment draws three or five circle centers at
equally spaced x coordinates on [1.8, 4.2] m with independent y coordinates
uniform on [-1.25, 1.25] m, allowing overlaps. Twenty fixed seeds are paired
between the two filters and two nominal policies. When x progress stays below
5 mm for three steps, the escape heuristic uses a nearest-obstacle tangent
target for six steps before returning to the goal direction. It is a local
heuristic and supplies no general feasibility or deadlock-avoidance condition.

These experiments are wholly synthetic. They do not implement a full ZOCBF or
SACBF baseline, a differential-drive plant, hardware execution, online bound
learning, or a safe fallback for infeasible multi-obstacle constraints.
