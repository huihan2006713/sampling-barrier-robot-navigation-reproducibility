"""Synthetic update-jitter and missed-update stress test (no hardware data).

The controller uses the declared 0.60 s maximum update interval for its gain.
The simulator varies the actual update interval and compares three explicitly
defined responses to a missed update. Only the outward response assumes a
fresh local position measurement during a missed remote update.
"""
from __future__ import annotations

import json

import numpy as np

from simulate import D, R, TMAX, V, control
from reviewer_experiments import GOAL, OFFSETS, OUT, REGIMES, SEEDS, scenario, segment_min, save


def run(method, offset, regime, seed, fallback):
    p, bias = scenario(offset, seed)
    c = np.array([3., offset])
    # The schedule and loss draws are paired between methods and fallbacks.
    stream = np.random.default_rng(9000 + seed + 100 * int(offset > .3)
                                   + 1000 * int(regime == "inward"))
    nominal_dt = .5
    design_horizon = .6
    t = 0.
    command = np.zeros(2)
    worst = float(np.linalg.norm(p - c) - R)
    status = "timeout"
    missed = 0
    max_interval = 0.
    while t < TMAX - 1e-10:
        interval = min(nominal_dt + stream.uniform(-.1, .1), TMAX - t)
        max_interval = max(max_interval, interval)
        if stream.random() < .1:
            missed += 1
            if fallback == "zero_command":
                command = np.zeros(2)
            elif fallback == "outward_escape":
                command = V * (p - c) / np.linalg.norm(p - c)
        else:
            next_command, _ = control(p, c, GOAL, design_horizon, method)
            if next_command is None:
                status = "infeasible"
                break
            command = next_command
        n = (p - c) / np.linalg.norm(p - c)
        d = bias if regime == "bias" else -D * n
        delta = interval * (command + d)
        worst = min(worst, segment_min(p, delta, [c]))
        p += delta
        t += interval
        if worst < -1e-8:
            status = "violation"
            break
        if np.linalg.norm(p - GOAL) <= .15:
            status = "arrived"
            break
    return dict(experiment="update_jitter_loss", method=method, fallback=fallback,
                dt=nominal_dt, design_horizon=design_horizon,
                jitter_bound=.1, loss_probability=.1,
                offset=offset, regime=regime, seed=seed, status=status,
                min_clearance=worst, time=t, missed=missed,
                max_realized_interval=max_interval)


def main():
    trials = [run(method, offset, regime, seed, fallback)
              for method in ("SRCBF", "CAP")
              for fallback in ("hold_previous", "zero_command", "outward_escape")
              for offset in OFFSETS for regime in REGIMES for seed in SEEDS]
    summary = save("update_jitter_loss", trials, ("method", "fallback"))
    (OUT / "update_jitter_loss_protocol.json").write_text(json.dumps(dict(
        nominal_dt=.5, design_horizon=.6, jitter_bound=.1, loss_probability=.1,
        disturbance_bound=D, schedule_seed_rule="9000 + seed + 100*(offset>0.3) + 1000*(regime==inward)",
        missed_update="No filter update: hold, issue zero velocity, or issue a local outward command using a fresh position measurement.",
        note="Synthetic, paired, ideal velocity actuation; no physical or embedded timing data."), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
