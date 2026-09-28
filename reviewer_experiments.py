"""Additional synthetic experiments for the reviewer response.

Run with the repository's Python environment: python reviewer_experiments.py
This file does not change the original 1,600-run benchmark. No hardware data.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
import scipy

from simulate import ALPHA, D, R, TMAX, V, control

OUT = Path(__file__).parent / "results" / "reviewer"
OUT.mkdir(parents=True, exist_ok=True)
GOAL = np.array([6., 0.])
OFFSETS = (.15, .45)
REGIMES = ("bias", "inward")
SEEDS = range(20)


def scenario(offset, seed, magnitude=D):
    rng = np.random.default_rng(seed)
    p = np.array([0., rng.uniform(-.08, .08)])
    phi = rng.uniform(0, 2 * np.pi)
    return p, magnitude * np.array([np.cos(phi), np.sin(phi)])


def segment_min(p, delta, centers):
    """Exact minimum Euclidean clearance for each straight hold segment."""
    clear = np.inf
    for c in centers:
        z = p - c
        s = np.clip(-(z @ delta) / max(delta @ delta, 1e-30), 0., 1.)
        clear = min(clear, float(np.linalg.norm(z + s * delta) - R))
    return clear


def accelerated_segment_min(p, velocity, acceleration, duration, center):
    """Exact minimum of the quadratic position curve for frozen acceleration."""
    r = p - center
    w = velocity
    a = acceleration
    coefficients = [.5 * (a @ a), 1.5 * (w @ a), w @ w + r @ a, r @ w]
    roots = np.roots(coefficients)
    times = [0., duration]
    times.extend(float(root.real) for root in roots
                 if abs(root.imag) < 1e-9 and 0 < root.real < duration)
    return min(float(np.linalg.norm(r + w * s + .5 * a * s * s) - R) for s in times)


def gain(dt, method="SRCBF"):
    return -np.expm1(-ALPHA * dt) / dt if method == "SRCBF" else min(ALPHA, 1 / dt)


def disk_halfspace_projection(nom, normals, bounds):
    """Exact Euclidean projection onto the 2-D speed disk and half-spaces.

    Boundary candidates cover the disk arc, each line segment and intersections
    of two line boundaries. Returns None for an empty intersection.
    """
    normals = np.asarray(normals, float)
    bounds = np.asarray(bounds, float)
    def feasible(u):
        return np.linalg.norm(u) <= V + 1e-9 and np.all(normals @ u >= bounds - 1e-9)
    candidates = [np.asarray(nom, float)]
    length = np.linalg.norm(nom)
    candidates.append(nom * min(1., V / max(length, 1e-15)))
    for n, b in zip(normals, bounds):
        if abs(b) > V + 1e-12:
            continue
        tangent = np.array([-n[1], n[0]])
        halfwidth = math.sqrt(max(0., V * V - b * b))
        candidates.append(b * n + np.clip(tangent @ nom, -halfwidth, halfwidth) * tangent)
        candidates.extend([b * n - halfwidth * tangent, b * n + halfwidth * tangent])
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            pair = normals[[i, j]]
            if abs(np.linalg.det(pair)) > 1e-12:
                candidates.append(np.linalg.solve(pair, bounds[[i, j]]))
    valid = [u for u in candidates if feasible(u)]
    return min(valid, key=lambda u: float(np.sum((u - nom) ** 2))) if valid else None


def mpc_cbf(p, c, dt):
    """A declared three-step receding-horizon MPC with robust half-spaces.

    Prediction uses zero disturbance; every first issued command satisfies the
    same robust sampled-data constraint as SRCBF. This is an adapted baseline,
    not a replication of a published MPC-CBF implementation.
    """
    horizon = 3
    a = gain(dt)
    guess = []
    state = p.copy()
    for _ in range(horizon):
        safe, _ = control(state, c, GOAL, dt, "SRCBF")
        if safe is None:
            return None, True, 0
        guess.extend(safe)
        state = state + dt * safe

    def states(x):
        return [p + dt * np.sum(x.reshape(horizon, 2)[:i], axis=0)
                for i in range(horizon + 1)]

    def objective(x):
        uu = x.reshape(horizon, 2)
        pp = states(x)
        deviation = 0.
        for i in range(horizon):
            direction = GOAL - pp[i]
            nominal = direction * min(1., V / max(np.linalg.norm(direction), 1e-15))
            deviation += float(np.sum((uu[i] - nominal) ** 2))
        smooth = sum(float(np.sum((uu[i] - uu[i - 1]) ** 2)) for i in range(1, horizon))
        return .4 * deviation + .3 * smooth + .4 * float(np.sum((pp[-1] - GOAL) ** 2))

    def constraints(x):
        uu = x.reshape(horizon, 2)
        pp = states(x)
        vals = []
        for i in range(horizon):
            r = pp[i] - c
            norm = np.linalg.norm(r)
            if norm < 1e-12:
                return np.full(2 * horizon, -1e3)
            vals.extend([(r / norm) @ uu[i] + a * (norm - R) - D,
                         V * V - uu[i] @ uu[i]])
        return np.array(vals)

    started = time.perf_counter_ns()
    fit = minimize(objective, guess, method="SLSQP", constraints=[{"type": "ineq", "fun": constraints}],
                   options={"maxiter": 40, "ftol": 1e-7, "disp": False})
    elapsed = time.perf_counter_ns() - started
    if np.min(constraints(fit.x)) >= -1e-7:
        return fit.x[:2], False, elapsed
    return np.asarray(guess[:2]), True, elapsed


def benchmark_run(method, dt, offset, regime, seed):
    p, bias = scenario(offset, seed)
    c = np.array([3., offset])
    t = 0.; worst = float(np.linalg.norm(p - c) - R)
    calls = fallback = 0
    duration_ns = 0
    max_call_ns = deadline_misses = 0
    status = "timeout"
    while t < TMAX - 1e-10:
        step = min(dt, TMAX - t)
        if method == "MPC-CBF-3":
            u, failed, elapsed = mpc_cbf(p, c, step)
            fallback += int(failed)
        else:
            started = time.perf_counter_ns()
            u, _ = control(p, c, GOAL, step, method)
            elapsed = time.perf_counter_ns() - started
        duration_ns += elapsed
        max_call_ns = max(max_call_ns, elapsed)
        deadline_misses += int(elapsed > 10_000_000)
        calls += 1
        if u is None:
            status = "infeasible"; break
        n = (p - c) / np.linalg.norm(p - c)
        d = bias if regime == "bias" else -D * n
        delta = step * (u + d)
        worst = min(worst, segment_min(p, delta, [c]))
        p += delta; t += step
        if worst < -1e-8:
            status = "violation"; break
        if np.linalg.norm(p - GOAL) <= .15:
            status = "arrived"; break
    return dict(experiment="matched_baseline", method=method, dt=dt, offset=offset,
                regime=regime, seed=seed, status=status, min_clearance=worst,
                time=t, calls=calls, fallback=fallback, total_call_wall_ms=duration_ns / 1e6,
                mean_call_ms=duration_ns / max(calls, 1) / 1e6,
                max_call_ms=max_call_ns / 1e6,
                deadline_misses_10ms=deadline_misses)


def observed_control(q, c, step, method, error_bound, compensated):
    r = q - c
    dist = np.linalg.norm(r)
    n = r / dist
    h = dist - R - (error_bound if compensated else 0.)
    a = gain(step, method)
    b = D - a * h
    nom = GOAL - q
    nom *= min(1., V / max(np.linalg.norm(nom), 1e-15))
    return disk_halfspace_projection(nom, [n], [b])


def uncertainty_run(method, dt, offset, regime, seed, epsilon, latency, compensated):
    p, bias = scenario(offset, seed)
    rng = np.random.default_rng(5000 + seed)
    c = np.array([3., offset]); t = 0.
    old_u = np.zeros(2)
    worst = float(np.linalg.norm(p - c) - R)
    status = "timeout"
    while t < TMAX - 1e-10:
        step = min(dt, TMAX - t)
        angle = rng.uniform(0, 2 * np.pi)
        measured = p + epsilon * np.array([np.cos(angle), np.sin(angle)])
        error_bound = epsilon + (V + D) * latency
        u = observed_control(measured, c, step, method, error_bound, compensated)
        if u is None:
            status = "infeasible"; break
        # The old command persists until the new command takes effect.
        d = bias if regime == "bias" else -D * (p - c) / np.linalg.norm(p - c)
        first = min(latency, step)
        for command, duration in ((old_u, first), (u, step - first)):
            if duration <= 0:
                continue
            delta = duration * (command + d)
            worst = min(worst, segment_min(p, delta, [c]))
            p += delta
        t += step; old_u = u
        if worst < -1e-8:
            status = "violation"; break
        if np.linalg.norm(p - GOAL) <= .15:
            status = "arrived"; break
    return dict(experiment="uncertainty", method=method, dt=dt, offset=offset,
                regime=regime, seed=seed, epsilon=epsilon, latency=latency,
                compensated=int(compensated), status=status, min_clearance=worst, time=t)


def multi_run(method, dt, gap, seed):
    p, bias = scenario(0., seed)
    centers = [np.array([3., -gap / 2]), np.array([3., gap / 2])]
    t = 0.; worst = min(float(np.linalg.norm(p - c) - R) for c in centers)
    status = "timeout"
    while t < TMAX - 1e-10:
        step = min(dt, TMAX - t)
        a = gain(step, method)
        normals = [(p - c) / np.linalg.norm(p - c) for c in centers]
        bounds = [D - a * (np.linalg.norm(p - c) - R) for c in centers]
        nom = GOAL - p
        nom *= min(1., V / max(np.linalg.norm(nom), 1e-15))
        u = disk_halfspace_projection(nom, normals, bounds)
        if u is None:
            status = "infeasible"; break
        # Controller-independent constant bias is paired across filters.
        delta = step * (u + bias)
        worst = min(worst, segment_min(p, delta, centers))
        p += delta; t += step
        if worst < -1e-8:
            status = "violation"; break
        if np.linalg.norm(p - GOAL) <= .15:
            status = "arrived"; break
    return dict(experiment="two_obstacles", method=method, dt=dt, gap=gap,
                seed=seed, status=status, min_clearance=worst, time=t)


def servo_hold(p, v, u, d, step, tau_v, center, acceleration_limit=2.):
    """Exact saturated-first-order servo state update for one constant command.

    The velocity error retains its direction. It shrinks linearly at the
    acceleration limit and then exponentially. Minimize clearance along both
    continuous pieces, including every bracketed stationary point.
    """
    error = u - v
    size = np.linalg.norm(error)
    if size < 1e-14:
        delta = (v + d) * step
        return p + delta, v.copy(), segment_min(p, delta, [center])
    direction = error / size
    saturated_time = min(step, max(0., (size - acceleration_limit * tau_v) / acceleration_limit))
    worst = float(np.linalg.norm(p - center) - R)
    if saturated_time > 0:
        a = acceleration_limit * direction
        worst = min(worst, accelerated_segment_min(p, v + d, a, saturated_time, center))
        p = p + (v + d) * saturated_time + .5 * a * saturated_time ** 2
        v = v + a * saturated_time
    duration = step - saturated_time
    if duration > 1e-14:
        e = u - v
        def position(s):
            return p + (u + d) * s - tau_v * e * (-np.expm1(-s / tau_v))
        def derivative(s):
            pos = position(s) - center
            vel = u + d - e * np.exp(-s / tau_v)
            return float(pos @ vel)
        grid = np.linspace(0., duration, 17)
        candidates = [0., duration]
        values = [derivative(s) for s in grid]
        for i in range(len(grid) - 1):
            if values[i] * values[i + 1] < 0:
                candidates.append(brentq(derivative, grid[i], grid[i + 1], xtol=1e-12))
        worst = min(worst, *(float(np.linalg.norm(position(s) - center) - R)
                            for s in candidates))
        p = position(duration)
        v = u - e * np.exp(-duration / tau_v)
    return p, v, worst


def lag_run(method, dt, offset, regime, seed, tau_v):
    """Exact acceleration-limited velocity servo; original proof does not apply."""
    p, bias = scenario(offset, seed)
    c = np.array([3., offset]); t = 0.
    v = np.zeros(2); acceleration_limit = 2.0
    worst = float(np.linalg.norm(p - c) - R)
    status = "timeout"
    while t < TMAX - 1e-10:
        step = min(dt, TMAX - t)
        u, _ = control(p, c, GOAL, step, method)
        if u is None:
            status = "infeasible"; break
        d = bias if regime == "bias" else -D * (p - c) / np.linalg.norm(p - c)
        p, v, local_min = servo_hold(p, v, u, d, step, tau_v, c, acceleration_limit)
        worst = min(worst, local_min)
        t += step
        if worst < -1e-8:
            status = "violation"; break
        if np.linalg.norm(p - GOAL) <= .15:
            status = "arrived"; break
    return dict(experiment="velocity_servo", method=method, dt=dt, offset=offset,
                regime=regime, seed=seed, tau_v=tau_v,
                acceleration_limit=acceleration_limit, solver="exact_piecewise_hold",
                status=status, min_clearance=worst, time=t)


def save(name, rows, group):
    with (OUT / f"{name}_trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    totals = {}
    for row in rows:
        key = "|".join(str(row[k]) for k in group)
        s = totals.setdefault(key, dict(n=0, arrived=0, violations=0, infeasible=0,
                                        timeouts=0, worst_clearance=float("inf")))
        field = {"violation": "violations", "timeout": "timeouts",
                 "arrived": "arrived", "infeasible": "infeasible"}[row["status"]]
        s["n"] += 1; s[field] += 1
        s["worst_clearance"] = min(s["worst_clearance"], row["min_clearance"])
    (OUT / f"{name}_summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    return totals


def main():
    (OUT / "environment.json").write_text(json.dumps(dict(
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        timing="perf_counter_ns; isolated solver time; 10-ms budget assessed offline"), indent=2) + "\n")
    b = [benchmark_run(m, dt, o, r, seed)
         for dt in (.2, .5, .8) for o in OFFSETS for r in REGIMES for seed in SEEDS
         for m in ("SRCBF", "CAP", "MPC-CBF-3")]
    bs = save("matched_baseline", b, ("dt", "method"))
    u = [uncertainty_run(m, dt, o, r, seed, eps, delay, comp)
         for dt in (.2, .5, .8) for (eps, delay) in ((0., 0.), (.02, 0.), (0., .04), (.02, .04))
         for o in OFFSETS for r in REGIMES for seed in SEEDS
         for m in ("SRCBF", "CAP") for comp in (False, True)]
    us = save("uncertainty", u, ("dt", "epsilon", "latency", "method", "compensated"))
    mu = [multi_run(m, dt, gap, seed)
          for gap in (1.7, 2.4) for dt in (.5, .8) for seed in SEEDS
          for m in ("SRCBF", "CAP")]
    ms = save("two_obstacles", mu, ("gap", "dt", "method"))
    lag = [lag_run(m, dt, o, r, seed, tau_v)
           for tau_v in (.1, .3) for dt in (.2, .5, .8)
           for o in OFFSETS for r in REGIMES for seed in SEEDS
           for m in ("SRCBF", "CAP")]
    ls = save("velocity_servo", lag, ("tau_v", "dt", "method"))
    for dt in (.2, .5, .8):
        assert bs[f"{dt}|SRCBF"]["violations"] == bs[f"{dt}|CAP"]["violations"] == 0
    print(json.dumps({"baseline": bs, "multi": ms, "lag": ls}, indent=2))
    print("uncertainty groups", len(us), "runs", len(u), "baseline runs", len(b),
          "two-obstacle runs", len(mu), "velocity-servo runs", len(lag))


if __name__ == "__main__":
    main()
