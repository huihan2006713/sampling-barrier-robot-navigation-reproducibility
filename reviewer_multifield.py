"""Random static multi-obstacle fields and a simple local tangent escape test.

This is a reproducible supplemental stress test, not a deadlock guarantee.
"""
from __future__ import annotations

import json

import numpy as np

from simulate import D, R, TMAX, V
from reviewer_experiments import GOAL, OUT, disk_halfspace_projection, gain, save, segment_min


def run(method, count, seed, escape):
    rng = np.random.default_rng(12000 + 100 * count + seed)
    centers = np.column_stack((np.linspace(1.8, 4.2, count),
                               rng.uniform(-1.25, 1.25, count)))
    p = np.array([0., rng.uniform(-.08, .08)])
    angle = rng.uniform(0., 2 * np.pi)
    bias = D * np.array([np.cos(angle), np.sin(angle)])
    step = .5
    t = 0.
    worst = min(float(np.linalg.norm(p - c) - R) for c in centers)
    status = "timeout"
    slow = 0
    escape_steps = 0
    activated = 0
    while t < TMAX - 1e-10:
        duration = min(step, TMAX-t)
        a = gain(duration, method)
        offsets = p - centers
        lengths = np.linalg.norm(offsets, axis=1)
        normals = offsets / lengths[:, None]
        bounds = D - a * (lengths - R)
        nominal = GOAL - p
        if escape and escape_steps > 0:
            closest = int(np.argmin(lengths - R))
            n = normals[closest]
            tangent = np.array([-n[1], n[0]])
            side = 1 if p[1] >= centers[closest, 1] else -1
            if side * tangent[1] < 0:
                tangent = -tangent
            nominal = tangent + .2 * nominal / max(np.linalg.norm(nominal), 1e-15)
            escape_steps -= 1
        nominal *= min(1., V / max(np.linalg.norm(nominal), 1e-15))
        u = disk_halfspace_projection(nominal, normals, bounds)
        if u is None:
            status = "infeasible"
            break
        delta = duration * (u + bias)
        worst = min(worst, segment_min(p, delta, centers))
        oldx = p[0]
        p += delta
        t += duration
        slow = slow + 1 if p[0] - oldx < .005 else 0
        if escape and slow >= 3 and escape_steps == 0:
            escape_steps = 6
            activated += 1
            slow = 0
        if worst < -1e-8:
            status = "violation"
            break
        if np.linalg.norm(p - GOAL) <= .15:
            status = "arrived"
            break
    return dict(experiment="random_static_field", method=method,
                obstacle_count=count, seed=seed, escape=int(escape),
                status=status, min_clearance=worst, time=t,
                escape_activations=activated)


def main():
    rows = [run(method, count, seed, escape)
            for count in (3, 5) for seed in range(20)
            for method in ("SRCBF", "CAP") for escape in (False, True)]
    summary = save("random_static_field", rows,
                   ("obstacle_count", "method", "escape"))
    (OUT / "random_static_field_protocol.json").write_text(json.dumps(dict(
        centers="x equally spaced on [1.8, 4.2], y independent uniform [-1.25, 1.25]",
        start="(0, uniform[-0.08, 0.08])", seeds="0..19; RNG seed 12000 + 100*count + seed",
        dt=.5, disturbance="constant bias of magnitude 0.12 m/s",
        escape="After three steps of <5 mm x progress, use six steps of a tangent command around the nearest center; no global planner.",
        limitation="Static random circles may overlap; no feasibility or deadlock-avoidance guarantee."), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
