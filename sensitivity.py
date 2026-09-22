# -*- coding: utf-8 -*-
"""Auxiliary experiments — do not alter the declared 1,600-run benchmark.

1) Disturbance-bound sensitivity: the true disturbance magnitude is raised
   above the declared bound D_bar = 0.12 m/s while every controller keeps
   assuming D_bar (deliberately leaving the operating assumptions).
2) (Delta, alpha) gain-period sweep: RCBF violation rate and SRCBF worst
   clearance on a 9 x 7 grid, mapped against the theoretical curve
   alpha * Delta = 1 from Proposition 1.

Same first-order model, tolerance and termination rules as simulate.py.
Outputs (results/): sensitivity_trials.csv, sensitivity_summary.json,
sensitivity.png, boundary_trials.csv, boundary_summary.json, boundary.png.
"""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / 'results'
V = 1.0; D = .12; R = .8; K = 1.0; TMAX = 30.
METHODS = ['CBF', 'RCBF', 'SCBF', 'SRCBF', 'CAP']
DTS = [.05, .2, .5, .8]
OFFSETS = [.15, .45]
REGIMES = ['bias', 'inward']
NSEED = 20
COLORS = dict(zip(METHODS, ['#D55E00', '#0072B2', '#CC79A7', '#009E73', '#8A6A00']))


def control(p, c, goal, dt, method, alpha, d_assumed):
    r = p - c
    dist = np.linalg.norm(r)
    n = r / dist
    h = dist - R
    v = goal - p
    v *= min(1., V / max(np.linalg.norm(v), 1e-15))
    a = min(alpha, 1 / dt) if method == 'CAP' else \
        (-np.expm1(-alpha * dt) / dt if method in ('SCBF', 'SRCBF') else alpha)
    b = -a * h + (d_assumed if method in ('RCBF', 'SRCBF', 'CAP') else 0.)
    if b > V + 1e-10:
        return None, v
    if n @ v >= b - 1e-12:
        return v, v
    tangent = np.array([-n[1], n[0]])
    z = np.clip(tangent @ v, -np.sqrt(max(0., V * V - b * b)), np.sqrt(max(0., V * V - b * b)))
    return b * n + z * tangent, v


def run(dt, method, offset, regime, seed, alpha=3.0, d_assumed=D, d_true=D):
    rng = np.random.default_rng(seed)
    p = np.array([0., rng.uniform(-.08, .08)])
    phi = rng.uniform(0, 2 * np.pi)
    bias = d_true * np.array([np.cos(phi), np.sin(phi)])
    c = np.array([3., offset])
    goal = np.array([6., 0.])
    t = 0.
    clearance = float(np.linalg.norm(p - c) - R)
    status = 'timeout'
    while t < TMAX - 1e-10:
        step = min(dt, TMAX - t)
        u, _ = control(p, c, goal, step, method, alpha, d_assumed)
        if u is None:
            status = 'infeasible'
            break
        n = (p - c) / np.linalg.norm(p - c)
        disturbance = bias if regime == 'bias' else -d_true * n
        delta = step * (u + disturbance)
        tau = np.clip(-((p - c) @ delta) / max(delta @ delta, 1e-30), 0., 1.)
        segclear = float(np.linalg.norm(p + tau * delta - c) - R)
        clearance = min(clearance, segclear)
        p = p + delta
        t += step
        if clearance < -1e-8:
            status = 'violation'
            break
        if np.linalg.norm(p - goal) <= .15:
            status = 'arrived'
            break
    return dict(method=method, dt=dt, offset=offset, regime=regime, seed=seed,
                alpha=alpha, d_true=d_true, status=status, min_clearance=clearance, time=t)


def summarize(rows, group_keys):
    """violation rate and worst clearance per group."""
    out = {}
    for r in rows:
        k = tuple(r[g] for g in group_keys)
        e = out.setdefault(k, [0, 0, np.inf])
        e[0] += 1
        e[1] += r['status'] == 'violation'
        e[2] = min(e[2], r['min_clearance'])
    return {k: dict(n=v[0], violations=v[1], violation_rate=v[1] / v[0],
                    worst_clearance=v[2]) for k, v in out.items()}


def sensitivity():
    d_trues = list(np.round(np.arange(.12, .29, .02), 2))
    rows = []
    for d_true in d_trues:
        for offset in OFFSETS:
            for regime in REGIMES:
                for dt in DTS:
                    for seed in range(NSEED):
                        for method in METHODS:
                            rows.append(run(dt, method, offset, regime, seed, d_true=d_true))
    with (OUT / 'sensitivity_trials.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows, ['d_true', 'dt', 'method'])
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(summ.items())},
              (OUT / 'sensitivity_summary.json').open('w'), indent=2)

    # sanity: at the declared bound the sweep must reproduce the benchmark counts
    bench = {(0.05, 'CBF'): 55, (0.05, 'SCBF'): 55, (0.5, 'RCBF'): 46, (0.8, 'RCBF'): 74,
             (0.5, 'CBF'): 74, (0.8, 'CBF'): 76, (0.5, 'SCBF'): 49, (0.8, 'SCBF'): 49,
             (0.2, 'CBF'): 53, (0.2, 'SCBF'): 52}
    for (dt, m), expected in bench.items():
        got = summ[(0.12, dt, m)]['violations']
        assert got == expected, f"sanity failed {(dt, m)}: {got} != {expected}"
    assert all(summ[(0.12, dt, 'SRCBF')]['violations'] == 0 for dt in DTS)
    assert all(summ[(0.12, dt, 'CAP')]['violations'] == 0 for dt in DTS)
    print("sanity check vs benchmark: OK")

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, dt in zip(axes, [.5, .8]):
        for m in ['RCBF', 'SRCBF', 'CAP']:
            y = [100 * summ[(d, dt, m)]['violation_rate'] for d in d_trues]
            ax.plot(d_trues, y, 'o-', label=m, color=COLORS[m])
        ax.axvline(D, color='black', ls='--', lw=.7)
        ax.text(D + .004, 2, r'$D_{\mathrm{true}}=\bar D$', fontsize=8)
        ax.set(xlabel=r'True disturbance magnitude (m/s)', ylabel='Safety violations (%)',
               ylim=(-4, 104), title=f'Δ = {dt} s')
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'sensitivity.png')
    plt.close(fig)

    print("--- sensitivity: violation rates (%) ---")
    for dt in DTS:
        line = [f"Δ={dt}"]
        for m in ['RCBF', 'SRCBF', 'CAP']:
            line.append(f"{m} " + " ".join(f"{summ[(d, dt, m)]['violation_rate']*100:.0f}"
                                           for d in d_trues))
        print(" | ".join(line))
    return rows


def boundary():
    alphas = [1., 1.5, 2., 3., 4., 6., 8.]
    dgrid = [.05, .1, .2, .3, .4, .5, .6, .8, 1.0]
    rows = []
    for alpha in alphas:
        for dt in dgrid:
            for offset in OFFSETS:
                for regime in REGIMES:
                    for seed in range(NSEED):
                        for method in ('RCBF', 'SRCBF'):
                            rows.append(run(dt, method, offset, regime, seed, alpha=alpha))
    with (OUT / 'boundary_trials.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows, ['alpha', 'dt', 'method'])
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(summ.items())},
              (OUT / 'boundary_summary.json').open('w'), indent=2)

    def grid(method, key):
        return np.array([[summ[(a, d, method)][key] for d in dgrid] for a in alphas])

    v_rate = 100 * grid('RCBF', 'violation_rate')
    wc = 1e3 * grid('SRCBF', 'worst_clearance')
    assert np.all(v_rate[:, 0] == 0), "RCBF violated at Δ=0.05 inside the curve"
    assert np.all(wc > 0), "SRCBF worst clearance not positive somewhere"

    # cell edges (linear midpoints) for pcolormesh
    xe = [dgrid[0] - (dgrid[1] - dgrid[0]) / 2] + \
         [(dgrid[i] + dgrid[i + 1]) / 2 for i in range(len(dgrid) - 1)] + \
         [dgrid[-1] + (dgrid[1] - dgrid[0]) / 2]
    ye = [alphas[0] - (alphas[1] - alphas[0]) / 2] + \
         [(alphas[i] + alphas[i + 1]) / 2 for i in range(len(alphas) - 1)] + \
         [alphas[-1] + (alphas[1] - alphas[0]) / 2]

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    xc = np.linspace(0.05, 1.0, 300)

    im0 = axes[0].pcolormesh(xe, ye, v_rate, cmap='YlOrRd', vmin=0, vmax=100)
    axes[0].plot(xc, 1 / xc, 'k--', lw=1.2)
    axes[0].text(.78, 7.1, r'$\alpha\Delta=1$', fontsize=9,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
    axes[0].scatter(DTS, [3] * 4, s=30, facecolors='none', edgecolors='black', linewidths=1.2, zorder=5)
    axes[0].set(xlabel='Sampling period Δ (s)', ylabel='Barrier gain α (s$^{-1}$)',
                yticks=alphas, xticks=dgrid, title='RCBF violations (%)')

    im1 = axes[1].pcolormesh(xe, ye, wc, cmap='viridis')
    axes[1].plot(xc, 1 / xc, 'k--', lw=1.2)
    axes[1].scatter(DTS, [3] * 4, s=30, facecolors='none', edgecolors='black', linewidths=1.2, zorder=5)
    axes[1].set(xlabel='Sampling period Δ (s)', ylabel='Barrier gain α (s$^{-1}$)',
                yticks=alphas, xticks=dgrid, title='SRCBF worst clearance (mm)')
    fig.colorbar(im0, ax=axes[0], shrink=.9)
    fig.colorbar(im1, ax=axes[1], shrink=.9)
    fig.tight_layout()
    fig.savefig(OUT / 'boundary.png')
    plt.close(fig)

    print("\n--- boundary: RCBF violation rate (%) rows=α, cols=Δ ---")
    print("α\\Δ", *dgrid)
    for a, row in zip(alphas, v_rate):
        print(a, " ".join(f"{v:5.1f}" for v in row))
    print("\n--- boundary: SRCBF worst clearance (mm) rows=α, cols=Δ ---")
    for a, row in zip(alphas, wc):
        print(a, " ".join(f"{v:6.2f}" for v in row))
    return rows


if __name__ == '__main__':
    sensitivity()
    boundary()
    print("\ndone")
