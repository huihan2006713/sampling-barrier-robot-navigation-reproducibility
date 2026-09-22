# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

OUT = Path(__file__).parent / 'results'
DTS = [0.05, 0.2, 0.5, 0.8]
METHODS = ['CBF', 'SCBF', 'RCBF', 'SRCBF', 'CAP']
colors = ['#D55E00', '#0072B2', '#CC79A7', '#009E73', '#8A6A00']
MARKERS = {'CBF': 'x', 'SCBF': 'D', 'RCBF': '^', 'SRCBF': 'o', 'CAP': 's'}
LSTYLES = {'CBF': '-', 'SCBF': '--', 'RCBF': '-.', 'SRCBF': '-', 'CAP': '--'}
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 220, 'lines.linewidth': 1.7, 'lines.markersize': 5.5,
                     'axes.linewidth': 0.9})

# ---------- Figure 1: sampling ----------
summary = json.loads((OUT / 'summary.json').read_text())
fig, ax = plt.subplots(1, 2, figsize=(6.9, 3.3))
handles = []
for method in METHODS:
    color = colors[METHODS.index(method)]
    rr = [r for r in summary if r['method'] == method]
    h, = ax[0].plot(DTS, [100 * r['violations'] / r['n'] for r in rr],
                    marker=MARKERS[method], ls=LSTYLES[method], label=method, color=color)
    ax[1].plot(DTS, [r['worst_clearance'] for r in rr],
               marker=MARKERS[method], ls=LSTYLES[method], color=color)
    handles.append(h)
ax[0].set(ylabel='Safety violations (%)', xlabel='Sampling period (s)', ylim=(-4, 104))
ax[1].set(ylabel='Worst signed clearance (m)', xlabel='Sampling period (s)')
ax[1].axhline(0, color='black', lw=1.0, ls='--')
# 右图放大内嵌图：SRCBF vs CAP 最差余量（毫米）
ins = ax[1].inset_axes([0.60, 0.18, 0.37, 0.38])
for method in ['SRCBF', 'CAP']:
    color = colors[METHODS.index(method)]
    rr = [r for r in summary if r['method'] == method]
    ins.plot(DTS, [1e3 * r['worst_clearance'] for r in rr],
             marker=MARKERS[method], ls=LSTYLES[method], color=color, lw=1.4, ms=4.5)
ins.set(ylabel='(mm)', xlabel='Δ (s)', title='Worst clearance (zoom)',
        xticks=DTS)
ins.tick_params(labelsize=8)
ins.set_title('Worst clearance (zoom)', fontsize=8)
ins.set_ylim(bottom=0)
fig.legend(handles, METHODS, loc='upper center', ncol=5, frameon=False, fontsize=12,
           bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(OUT / 'sampling.png')
plt.close(fig)
print('Figure 1 regenerated')

# ---------- Figure 2: trajectories ----------
rows = {}
for method in METHODS:
    p = OUT / ('trace_%s.csv' % method)
    if p.exists():
        rows[method] = [(float(v[0]), float(v[1]), float(v[2]), float(v[3]))
                        for v in (l.split(',') for l in p.read_text().splitlines()[1:])]
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.9))
ax[0].add_patch(plt.Circle((3, .15), 0.80, color='#DDDDDD'))
ax[0].add_patch(plt.Circle((3, .15), .5, color='#999999'))
handles = []
for method in METHODS:
    if method not in rows:
        continue
    hist = np.array(rows[method])
    color = colors[METHODS.index(method)]
    h, = ax[0].plot(hist[:, 1], hist[:, 2], label=method, color=color, lw=1.6,
                    ls=LSTYLES[method], marker=MARKERS[method], ms=4.5, markevery=20)
    ax[0].plot(hist[-1, 1], hist[-1, 2], 'x', color=color, ms=9, mew=2)
    ax[1].plot(hist[:, 0], hist[:, 3], color=color, lw=1.6, ls=LSTYLES[method],
               marker=MARKERS[method], ms=4.5, markevery=20)
    handles.append(h)
ax[0].plot(6, 0, 'k*', ms=14)
ax[0].set(aspect='equal', xlabel='x (m)', ylabel='y (m)', xlim=(-.2, 6.3), ylim=(-1.4, 1.3))
ax[1].set(xlabel='Time (s)', ylabel='Endpoint clearance (m)')
ax[1].axhline(0, color='black', ls='--', lw=1.0)
fig.legend(handles, METHODS, loc='upper center', ncol=5, frameon=False, fontsize=11.5,
           bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.82))
fig.savefig(OUT / 'trajectories.png')
plt.close(fig)
print('Figure 2 regenerated')

# ---------- Figure 3: boundary heatmaps ----------
bsumm = {tuple(k.split('|')): v for k, v in json.loads((OUT / 'boundary_summary.json').read_text()).items()}
alphas = [1., 1.5, 2., 3., 4., 6., 8.]
dgrid = [.05, .1, .2, .3, .4, .5, .6, .8, 1.0]


def grid(method, key):
    return np.array([[bsumm[(f"{a}", f"{d}", method)][key] for d in dgrid] for a in alphas])


v_rate = 100 * grid('RCBF', 'violation_rate')
wc = 1e3 * grid('SRCBF', 'worst_clearance')
xe = [dgrid[0] - (dgrid[1] - dgrid[0]) / 2] + \
     [(dgrid[i] + dgrid[i + 1]) / 2 for i in range(len(dgrid) - 1)] + \
     [dgrid[-1] + (dgrid[1] - dgrid[0]) / 2]
ye = [alphas[0] - (alphas[1] - alphas[0]) / 2] + \
     [(alphas[i] + alphas[i + 1]) / 2 for i in range(len(alphas) - 1)] + \
     [alphas[-1] + (alphas[1] - alphas[0]) / 2]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
xc = np.linspace(0.05, 1.0, 300)
xt = [.05, .2, .4, .6, .8, 1.0]
yt = [1., 2., 3., 4., 6., 8.]
im0 = axes[0].pcolormesh(xe, ye, v_rate, cmap='YlOrRd', vmin=0, vmax=100)
axes[0].plot(xc, 1 / xc, 'k--', lw=1.6)
axes[0].scatter(DTS, [3] * 4, s=52, facecolors='none', edgecolors='black', linewidths=1.4, zorder=5)
axes[0].set(xlabel='Sampling period Δ (s)', ylabel='Barrier gain α (s$^{-1}$)',
            yticks=yt, xticks=xt, ylim=(0.9, 8.2),
            title='RCBF violations (%)')
im1 = axes[1].pcolormesh(xe, ye, wc, cmap='viridis', norm=LogNorm(vmin=wc.min(), vmax=wc.max()))
axes[1].plot(xc, 1 / xc, 'k--', lw=1.6)
axes[1].scatter(DTS, [3] * 4, s=52, facecolors='none', edgecolors='black', linewidths=1.4, zorder=5)
axes[1].set(xlabel='Sampling period Δ (s)', ylabel='Barrier gain α (s$^{-1}$)',
            yticks=yt, xticks=xt, ylim=(0.9, 8.2),
            title='SRCBF worst clearance (mm, log)')
for a in axes:
    a.tick_params(labelsize=10)
cb0 = fig.colorbar(im0, ax=axes[0], shrink=.85, pad=.04)
cb0.ax.tick_params(labelsize=10)
cb1 = fig.colorbar(im1, ax=axes[1], shrink=.85, pad=.04)
cb1.ax.tick_params(labelsize=10)
fig.tight_layout(w_pad=1.4)
fig.savefig(OUT / 'boundary.png')
plt.close(fig)
print('Figure 3 regenerated')

# ---------- Figure 4: sensitivity ----------
summ = {tuple(k.split('|')): v for k, v in json.loads((OUT / 'sensitivity_summary.json').read_text()).items()}
d_trues = sorted({float(k[0]) for k in summ})
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
handles = []
for ax, dt in zip(axes, [.5, .8]):
    for m in ['RCBF', 'SRCBF', 'CAP']:
        y = [100 * summ[(f"{d}", f"{dt}", m)]['violation_rate'] for d in d_trues]
        if dt == .5:
            h, = ax.plot(d_trues, y, marker=MARKERS[m], ls=LSTYLES[m], label=m,
                         color=colors[METHODS.index(m)])
            handles.append(h)
        else:
            ax.plot(d_trues, y, marker=MARKERS[m], ls=LSTYLES[m],
                    color=colors[METHODS.index(m)])
    ax.axvline(0.12, color='black', ls='--', lw=1.0)
    ax.text(0.124, 2, r'$D_{\mathrm{true}}=\bar D$', fontsize=10)
    ax.set(xlabel='True disturbance (m/s)', ylabel='Safety violations (%)',
           ylim=(-4, 104), title=f'Δ = {dt} s')
    ax.set_xticks([0.12, 0.16, 0.20, 0.24, 0.28])
    ax.tick_params(labelsize=11)
fig.legend(handles, ['RCBF', 'SRCBF', 'CAP'], loc='upper center', ncol=3, frameon=False,
           fontsize=12, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT / 'sensitivity.png', bbox_inches='tight')
plt.close(fig)
print('Figure 4 regenerated')
