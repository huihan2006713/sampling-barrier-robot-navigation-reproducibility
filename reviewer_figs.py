"""Plot the additional synthetic experiments after reviewer_experiments.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results" / "reviewer"
def read(stem):
    return json.loads((OUT / f"{stem}_summary.json").read_text())

baseline, uncertainty, servo = (read(n) for n in
                                ("matched_baseline", "uncertainty", "velocity_servo"))
periods = [.2, .5, .8]
colors = {"SRCBF": "#007C78", "CAP": "#A87918", "MPC-CBF-3": "#555555"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 240})
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.05))

for method in ("SRCBF", "CAP", "MPC-CBF-3"):
    y = [100 * baseline[f"{dt}|{method}"]["arrived"] / 80 for dt in periods]
    ax[0].plot(periods, y, "o-", label=method, color=colors[method])
ax[0].set(title="A  Goal arrival", ylabel="Arrival (%)", ylim=(-4, 104))

for method in ("SRCBF", "CAP"):
    for corrected, marker, style in ((0, "o", "-"), (1, "s", "--")):
        y = [uncertainty[f"{dt}|0.02|0.04|{method}|{corrected}"]["violations"]
             for dt in periods]
        ax[1].plot(periods, y, marker=marker, ls=style,
                   label=f"{method} {'corrected' if corrected else 'plain'}",
                   color=colors[method])
ax[1].set(title="B  Position error 2 cm, delay 40 ms",
          ylabel="Violations / 80", ylim=(-2, 80))

for method in ("SRCBF", "CAP"):
    for tau, marker, style in ((.1, "o", "-"), (.3, "s", "--")):
        y = [servo[f"{tau}|{dt}|{method}"]["violations"] for dt in periods]
        ax[2].plot(periods, y, marker=marker, ls=style,
                   label=f"{method}, lag {tau} s", color=colors[method])
ax[2].set(title="C  Acceleration-limited velocity servo",
          ylabel="Violations / 80", ylim=(-2, 80))

for axes in ax:
    axes.set(xlabel="Sampling period (s)", xticks=periods)
    axes.grid(axis="y", color="#E4E4E4", lw=.7)
    axes.legend(frameon=False, fontsize=7, loc="best")
fig.tight_layout(w_pad=1.7)
fig.savefig(OUT / "stress_tests.png")
print(OUT / "stress_tests.png")
