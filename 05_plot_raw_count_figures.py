"""
clean7_raw_count_figures.py
============================
Generate raw record count distribution figures for the manuscript.

Figure 1: Single panel — 333 combined (single+avg) records by local time.
Figure 2: 2x2 grid — 333, 444, 555, 666 combined records, same format.

Visual style matches new_fig_cognitive_gradient.png:
  same fonts, colors, axis style, gridspec layout.

Outputs (300 DPI):
  figures/figure_raw_counts_333.png
  figures/figure_raw_counts_bigcubes.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

os.makedirs("figures", exist_ok=True)

# ── Style constants (match cognitive gradient) ────────────────────────────────
EVENT_COLORS = {
    "333": "#2166ac",
    "444": "#4dac26",
    "555": "#d6604d",
    "666": "#762a83",
}
EVENT_LABELS = {
    "333": "3x3x3 Cube",
    "444": "4x4x4 Cube",
    "555": "5x5x5 Cube",
    "666": "6x6x6 Cube",
}

RECORD_COL = "n_records"
WR_COL     = "n_wr"
CR_COL     = "n_cr"
NR_COL     = "n_nr"

XMIN = 6.0    # 06:00
XMAX = 22.0   # 22:00
BIN_W = 0.22  # bar width slightly narrower than 0.25 for spacing

# matplotlib global style
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    12,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.6,
    "grid.linestyle":    "--",
})


def make_xticks(step=3):
    hrs = np.arange(0, 25, step)
    return hrs, [f"{int(h):02d}:00" for h in hrs]


def load_bins(event):
    """Load round table, aggregate single record counts by 15-min bin."""
    df = pd.read_csv(f"new_round_table_{event}.csv")
    ds = df.dropna(subset=["local_hour"]).copy()
    ds["bin_15min"] = (ds["local_hour"] // 0.25) * 0.25

    grp = ds.groupby("bin_15min").agg(
        n_rec = (RECORD_COL, "sum"),
        n_wr  = (WR_COL,     "sum"),
        n_cr  = (CR_COL,     "sum"),
        n_nr  = (NR_COL,     "sum"),
    ).reset_index()

    total_rec = int(ds[RECORD_COL].sum())
    total_wr  = int(ds[WR_COL].sum())
    total_cr  = int(ds[CR_COL].sum())
    total_nr  = int(ds[NR_COL].sum())

    return grp, total_rec, total_wr, total_cr, total_nr


def draw_panel(ax, grp, event, total_rec, total_wr, total_cr, total_nr,
               show_xlabel=True, show_ylabel=True):
    """Draw a single histogram panel onto ax."""
    color = EVENT_COLORS[event]
    label = EVENT_LABELS[event]

    # Filter to display window
    grp_plot = grp[(grp["bin_15min"] >= XMIN) & (grp["bin_15min"] < XMAX)].copy()

    ax.bar(grp_plot["bin_15min"], grp_plot["n_rec"],
           width=BIN_W, color=color, alpha=0.85, linewidth=0.3,
           edgecolor="white", align="edge")

    # Y axis: integer ticks only
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

    # X axis
    xt, xl = make_xticks(3)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(bottom=0)

    if show_xlabel:
        ax.set_xlabel("Local time of day (round start)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Number of record-setting solves", fontsize=9)

    # Title with n and breakdown
    ax.set_title(
        f"{label}\nn = {total_rec:,}  (WR={total_wr}, CR={total_cr}, NR={total_nr})",
        fontsize=12, color=color, fontweight="bold", pad=6,
    )

    # Annotation: n in top-right corner (redundant but clear for print)
    ax.text(0.98, 0.97,
            f"n = {total_rec:,}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=color, alpha=0.85, linewidth=0.8))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 333 single panel
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 1 (333 raw counts)...")

grp333, n333, wr333, cr333, nr333 = load_bins("333")

fig1, ax1 = plt.subplots(figsize=(10, 5))

draw_panel(ax1, grp333, "333", n333, wr333, cr333, nr333,
           show_xlabel=True, show_ylabel=True)

ax1.set_title(
    f"3x3x3 Cube — raw single record count by local time of day\n"
    f"n = {n333:,}  (WR={wr333}, CR={cr333}, NR={nr333})",
    fontsize=12, color=EVENT_COLORS["333"], fontweight="bold", pad=8,
)

fig1.text(0.5, -0.04,
          "15-minute bins | local clock time of scheduled round start | singles + averages combined",
          ha="center", fontsize=8, color="#666666")

plt.tight_layout()
fig1.savefig("figures/figure_raw_counts_333.png", dpi=300, bbox_inches="tight")
plt.close(fig1)
print("  Saved: figures/figure_raw_counts_333.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — 444 / 555 / 666 (2x2 grid, 4th panel hidden)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 2 (333/444/555/666 raw counts)...")

data = {}
for event in ["333", "444", "555", "666"]:
    data[event] = load_bins(event)

fig2 = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.52, wspace=0.32)

positions = {"333": (0, 0), "444": (0, 1), "555": (1, 0), "666": (1, 1)}

for event, (row, col) in positions.items():
    ax = fig2.add_subplot(gs[row, col])
    grp, n_rec, n_wr, n_cr, n_nr = data[event]

    show_xl = (row == 1)   # x label only on bottom row
    show_yl = (col == 0)   # y label only on left column

    draw_panel(ax, grp, event, n_rec, n_wr, n_cr, n_nr,
               show_xlabel=show_xl, show_ylabel=show_yl)

# Shared super-title
fig2.suptitle(
    "Raw record count by local time of day — 3×3×3, 4×4×4, 5×5×5, 6×6×6",
    fontsize=13, y=1.01, fontweight="bold", color="#222222",
)

fig2.text(0.5, -0.01,
          "15-minute bins | local clock time of scheduled round start | "
          "singles + averages combined | no exposure adjustment",
          ha="center", fontsize=8, color="#666666")

plt.savefig("figures/figure_raw_counts_bigcubes.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("  Saved: figures/figure_raw_counts_bigcubes.png")

print("\nDone.")
