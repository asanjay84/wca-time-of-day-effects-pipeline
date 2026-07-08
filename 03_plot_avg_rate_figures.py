"""
clean5_avg_analysis_multi.py
============================
Run exposure-adjusted time-of-day analysis for AVERAGE (Ao5/Mo3) records
across 333, 444, 555, 666.

  - 333, 444, 555: Average = Ao5 (average of 5 solves)
  - 666:           Average = Mo3 (mean of 3 solves)

For each event:
  - Adaptive rate scale (chosen so overall rate displays as ~10-30 per N)
  - Adaptive minimum-attempts filter per bin
  - Within-competition permutation test (10,000 reps)
  - Poisson regression

Combined figures:
  - avg_fig_cognitive_gradient.png  (333, 444, 555, 666 in one page)

Outputs per event:
  avg_fig_{event}_rate.png
  avg_permutation_{event}.csv
  avg_regression_{event}.csv
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
EVENTS    = ["333", "444", "555", "666"]
N_PERMS   = 10_000
RNG_SEED  = 42

RECORD_COL = "n_avg_rec"
WR_COL     = "n_wr_avg"
CR_COL     = "n_cr_avg"
NR_COL     = "n_nr_avg"

EVENT_LABELS = {
    "333": "3x3x3 Cube (Ao5)",
    "444": "4x4x4 Cube (Ao5)",
    "555": "5x5x5 Cube (Ao5)",
    "666": "6x6x6 Cube (Mo3)",
}
EVENT_COLORS = {
    "333": "#2166ac",
    "444": "#4dac26",
    "555": "#d6604d",
    "666": "#762a83",
}
NULL_COLOR = "#d9d9d9"
NULL_EDGE  = "#999999"

# ============================================================
# HELPERS
# ============================================================
def choose_scale(total_records, total_attempts):
    if total_attempts == 0 or total_records == 0:
        return 10_000
    overall = total_records / total_attempts
    raw = 20 / overall
    for nice in [100, 200, 500, 1_000, 2_000, 5_000, 10_000,
                 20_000, 50_000, 100_000, 200_000, 500_000]:
        if raw <= nice * 1.5:
            return nice
    return 100_000


def choose_min_attempts(total_attempts, n_bins_with_data):
    floor_val = max(100, int(total_attempts * 0.005))
    return min(floor_val, 10_000)


def poisson_ci(k, n, scale, alpha=0.05):
    if n == 0:
        return 0.0, 0.0
    lo = stats.chi2.ppf(alpha / 2,  2 * k)         / (2 * n) if k > 0 else 0.0
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / (2 * n)
    return lo * scale, hi * scale


def bin_aggregate(data, record_col=RECORD_COL, scale=100_000):
    grp = data.groupby("bin_15min").agg(
        n_records  = (record_col, "sum"),
        n_attempts = ("n_attempts", "sum"),
    ).reset_index()
    grp = grp[grp["n_attempts"] > 0].copy()
    grp["rate"]        = grp["n_records"] / grp["n_attempts"]
    grp["rate_scaled"] = grp["rate"] * scale
    cis = [poisson_ci(int(r["n_records"]), r["n_attempts"], scale)
           for _, r in grp.iterrows()]
    grp["ci_lo"] = [c[0] for c in cis]
    grp["ci_hi"] = [c[1] for c in cis]
    return grp


def run_permutation(df_s, scale, min_att, n_perms=N_PERMS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    bins_all    = bin_aggregate(df_s, scale=scale)
    att_by_bin  = bins_all.set_index("bin_15min")["n_attempts"].to_dict()

    comp_rounds = df_s[["competitionId", "bin_15min", "n_attempts", RECORD_COL]].copy()
    comp_rounds = comp_rounds.rename(columns={RECORD_COL: "n_records"})
    comp_rounds["bin_idx"] = (comp_rounds["bin_15min"] / 0.25).round().astype(int)

    comp_groups = []
    for _, grp in comp_rounds.groupby("competitionId"):
        K = int(grp["n_records"].sum())
        if K == 0:
            continue
        att = grp["n_attempts"].values.astype(float)
        N   = att.sum()
        if N == 0:
            continue
        comp_groups.append((K, grp["bin_idx"].values, att / N))

    denom      = np.array([att_by_bin.get(b * 0.25, 0) for b in range(96)], dtype=float)
    denom_safe = np.where(denom > 0, denom, np.nan)

    obs = np.zeros(96)
    for _, row in bins_all.iterrows():
        b = round(row["bin_15min"] / 0.25)
        if 0 <= b < 96:
            obs[b] = row["rate_scaled"]

    perm_records = np.zeros((n_perms, 96))
    for i in range(n_perms):
        null_rec = np.zeros(96)
        for K, bins_arr, weights in comp_groups:
            counts = rng.multinomial(K, weights)
            np.add.at(null_rec, bins_arr, counts)
        perm_records[i] = null_rec

    perm_rates = (perm_records / denom_safe[None, :]) * scale

    perm_mean = np.nanmean(perm_rates, axis=0)
    perm_lo   = np.nanpercentile(perm_rates,  2.5, axis=0)
    perm_hi   = np.nanpercentile(perm_rates, 97.5, axis=0)

    p_vals = np.full(96, np.nan)
    for b in range(96):
        if denom[b] >= min_att:
            null_b = perm_rates[:, b]
            null_b = null_b[~np.isnan(null_b)]
            if len(null_b) > 0:
                nm = null_b.mean()
                p_vals[b] = np.mean(np.abs(null_b - nm) >= np.abs(obs[b] - nm))

    return {
        "bins_all":  bins_all,
        "denom":     denom,
        "obs":       obs,
        "perm_mean": perm_mean,
        "perm_lo":   perm_lo,
        "perm_hi":   perm_hi,
        "p_vals":    p_vals,
        "n_comps":   len(comp_groups),
    }


def run_regression(df_s):
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        reg = df_s.dropna(subset=["local_hour", "year", "is_final"]).copy()
        reg = reg[reg["n_attempts"] > 0]
        reg = reg.rename(columns={RECORD_COL: "n_records_avg"})
        h = reg["local_hour"]
        reg["sin1"]         = np.sin(2 * np.pi * h / 24)
        reg["cos1"]         = np.cos(2 * np.pi * h / 24)
        reg["sin2"]         = np.sin(4 * np.pi * h / 24)
        reg["cos2"]         = np.cos(4 * np.pi * h / 24)
        reg["is_final_int"] = reg["is_final"].astype(int)
        reg["year_c"]       = reg["year"] - reg["year"].mean()

        model = smf.glm(
            "n_records_avg ~ sin1 + cos1 + sin2 + cos2 + is_final_int + year_c",
            data=reg,
            family=sm.families.Poisson(),
            offset=np.log(reg["n_attempts"].clip(lower=1)),
        ).fit(disp=False)
        return model
    except Exception as e:
        print(f"    Regression failed: {e}")
        return None


def make_xticks(step=3):
    hrs = np.arange(0, 25, step)
    return hrs, [f"{int(h):02d}:00" for h in hrs]


def hour_label(h):
    hh = int(h)
    mm = int(round((h - hh) * 60))
    return f"{hh:02d}:{mm:02d}"


# ============================================================
# MAIN LOOP
# ============================================================
results_store = {}

for event in EVENTS:
    print(f"\n{'='*55}")
    print(f"  EVENT: {event}  ({EVENT_LABELS[event]})")
    print(f"{'='*55}")

    fname = f"new_round_table_{event}.csv"
    try:
        df = pd.read_csv(fname)
    except FileNotFoundError:
        print(f"  File not found: {fname} -- skipping")
        continue

    df_s = df.dropna(subset=["local_hour", "n_attempts"]).copy()
    df_s = df_s[df_s["n_attempts"] > 0]

    total_rec = df_s[RECORD_COL].sum()
    total_att = df_s["n_attempts"].sum()
    n_wr = df_s[WR_COL].sum()
    n_cr = df_s[CR_COL].sum()
    n_nr = df_s[NR_COL].sum()

    print(f"  Scheduled rounds: {len(df_s):,}")
    print(f"  Avg records: {total_rec:,}  (WR={n_wr}, CR={n_cr}, NR={n_nr})")
    print(f"  Attempts: {total_att:,}")

    if total_rec == 0 or total_att == 0:
        print("  No avg record data -- skipping")
        continue

    scale   = choose_scale(total_rec, total_att)
    min_att = choose_min_attempts(total_att, 83)
    print(f"  Scale: per {scale:,} attempts | Min-attempts filter: {min_att:,}")

    bins_all  = bin_aggregate(df_s, scale=scale)
    bins_plot = bins_all[bins_all["n_attempts"] >= min_att]
    print(f"  Bins with data: {len(bins_all)} | Passing filter: {len(bins_plot)}")

    # --- Permutation test ---
    print("  Running permutation test...")
    perm = run_permutation(df_s, scale, min_att)
    print(f"    Competitions used: {perm['n_comps']}")

    # --- Regression ---
    print("  Running regression...")
    model = run_regression(df_s)
    if model:
        print(f"    sin1 p={model.pvalues['sin1']:.4f}, cos1 p={model.pvalues['cos1']:.4f}")
        reg_table = pd.DataFrame({
            "predictor": model.params.index,
            "coef":      model.params.values,
            "se":        model.bse.values,
            "z":         model.tvalues.values,
            "p_value":   model.pvalues.values,
            "IRR":       np.exp(model.params.values),
            "IRR_lo95":  np.exp(model.conf_int().iloc[:, 0].values),
            "IRR_hi95":  np.exp(model.conf_int().iloc[:, 1].values),
        })
        reg_table.to_csv(f"avg_regression_{event}.csv", index=False)

    # --- Summary stats ---
    if len(bins_plot) > 0:
        peak_row   = bins_plot.loc[bins_plot["rate_scaled"].idxmax()]
        trough_row = bins_plot.loc[bins_plot["rate_scaled"].idxmin()]
        overall_rate = total_rec / total_att * scale
        print(f"  Overall rate: {overall_rate:.2f} per {scale:,}")
        print(f"  Peak:   {hour_label(peak_row['bin_15min'])} -> {peak_row['rate_scaled']:.2f}")
        print(f"  Trough: {hour_label(trough_row['bin_15min'])} -> {trough_row['rate_scaled']:.2f}")

    results_store[event] = {
        "df_s":      df_s,
        "bins_all":  bins_all,
        "bins_plot": bins_plot,
        "perm":      perm,
        "model":     model,
        "scale":     scale,
        "min_att":   min_att,
        "n_wr":      n_wr,
        "n_cr":      n_cr,
        "n_nr":      n_nr,
        "total_rec": total_rec,
        "total_att": total_att,
    }

    # Save permutation CSV
    perm_out = pd.DataFrame({
        "bin_15min":  [b * 0.25 for b in range(96)],
        "obs_rate":   perm["obs"],
        "perm_mean":  perm["perm_mean"],
        "perm_lo":    perm["perm_lo"],
        "perm_hi":    perm["perm_hi"],
        "p_2sided":   perm["p_vals"],
        "n_attempts": perm["denom"],
    })
    perm_out.to_csv(f"avg_permutation_{event}.csv", index=False)

    # ----------------------------------------------------------------
    # Per-event figure
    # ----------------------------------------------------------------
    color = EVENT_COLORS[event]
    label = EVENT_LABELS[event]

    fig, ax = plt.subplots(figsize=(11, 5))

    mask   = perm["denom"] >= min_att
    null_x = np.array([b * 0.25 for b in range(96)])
    ax.fill_between(null_x[mask], perm["perm_lo"][mask], perm["perm_hi"][mask],
                    color=NULL_COLOR, alpha=0.8,
                    label="Permutation null 95% band")
    ax.plot(null_x[mask], perm["perm_mean"][mask],
            color=NULL_EDGE, lw=1.2, ls="--", label="Null mean")

    if len(bins_plot) > 0:
        x = bins_plot["bin_15min"].values
        ax.fill_between(x, bins_plot["ci_lo"], bins_plot["ci_hi"],
                        color=color, alpha=0.25)
        ax.plot(x, bins_plot["rate_scaled"], color=color,
                lw=2, marker="o", ms=4, label="Observed rate (95% CI)")

    ax.axvspan(12.5, 15, color="#ffffb2", alpha=0.4, zorder=0)
    ax.set_xlabel("Local time of day", fontsize=12)
    ax.set_ylabel(f"Avg records per {scale:,} attempts", fontsize=12)
    ax.set_title(f"{label}: time-of-day average record rate", fontsize=12)

    xt, xl = make_xticks(2)
    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=45)
    ax.set_xlim(5, 23)

    note = (f"n = {total_rec:,} avg records (WR={n_wr}, CR={n_cr}, NR={n_nr})\n"
            f"{total_att:,} attempts | bins >= {min_att:,} attempts shown")
    ax.text(0.01, 0.98, note, transform=ax.transAxes,
            va="top", fontsize=8, color="#444")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"avg_fig_{event}_rate.png", dpi=150)
    plt.close()
    print(f"  Saved: avg_fig_{event}_rate.png")


# ============================================================
# COMBINED COGNITIVE GRADIENT FIGURE
# ============================================================
print("\n" + "="*55)
print("  COMBINED COGNITIVE GRADIENT FIGURE (Avg records)")
print("="*55)

grad_events = [e for e in EVENTS if e in results_store]

if len(grad_events) >= 2:
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    event_positions = {
        "333": (0, 0), "444": (0, 1), "555": (1, 0), "666": (1, 1)
    }

    for event in grad_events:
        r, c = event_positions[event]
        ax = fig.add_subplot(gs[r, c])
        store    = results_store[event]
        color    = EVENT_COLORS[event]
        scale    = store["scale"]
        min_att  = store["min_att"]
        perm     = store["perm"]
        bins_plot = store["bins_plot"]

        mask   = perm["denom"] >= min_att
        null_x = np.array([b * 0.25 for b in range(96)])
        ax.fill_between(null_x[mask], perm["perm_lo"][mask], perm["perm_hi"][mask],
                        color=NULL_COLOR, alpha=0.8)
        ax.plot(null_x[mask], perm["perm_mean"][mask],
                color=NULL_EDGE, lw=1, ls="--")

        if len(bins_plot) > 0:
            x = bins_plot["bin_15min"].values
            ax.fill_between(x, bins_plot["ci_lo"], bins_plot["ci_hi"],
                            color=color, alpha=0.3)
            ax.plot(x, bins_plot["rate_scaled"], color=color,
                    lw=2, marker="o", ms=3.5)

        ax.axvspan(12.5, 15, color="#ffffb2", alpha=0.4, zorder=0)

        model = store["model"]
        if model is not None:
            p_sin = model.pvalues.get("sin1", np.nan)
            p_cos = model.pvalues.get("cos1", np.nan)
            if not np.isnan(p_sin):
                sig = "**" if min(p_sin, p_cos) < 0.01 else ("*" if min(p_sin, p_cos) < 0.05 else "ns")
                ax.text(0.98, 0.97, f"Time-of-day: {sig}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8.5, color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))

        n_wr = store["n_wr"]; n_cr = store["n_cr"]; n_nr = store["n_nr"]
        ax.text(0.01, 0.98,
                f"n={store['total_rec']:,} (WR={n_wr}, CR={n_cr}, NR={n_nr})",
                transform=ax.transAxes, va="top", fontsize=7.5, color="#444")

        ax.set_title(EVENT_LABELS[event], fontsize=12, color=color, fontweight="bold")
        ax.set_xlabel("Local time of day", fontsize=9)
        ax.set_ylabel(f"Avg records per {scale:,} attempts", fontsize=9)

        xt, xl = make_xticks(3)
        ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=45, fontsize=8)
        ax.set_xlim(5, 23)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color="black", lw=2, marker="o", ms=4,
               label="Observed avg rate (95% Poisson CI)"),
        Patch(facecolor=NULL_COLOR, edgecolor=NULL_EDGE, ls="--",
              label="Within-competition permutation null (95%)"),
        Patch(facecolor="#ffffb2", alpha=0.6, label="Post-lunch window (12:30-15:00)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Circadian time-of-day effects on WCA average record-breaking (Ao5/Mo3): cognitive complexity gradient\n"
        "333/444/555 = Ao5 | 666 = Mo3 | Shaded yellow = post-lunch dip window",
        fontsize=11, y=1.02,
    )

    plt.savefig("avg_fig_cognitive_gradient.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: avg_fig_cognitive_gradient.png")

print("\nAll done.")
