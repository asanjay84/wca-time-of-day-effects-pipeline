"""
clean6_dip_spike_quantification.py
====================================
Quantify post-prandial dip and late-evening spike for 333, 444, 555, 666.
Uses AVERAGE (Ao5/Mo3) records from new_round_table_{event}.csv.

For each event computes:
  1. Dip: mean rate inside 12:30-15:00 vs outside; ratio (out/in), absolute
     difference, one-tailed permutation p-value (is outside > inside?)
  2. Morning peak: peak bin before 12:00 (time + rate)
  3. Evening spike: peak bin after 17:00 (time + rate) + % from finals vs early
  4. Overall non-uniformity: permutation p-value via max-deviation statistic

Outputs:
  data/processed/dip_quantification_summary.csv
  Printed summary table
"""

import os
import pandas as pd
import numpy as np
import scipy.stats as stats

os.makedirs("data/processed", exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
EVENTS     = ["333", "444", "555", "666"]
N_PERMS    = 10_000
RNG_SEED   = 42

RECORD_COL = "n_avg_rec"
WR_COL     = "n_wr_avg"
CR_COL     = "n_cr_avg"
NR_COL     = "n_nr_avg"

EVENT_LABELS = {
    "333": "3x3x3 (Ao5)",
    "444": "4x4x4 (Ao5)",
    "555": "5x5x5 (Ao5)",
    "666": "6x6x6 (Mo3)",
}

DIP_START = 12.5   # 12:30 inclusive
DIP_END   = 15.0   # 15:00 exclusive (bins 12:30, 12:45, ..., 14:45)
EVE_START = 17.0   # evening spike: bins >= 17:00
MORN_END  = 12.0   # morning peak: bins < 12:00

# ── HELPERS ───────────────────────────────────────────────────────────────────
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


def choose_min_attempts(total_attempts):
    return min(max(100, int(total_attempts * 0.005)), 10_000)


def poisson_ci(k, n, scale, alpha=0.05):
    if n == 0:
        return 0.0, 0.0
    lo = stats.chi2.ppf(alpha / 2,  2 * k)          / (2 * n) if k > 0 else 0.0
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / (2 * n)
    return lo * scale, hi * scale


def bin_agg(data, record_col, scale):
    grp = data.groupby("bin_15min").agg(
        n_rec = (record_col, "sum"),
        n_att = ("n_attempts", "sum"),
    ).reset_index()
    grp = grp[grp["n_att"] > 0].copy()
    grp["rate"] = grp["n_rec"] / grp["n_att"] * scale
    cis = [poisson_ci(int(r["n_rec"]), r["n_att"], scale) for _, r in grp.iterrows()]
    grp["ci_lo"] = [c[0] for c in cis]
    grp["ci_hi"] = [c[1] for c in cis]
    return grp


def hour_label(h):
    hh = int(h); mm = int(round((h - hh) * 60))
    return f"{hh:02d}:{mm:02d}"


def run_permutation(df_s, record_col, scale, min_att, n_perms, seed):
    """
    One permutation loop. Returns per-permutation arrays needed for:
      - dip ratio test (one-tailed)
      - overall non-uniformity (max-deviation statistic)
    Also returns observed bin-level stats.
    """
    rng = np.random.default_rng(seed)

    bins_all   = bin_agg(df_s, record_col, scale)
    att_by_bin = bins_all.set_index("bin_15min")["n_att"].to_dict()

    comp_rounds = df_s[["competitionId", "bin_15min", "n_attempts", record_col]].copy()
    comp_rounds = comp_rounds.rename(columns={record_col: "n_rec"})
    comp_rounds["bin_idx"] = (comp_rounds["bin_15min"] / 0.25).round().astype(int)

    comp_groups = []
    for _, grp in comp_rounds.groupby("competitionId"):
        K = int(grp["n_rec"].sum())
        if K == 0:
            continue
        att = grp["n_attempts"].values.astype(float)
        N   = att.sum()
        if N == 0:
            continue
        comp_groups.append((K, grp["bin_idx"].values, att / N))

    denom      = np.array([att_by_bin.get(b * 0.25, 0) for b in range(96)], dtype=float)
    denom_safe = np.where(denom > 0, denom, np.nan)

    # Observed rates per bin (96-length array, nan where no data)
    obs_rates = np.full(96, np.nan)
    for _, row in bins_all.iterrows():
        b = round(row["bin_15min"] / 0.25)
        if 0 <= b < 96:
            obs_rates[b] = row["rate"]

    # Masks
    valid_mask = denom >= min_att
    dip_mask   = np.array([(DIP_START <= b * 0.25 < DIP_END) for b in range(96)])
    # out_mask uses all bins with any data (denom > 0), matching the observed
    # out-of-window denominator exactly. Previously used valid_mask here, which
    # caused the null and observed ratios to be computed over different bin sets.
    out_mask   = (denom > 0) & ~dip_mask

    # Observed dip statistics
    dip_valid  = valid_mask & dip_mask
    dip_rec    = sum(
        bins_all.loc[bins_all["bin_15min"].between(DIP_START, DIP_END - 0.01), "n_rec"]
    )
    dip_att    = sum(
        bins_all.loc[bins_all["bin_15min"].between(DIP_START, DIP_END - 0.01), "n_att"]
    )
    out_rec = bins_all.loc[
        (bins_all["bin_15min"] < DIP_START) | (bins_all["bin_15min"] >= DIP_END),
        "n_rec"
    ].sum()
    out_att = bins_all.loc[
        (bins_all["bin_15min"] < DIP_START) | (bins_all["bin_15min"] >= DIP_END),
        "n_att"
    ].sum()

    obs_dip_rate = (dip_rec / dip_att * scale) if dip_att > 0 else np.nan
    obs_out_rate = (out_rec / out_att * scale) if out_att > 0 else np.nan
    obs_ratio    = obs_out_rate / obs_dip_rate if obs_dip_rate > 0 else np.nan

    # Observed max-deviation stat (for non-uniformity test)
    # stat = max(|rate - mean_rate|) / mean_rate across bins with sufficient exposure
    valid_rates = obs_rates[valid_mask]
    valid_rates = valid_rates[~np.isnan(valid_rates)]
    obs_mean    = valid_rates.mean() if len(valid_rates) > 0 else np.nan
    obs_maxdev  = (np.max(np.abs(valid_rates - obs_mean)) / obs_mean
                   if obs_mean > 0 else np.nan)

    # Permutation loop
    null_ratios  = np.zeros(n_perms)
    null_maxdevs = np.zeros(n_perms)

    for i in range(n_perms):
        null_rec = np.zeros(96)
        for K, bins_arr, weights in comp_groups:
            counts = rng.multinomial(K, weights)
            np.add.at(null_rec, bins_arr, counts)

        null_rates_arr = null_rec / denom_safe * scale  # 96-length, nan where no data

        # dip ratio
        null_dip = np.nansum(null_rec[dip_mask]) / np.nansum(denom[dip_mask]) * scale if np.nansum(denom[dip_mask]) > 0 else np.nan
        null_out = np.nansum(null_rec[out_mask]) / np.nansum(denom[out_mask]) * scale if np.nansum(denom[out_mask]) > 0 else np.nan
        null_ratios[i] = null_out / null_dip if (null_dip and null_dip > 0) else np.nan

        # max-deviation
        vr = null_rates_arr[valid_mask]
        vr = vr[~np.isnan(vr)]
        if len(vr) > 0:
            vm = vr.mean()
            null_maxdevs[i] = np.max(np.abs(vr - vm)) / vm if vm > 0 else np.nan

    # One-tailed p: fraction of null ratios >= observed ratio
    valid_null_ratios = null_ratios[~np.isnan(null_ratios)]
    p_dip_onetail = (np.sum(valid_null_ratios >= obs_ratio) / len(valid_null_ratios)
                     if (len(valid_null_ratios) > 0 and not np.isnan(obs_ratio)) else np.nan)

    valid_null_maxdevs = null_maxdevs[~np.isnan(null_maxdevs)]
    p_nonuniform = (np.sum(valid_null_maxdevs >= obs_maxdev) / len(valid_null_maxdevs)
                    if (len(valid_null_maxdevs) > 0 and not np.isnan(obs_maxdev)) else np.nan)

    return {
        "bins_all":      bins_all,
        "denom":         denom,
        "obs_rates":     obs_rates,
        "obs_dip_rate":  obs_dip_rate,
        "obs_out_rate":  obs_out_rate,
        "obs_ratio":     obs_ratio,
        "obs_maxdev":    obs_maxdev,
        "p_dip_onetail": p_dip_onetail,
        "p_nonuniform":  p_nonuniform,
        "dip_rec":       dip_rec,
        "dip_att":       dip_att,
        "out_rec":       out_rec,
        "out_att":       out_att,
        "scale":         scale,
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────
rows = []

for event in EVENTS:
    print(f"\n{'='*60}")
    print(f"  {event}  {EVENT_LABELS[event]}")
    print(f"{'='*60}")

    fname = f"new_round_table_{event}.csv"
    try:
        df = pd.read_csv(fname)
    except FileNotFoundError:
        print(f"  File not found: {fname} -- skipping")
        continue

    df_s = df.dropna(subset=["local_hour", "n_attempts"]).copy()
    df_s = df_s[df_s["n_attempts"] > 0]
    df_s["bin_15min"] = (df_s["local_hour"] // 0.25) * 0.25

    total_rec = df_s[RECORD_COL].sum()
    total_att = df_s["n_attempts"].sum()
    scale     = choose_scale(total_rec, total_att)
    min_att   = choose_min_attempts(total_att)

    print(f"  Total avg records: {total_rec:,} | Attempts: {total_att:,}")
    print(f"  Scale: per {scale:,} | Min-attempts filter: {min_att:,}")
    print(f"  Running permutation ({N_PERMS:,} reps)...")

    res = run_permutation(df_s, RECORD_COL, scale, min_att, N_PERMS, RNG_SEED)
    bins_all = res["bins_all"]

    # ── 1. DIP ────────────────────────────────────────────────────────────────
    dip_rate = res["obs_dip_rate"]
    out_rate = res["obs_out_rate"]
    ratio    = res["obs_ratio"]
    abs_diff = out_rate - dip_rate if (dip_rate and out_rate) else np.nan
    p_dip    = res["p_dip_onetail"]

    print(f"\n  [DIP 12:30-15:00]")
    print(f"    Inside dip:  {dip_rate:.2f} per {scale:,} attempts  ({int(res['dip_rec'])} records, {int(res['dip_att']):,} attempts)")
    print(f"    Outside dip: {out_rate:.2f} per {scale:,} attempts  ({int(res['out_rec'])} records, {int(res['out_att']):,} attempts)")
    print(f"    Ratio (out/in): {ratio:.2f}x")
    print(f"    Abs diff:    {abs_diff:.2f} per {scale:,} attempts")
    print(f"    One-tailed perm p (outside > inside): {p_dip:.4f}")

    # ── 2. MORNING PEAK ───────────────────────────────────────────────────────
    bins_morn = bins_all[(bins_all["bin_15min"] < MORN_END) &
                         (bins_all["n_att"] >= min_att)]
    if len(bins_morn) > 0:
        peak_morn_row = bins_morn.loc[bins_morn["rate"].idxmax()]
        morn_peak_time = hour_label(peak_morn_row["bin_15min"])
        morn_peak_rate = peak_morn_row["rate"]
        morn_peak_n    = int(peak_morn_row["n_rec"])
        morn_peak_att  = int(peak_morn_row["n_att"])
    else:
        morn_peak_time = "N/A"
        morn_peak_rate = np.nan
        morn_peak_n    = 0
        morn_peak_att  = 0

    print(f"\n  [MORNING PEAK <12:00]")
    print(f"    Peak bin: {morn_peak_time} -> {morn_peak_rate:.2f} per {scale:,}  ({morn_peak_n} records, {morn_peak_att:,} attempts)")

    # ── 3. EVENING SPIKE ──────────────────────────────────────────────────────
    bins_eve = bins_all[(bins_all["bin_15min"] >= EVE_START) &
                        (bins_all["n_att"] >= min_att)]
    if len(bins_eve) > 0:
        peak_eve_row   = bins_eve.loc[bins_eve["rate"].idxmax()]
        eve_peak_time  = hour_label(peak_eve_row["bin_15min"])
        eve_peak_rate  = peak_eve_row["rate"]
        eve_peak_bin   = peak_eve_row["bin_15min"]

        # % from finals vs early in that peak bin
        peak_bin_data  = df_s[df_s["bin_15min"] == eve_peak_bin]
        finals_rec     = peak_bin_data.loc[peak_bin_data["round_stage"] == "Final", RECORD_COL].sum()
        early_rec      = peak_bin_data.loc[peak_bin_data["round_stage"] == "Early", RECORD_COL].sum()
        total_bin_rec  = finals_rec + early_rec
        pct_finals     = (finals_rec / total_bin_rec * 100) if total_bin_rec > 0 else np.nan
        pct_early      = (early_rec  / total_bin_rec * 100) if total_bin_rec > 0 else np.nan
    else:
        eve_peak_time = "N/A"
        eve_peak_rate = np.nan
        eve_peak_bin  = np.nan
        finals_rec    = 0
        early_rec     = 0
        total_bin_rec = 0
        pct_finals    = np.nan
        pct_early     = np.nan

    print(f"\n  [EVENING SPIKE >=17:00]")
    print(f"    Peak bin: {eve_peak_time} -> {eve_peak_rate:.2f} per {scale:,}  ({total_bin_rec} records)")
    print(f"    Finals records in peak bin:  {int(finals_rec)} ({pct_finals:.1f}%)")
    print(f"    Early records in peak bin:   {int(early_rec)} ({pct_early:.1f}%)")

    # ── 4. OVERALL NON-UNIFORMITY ─────────────────────────────────────────────
    p_nonunif = res["p_nonuniform"]
    print(f"\n  [OVERALL NON-UNIFORMITY]")
    print(f"    Max-deviation stat (obs/mean): {res['obs_maxdev']:.3f}")
    print(f"    Permutation p (two-tailed):    {p_nonunif:.4f}")

    rows.append({
        "event":                   event,
        "event_label":             EVENT_LABELS[event],
        "scale":                   scale,
        "total_avg_records":       int(total_rec),
        "total_attempts":          int(total_att),
        # Dip
        "dip_window":              "12:30-15:00",
        "dip_rate_inside":         round(dip_rate, 4) if not np.isnan(dip_rate) else None,
        "dip_rate_outside":        round(out_rate, 4) if not np.isnan(out_rate) else None,
        "dip_ratio_out_over_in":   round(ratio, 3) if not np.isnan(ratio) else None,
        "dip_abs_diff":            round(abs_diff, 4) if not np.isnan(abs_diff) else None,
        "dip_records_inside":      int(res["dip_rec"]),
        "dip_attempts_inside":     int(res["dip_att"]),
        "dip_p_onetail":           round(p_dip, 4) if not np.isnan(p_dip) else None,
        "rate_units":              f"records per {scale:,} attempts",
        # Morning
        "morning_peak_time":       morn_peak_time,
        "morning_peak_rate":       round(morn_peak_rate, 4) if not np.isnan(morn_peak_rate) else None,
        "morning_peak_n_records":  morn_peak_n,
        # Evening
        "evening_peak_time":       eve_peak_time,
        "evening_peak_rate":       round(eve_peak_rate, 4) if not np.isnan(eve_peak_rate) else None,
        "evening_peak_n_records":  int(total_bin_rec),
        "evening_peak_pct_finals": round(pct_finals, 1) if not np.isnan(pct_finals) else None,
        "evening_peak_pct_early":  round(pct_early, 1) if not np.isnan(pct_early) else None,
        # Non-uniformity
        "nonuniformity_maxdev_stat": round(res["obs_maxdev"], 4) if not np.isnan(res["obs_maxdev"]) else None,
        "nonuniformity_p":           round(p_nonunif, 4) if not np.isnan(p_nonunif) else None,
    })


# ── SAVE + PRINT SUMMARY TABLE ───────────────────────────────────────────────
summary = pd.DataFrame(rows)
summary.to_csv("data/processed/dip_quantification_summary.csv", index=False)
print(f"\n\nSaved: data/processed/dip_quantification_summary.csv")

# Pretty summary table
print("\n" + "="*80)
print("SUMMARY TABLE — Post-prandial dip and evening spike (Ao5/Mo3 avg records)")
print("="*80)

header = f"{'Event':<14} {'Scale':<16} {'Dip in':>8} {'Dip out':>8} {'Ratio':>6} {'p(dip)':>8} {'Morn peak':>10} {'Eve peak':>10} {'%Finals':>8} {'p(unif)':>8}"
print(header)
print("-"*80)

for r in rows:
    scale_str = r["rate_units"].replace("records per ", "per ")
    print(
        f"{r['event_label']:<14} "
        f"{scale_str:<16} "
        f"{r['dip_rate_inside']:>8.2f} "
        f"{r['dip_rate_outside']:>8.2f} "
        f"{r['dip_ratio_out_over_in']:>6.2f}x "
        f"{r['dip_p_onetail']:>8.4f} "
        f"{r['morning_peak_time']:>10} "
        f"{r['evening_peak_time']:>10} "
        f"{r['evening_peak_pct_finals']:>7.1f}% "
        f"{r['nonuniformity_p']:>8.4f}"
    )

print("-"*80)
print("Columns: Dip in/out = record rate inside/outside 12:30-15:00 window")
print("         Ratio = outside/inside  |  p(dip) = one-tailed permutation p")
print("         Morn peak = peak bin before 12:00  |  Eve peak = peak bin after 17:00")
print("         %Finals = % of records in evening peak bin from final rounds")
print("         p(unif) = overall non-uniformity permutation p (max-deviation stat)")
