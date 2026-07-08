"""
clean4_tables_multi.py
======================
Build round-level data tables for 333, 444, 555, 666, 333fm.

For each (competitionId, eventId, roundTypeId):
  - n_competitors, n_attempts (valid solves counted from value1-5)
  - n_records, n_wr, n_cr, n_nr  (single + average combined)
  - local_hour, bin_15min, year, is_final, round_stage

Output: new_round_table_{event}.csv  for each event
"""

import pandas as pd
import numpy as np

EVENTS = ["333", "444", "555", "666", "333fm"]
CONTINENTAL = {"AfR", "AsR", "ER", "NAR", "OcR", "SAR"}
VAL_COLS = ["value1", "value2", "value3", "value4", "value5"]

# ── Load support tables ───────────────────────────────────────────────────────
print("Loading support tables...")
sched = pd.read_csv("new_schedule_multi.csv",
                    usecols=["competitionId", "eventId", "roundTypeId",
                             "local_hour", "bin_15min", "timezone"])
# One row per (comp, event, roundTypeId) - already deduplicated
sched = sched.drop_duplicates(subset=["competitionId", "eventId", "roundTypeId"])

comps = pd.read_csv("WCA_export_Competitions.tsv", sep="\t",
                    usecols=["id", "year", "countryId"]).rename(columns={"id": "competitionId"})

rt = pd.read_csv("WCA_export_RoundTypes.tsv", sep="\t")
rt = rt.rename(columns={"id": "roundTypeId"})
rt["is_final"]    = rt["final"] == 1
rt["round_stage"] = rt["is_final"].map({True: "Final", False: "Early"})

print(f"  Schedule: {len(sched):,} rows covering {sched['eventId'].unique().tolist()}")

# ── Stream Results TSV once, keeping all target events ───────────────────────
print("\nStreaming Results TSV...")
USECOLS = (["competitionId", "eventId", "roundTypeId", "personId",
            "regionalSingleRecord", "regionalAverageRecord"] + VAL_COLS)

all_chunks = {e: [] for e in EVENTS}

for chunk in pd.read_csv(
    "WCA_export_Results.tsv", sep="\t", chunksize=200_000, usecols=USECOLS,
    dtype={v: "Int32" for v in VAL_COLS},
):
    for event in EVENTS:
        sub = chunk[chunk["eventId"] == event]
        if len(sub) > 0:
            all_chunks[event].append(sub)

results_by_event = {}
for event in EVENTS:
    if all_chunks[event]:
        df = pd.concat(all_chunks[event], ignore_index=True)
        results_by_event[event] = df
        print(f"  {event}: {len(df):,} rows")
    else:
        results_by_event[event] = pd.DataFrame(columns=USECOLS)
        print(f"  {event}: 0 rows")

# ── Process each event ────────────────────────────────────────────────────────
def process_event(event_id, results):
    """Build round-level table for one event."""
    if len(results) == 0:
        return pd.DataFrame()

    # Count valid attempts per row (not DNS=-2, not unused=0)
    for c in VAL_COLS:
        results[c] = pd.to_numeric(results[c], errors="coerce").fillna(0).astype(int)
    attempt_mask = results[VAL_COLS].apply(lambda col: (col != 0) & (col != -2))
    results["row_attempts"] = attempt_mask.sum(axis=1)

    # Record flags
    for col, lbl in [("regionalSingleRecord", "s"), ("regionalAverageRecord", "a")]:
        v = results[col].fillna("")
        results[f"rec_{lbl}"]  = (v != "").astype(int)
        results[f"wr_{lbl}"]   = (v == "WR").astype(int)
        results[f"cr_{lbl}"]   = v.isin(CONTINENTAL).astype(int)
        results[f"nr_{lbl}"]   = (v == "NR").astype(int)

    # Aggregate to round level
    grp = results.groupby(["competitionId", "roundTypeId"]).agg(
        n_competitors = ("personId",      "count"),
        n_attempts    = ("row_attempts",  "sum"),
        n_single_rec  = ("rec_s",         "sum"),
        n_avg_rec     = ("rec_a",         "sum"),
        n_wr_single   = ("wr_s",          "sum"),
        n_wr_avg      = ("wr_a",          "sum"),
        n_cr_single   = ("cr_s",          "sum"),
        n_cr_avg      = ("cr_a",          "sum"),
        n_nr_single   = ("nr_s",          "sum"),
        n_nr_avg      = ("nr_a",          "sum"),
    ).reset_index()

    grp["eventId"]   = event_id
    grp["n_records"] = grp["n_single_rec"] + grp["n_avg_rec"]
    grp["n_wr"]      = grp["n_wr_single"]  + grp["n_wr_avg"]
    grp["n_cr"]      = grp["n_cr_single"]  + grp["n_cr_avg"]
    grp["n_nr"]      = grp["n_nr_single"]  + grp["n_nr_avg"]

    # Join schedule
    ev_sched = sched[sched["eventId"] == event_id][
        ["competitionId", "roundTypeId", "local_hour", "bin_15min", "timezone"]
    ]
    grp = grp.merge(ev_sched,  on=["competitionId", "roundTypeId"], how="left")
    grp = grp.merge(comps,     on="competitionId",                  how="left")
    grp = grp.merge(rt[["roundTypeId", "is_final", "round_stage", "cellName"]],
                    on="roundTypeId", how="left")
    grp = grp.rename(columns={"cellName": "round_name"})

    grp["era"] = pd.cut(grp["year"],
                        bins=[2010, 2015, 2018, 2022, 2027],
                        labels=["2011-2015", "2016-2018", "2019-2022", "2023-2026"])
    return grp


for event in EVENTS:
    print(f"\nProcessing {event}...")
    df = process_event(event, results_by_event[event].copy())
    if len(df) == 0:
        print(f"  No data for {event}")
        continue

    sched_rows = df.dropna(subset=["local_hour"])
    total_rec  = df["n_records"].sum()
    sched_rec  = sched_rows["n_records"].sum()
    total_att  = df["n_attempts"].sum()
    sched_att  = sched_rows["n_attempts"].sum()

    print(f"  Rounds total: {len(df):,}  |  With schedule: {len(sched_rows):,} ({100*len(sched_rows)/len(df):.0f}%)")
    print(f"  Records: {sched_rec:,} / {total_rec:,} have local time")
    print(f"  Attempts (scheduled): {sched_att:,}")
    print(f"  WR={df['n_wr'].sum()}, CR={df['n_cr'].sum()}, NR={df['n_nr'].sum()}")

    fname = f"new_round_table_{event.replace('333fm','333fm')}.csv"
    df.to_csv(fname, index=False)
    print(f"  Saved: {fname}")

print("\nDone.")
