"""
Parse WCIF cache to extract round start times (local) for:
  333, 444, 555, 666, 333fm

For each WCIF file:
  - Extract activityCode / startTime (UTC) / timezone (from venue)
  - Convert to local hour

Then build (competitionId, eventId, round_number) -> local_hour table.
Also build (competitionId, eventId, roundTypeId) -> round_number mapping
from Results TSV (rank-order of round types actually used per comp/event).

Output: new_schedule_multi.csv
  competitionId, eventId, round_number, roundTypeId, local_hour, timezone
"""

import json
import os
import re
import pandas as pd
import numpy as np
from dateutil import parser as dtparser
import pytz

CACHE_DIR = "wcif_cache_public"
EVENTS    = ["333", "444", "555", "666", "333fm"]
# FMC uses codes like 333fm-r1-a1 (attempt sub-activities), so we capture round number and ignore attempt suffix
ACT_RE    = re.compile(r"^(" + "|".join(re.escape(e) for e in EVENTS) + r")-r(\d+)(?:-a\d+)?$")

# Parse WCIF cache
print("Parsing WCIF cache...")
files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
print(f"  {len(files):,} files found")

rows = []
errors = 0
for i, fname in enumerate(files):
    if i % 2000 == 0:
        print(f"  {i}/{len(files)}...")
    comp_id = fname.replace(".json", "")
    path = os.path.join(CACHE_DIR, fname)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        sched = data.get("schedule", {})
        for venue in sched.get("venues", []):
            tz_name = venue.get("timezone", "")
            try:
                tz = pytz.timezone(tz_name)
            except Exception:
                tz = None
            for room in venue.get("rooms", []):
                for act in room.get("activities", []):
                    code  = act.get("activityCode", "")
                    start = act.get("startTime", "")
                    m = ACT_RE.match(code)
                    if m and start:
                        event_id     = m.group(1)
                        round_number = int(m.group(2))
                        try:
                            dt_utc = dtparser.parse(start).replace(tzinfo=pytz.utc)
                            if tz is not None:
                                dt_local  = dt_utc.astimezone(tz)
                                local_hour = dt_local.hour + dt_local.minute / 60
                            else:
                                local_hour = np.nan
                        except Exception:
                            local_hour = np.nan
                        rows.append({
                            "competitionId": comp_id,
                            "eventId":       event_id,
                            "round_number":  round_number,
                            "local_hour":    local_hour,
                            "timezone":      tz_name,
                        })
    except Exception:
        errors += 1

print(f"  Done. {len(rows):,} activity rows parsed, {errors} file errors")

sched_df = pd.DataFrame(rows)
print(f"\nEvent coverage:")
print(sched_df.groupby("eventId")[["competitionId"]].nunique().rename(
    columns={"competitionId": "n_competitions"}))

# Build round_type_id -> round_number mapping from Results TSV
print("\nBuilding round_type -> round_number mapping from Results TSV...")

USECOLS_R = ["competitionId", "eventId", "roundTypeId"]
rt_map_chunks = []
for chunk in pd.read_csv(
    "WCA_export_Results.tsv", sep="\t", chunksize=200_000, usecols=USECOLS_R
):
    rt_map_chunks.append(chunk[chunk["eventId"].isin(EVENTS)])

results_slim = pd.concat(rt_map_chunks, ignore_index=True).drop_duplicates()
print(f"  {len(results_slim):,} unique (comp, event, roundType) combos")

# Load round type ranks
rt_ranks = pd.read_csv("WCA_export_RoundTypes.tsv", sep="\t")[["id", "rank"]].rename(
    columns={"id": "roundTypeId"}
)
results_slim = results_slim.merge(rt_ranks, on="roundTypeId", how="left")

# Within each (comp, event): sort by rank → assign round_number 1, 2, 3...
results_slim = results_slim.sort_values(["competitionId", "eventId", "rank"])
results_slim["round_number"] = results_slim.groupby(
    ["competitionId", "eventId"]
).cumcount() + 1

rt_map = results_slim[["competitionId", "eventId", "roundTypeId", "round_number"]]
print(f"  Mapping rows: {len(rt_map):,}")

# Deduplicate schedule (keep earliest per comp/event/round_number)
sched_dedup = (sched_df
    .dropna(subset=["local_hour"])
    .sort_values("local_hour")
    .drop_duplicates(subset=["competitionId", "eventId", "round_number"])
)

# Join: schedule + round_type mapping
combined = sched_dedup.merge(rt_map, on=["competitionId", "eventId", "round_number"],
                             how="inner")

print(f"\nJoined schedule rows: {len(combined):,}")
print("Per event:")
print(combined.groupby("eventId").agg(
    n_comps  = ("competitionId", "nunique"),
    n_rounds = ("round_number", "count")
))

# 15-min bin
combined["bin_15min"] = (combined["local_hour"] // 0.25) * 0.25

combined.to_csv("new_schedule_multi.csv", index=False)
print("\nSaved: new_schedule_multi.csv")
