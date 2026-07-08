"""
00_fetch_wcif_cache.py
=======================
Fetch and cache the public WCIF (competition schedule) for every competition
that has 333, 444, 555, 666, or 333fm results in the WCA export.

A competition's WCIF contains the *full* event schedule (all events, not just
the one it's fetched for), so caching one WCIF per competition is enough to
cover every event analyzed by the rest of this pipeline.

Input:
  WCA_export_Results.tsv  (official WCA public data export)

Output:
  wcif_cache_public/{competitionId}.json  (one cached file per competition)

This step only needs to be run once; re-running it skips competitions that
are already cached. It makes one HTTP request per competition (rate-limited),
so expect it to take roughly an hour for the full history.
"""

import os
import time
import json
import pandas as pd
import requests

RESULTS_TSV = "WCA_export_Results.tsv"
CACHE_DIR   = "wcif_cache_public"
EVENTS      = ["333", "444", "555", "666", "333fm"]
WCIF_URL    = "https://www.worldcubeassociation.org/api/v0/competitions/{cid}/wcif/public"
SLEEP_SECONDS = 0.25  # be polite to the API

os.makedirs(CACHE_DIR, exist_ok=True)

# ── 1. Find every competition that has a relevant event ──────────────────────
print("Scanning Results TSV for competition IDs...")
comp_ids = set()
for chunk in pd.read_csv(RESULTS_TSV, sep="\t", chunksize=200_000,
                         usecols=["competitionId", "eventId"]):
    comp_ids.update(chunk.loc[chunk["eventId"].isin(EVENTS), "competitionId"].unique())

comp_ids = sorted(comp_ids)
print(f"  {len(comp_ids):,} competitions to fetch")

# ── 2. Fetch + cache WCIF for each competition ────────────────────────────────
fetched, cached, failed = 0, 0, 0
for i, cid in enumerate(comp_ids):
    if i % 500 == 0:
        print(f"  {i}/{len(comp_ids)}...")

    cache_path = os.path.join(CACHE_DIR, f"{cid}.json")
    if os.path.exists(cache_path):
        cached += 1
        continue

    try:
        r = requests.get(WCIF_URL.format(cid=cid), timeout=30)
        if r.status_code != 200:
            failed += 1
            continue
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(r.json(), f)
        fetched += 1
        time.sleep(SLEEP_SECONDS)
    except Exception:
        failed += 1

print(f"\nDone. Fetched: {fetched:,} | Already cached: {cached:,} | Failed: {failed:,}")
