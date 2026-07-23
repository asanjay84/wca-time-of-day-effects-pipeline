# Time-of-Day Effects on Competitive Speedcubing Performance

Code and data supporting a sample time-of-day analysis of World Cube
Association (WCA) records. Similar pipeline was used to publish in the *Journal of Circadian Rhythms*
as "Time-of-Day Effects on Competitive Speedcubing Performance."

This repository contains a subset of the project's analysis pipeline that
produces:
- The average-record rate figures for 3x3x3 through 6x6x6 (`avg_fig_333_rate.png`,
  `avg_fig_444_rate.png`, `avg_fig_555_rate.png`, `avg_fig_666_rate.png`)
- The raw record-count distribution figures (`figures/figure_raw_counts_333.png`,
  `figures/figure_raw_counts_bigcubes.png`)
- The post-prandial dip / evening-spike quantification behind those figures
  (`data/processed/dip_quantification_summary.csv`)

## What the analysis does

For each event (333, 444, 555, 666), every competition round is assigned a local
clock time (from the competition's published schedule) and binned into 15-minute
buckets. Record-breaking rate is computed as **records / attempts** within each
bucket — an exposure-adjusted rate — so that bins with more
competitors don't look artificially more record-prone. A within-competition
permutation test builds a null distribution that respects each competition's own
schedule, and a Poisson regression tests for a time-of-day effect after
controlling for round stage (final vs. early) and year.

## Pipeline

```
WCA_export_Results.tsv ──────────┐
WCA_export_Competitions.tsv ─────┤
WCA_export_RoundTypes.tsv ───────┤
                                  │
00_fetch_wcif_cache.py           │        (fetches competition schedules
     -> wcif_cache_public/*.json │         from the WCA API; ~1 hour, one-time)
                                  │
              │                  │
              v                  │
01_build_schedule.py <───────────┘
     -> new_schedule_multi.csv

new_schedule_multi.csv ──────────┐
WCA_export_Competitions.tsv ─────┤
WCA_export_RoundTypes.tsv ───────┤
WCA_export_Results.tsv ──────────┤
                                  v
              02_build_round_tables.py
                     -> new_round_table_{333,444,555,666}.csv
                                  │
              ┌───────────────────┼───────────────────┐
              v                   v                   v
03_plot_avg_rate_figures.py   04_dip_spike_          05_plot_raw_count_
  -> avg_fig_{event}_rate.png    quantification.py     figures.py
  -> avg_permutation_*.csv        -> dip_quantification  -> figures/figure_raw_
  -> avg_regression_*.csv            _summary.csv            counts_*.png
```

Scripts `01`–`05` are modified copies of the exact code used for the paper
(renamed for clarity of run order).

## Quick start

The derived data tables (`new_schedule_multi.csv`, `new_round_table_*.csv`) are
already included in this repo, so you can generate figures
without touching WCA's raw export or the WCIF cache:

```bash
python -m venv venv
venv\Scripts\activate        # on Windows
pip install -r requirements.txt

python 03_plot_avg_rate_figures.py     # -> avg_fig_{333,444,555,666}_rate.png
python 04_dip_spike_quantification.py  # -> data/processed/dip_quantification_summary.csv
python 05_plot_raw_count_figures.py    # -> figures/figure_raw_counts_*.png
```

## Full reproduction from scratch

To rebuild the derived tables themselves from raw WCA data:

1. **Download the WCA results export.** This analysis was built against WCA's
   pre-"v2" TSV export (camelCase columns: `competitionId`, `regionalSingleRecord`,
   `value1`-`value5`, etc.), available from
   [worldcubeassociation.org/export/results](https://www.worldcubeassociation.org/export/results).
   You need `WCA_export_Results.tsv`, `WCA_export_Competitions.tsv`, and
   `WCA_export_RoundTypes.tsv` in the repo root.

   > **Note:** WCA migrated to a new "v2" export format (snake_case columns, a
   > separate `result_attempts` table, no `value1`-`value5`) in early 2026. A
   > freshly downloaded export today will use that new schema, so
   > `02_build_round_tables.py` (which expects the old column names) will need
   > small column-name adjustments before it will run against a v2 export.

2. **Fetch the competition schedules:**
   ```bash
   python 00_fetch_wcif_cache.py
   ```
   This calls the public WCA API once per competition so
   it takes roughly an hour and produces `wcif_cache_public/`
   (~2.4GB of cached JSON, not committed to this repo).

4. **Build the schedule and round tables:**
   ```bash
   python 01_build_schedule.py
   python 02_build_round_tables.py
   ```

5. Then run `03`–`05` as in the quick start above.

## Repository contents

| Path | Description |
|---|---|
| `00_fetch_wcif_cache.py` | Fetches per-competition schedules from the public WCA API |
| `01_build_schedule.py` | Parses the WCIF cache into local-time round schedules |
| `02_build_round_tables.py` | Joins schedules + results into per-event round tables |
| `03_plot_avg_rate_figures.py` | Exposure-adjusted average-record rate, permutation test, Poisson regression, figures |
| `04_dip_spike_quantification.py` | Post-prandial dip / evening-spike statistics |
| `05_plot_raw_count_figures.py` | Raw (non-exposure-adjusted) record count distribution figures |
| `new_schedule_multi.csv` | Derived: competition/event/round → local start time |
| `new_round_table_{event}.csv` | Derived: per-round records, attempts, local time, round stage |
| `avg_fig_{event}_rate.png` | Output figure: average-record rate by time of day |
| `avg_permutation_{event}.csv`, `avg_regression_{event}.csv` | Output statistics behind the rate figures |
| `figures/figure_raw_counts_*.png` | Output figure: raw record counts by time of day |
| `data/processed/dip_quantification_summary.csv` | Output statistics: dip/spike quantification |

## Data provenance and license

The underlying competition data is owned and maintained by the World Cube
Association and published at
[[worldcubeassociation.org/export/results](https://www.worldcubeassociation.org/export/results)].
The derived CSVs in this repo (`new_schedule_multi.csv`, `new_round_table_*.csv`)
are aggregated/derived from that public export.

The code in this repository is released under the [MIT License](LICENSE).

## Citation

If you use this code or data, please cite:

> [Adireddi, S. (2026). Time-of-Day Effects on Competitive Speedcubing Performance. Journal of Circadian Rhythms, 24(1). https://doi.org/10.5334/jcr.266]
