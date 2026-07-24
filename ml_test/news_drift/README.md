# news_drift — post-news drift study on XAUUSD

Does gold keep moving in the surprise direction *after* the initial news spike?
This package tests that, with real consensus surprises (ForexFactory
actual − forecast, not the change-vs-previous proxy used by the older
`news_reaction_backtest.py`) and entries **after** the impulse, not into the
release-instant spread blowout.

## Design

Each release is decomposed on the M1 grid (minute 0 = release):

```
| impulse (0..k min, NOT traded) | drift (k..T min, traded) |
```

Default k=3; horizons T ∈ {15, 30, 60, 120}. Three hypotheses:

- **H1 confirmation** — impulse agrees with the surprise sign → enter with it.
- **H2 fade** — impulse disagrees with the surprise → enter against the impulse.
- **H3 magnitude** — H1 gated on |surprise_z| ≥ threshold.

Honesty controls: expanding past-only surprise z-scores; observed post-news
spread × 1.5 as round-trip cost; NFP-style simultaneous releases collapsed to
one cluster primary; per-year breakdowns; and a **time-ordered holdout** — the
most recent 30% of events is locked behind a manifest
(`data/processed/news_drift_holdout.json`) and `study.run_holdout` refuses to
run without an explicit confirmation phrase. Exploration never touches it.

Timezone: server time = US-Eastern + 7h (DST-aware via zoneinfo), replacing the
old try-+7-then-+6 hack. `align.alignment_report` verifies empirically that the
release-minute volatility spike lands on offset 0.

## Usage (from the repo root, needs numpy+pandas)

```bash
# 1) one-time: fetch the FF calendar history (polite, resumable, ~2s/week)
python -m news_drift.fetch_history --start 2025-01-06 --out ../data/raw/ff_history.csv   # run from ml_test/

# 2) place your MT5 M1 export at data/raw/XAUUSD1.csv (tab-separated export)

# 3) sanity-check the timezone rule, then run the exploration study
python ml_test/run_news_drift.py --calendar data/raw/ff_history.csv --check-alignment
python ml_test/run_news_drift.py --calendar data/raw/ff_history.csv
```

`python -m news_drift.selftest` (from `ml_test/`) runs the whole pipeline on
synthetic bars with a planted drift and asserts it is detected — run it after
any change to the pipeline.

## Files

| file | role |
|---|---|
| `ff_calendar.py` | FF weekly JSON feed + history CSV parsing, event canonicalization, surprise_z |
| `fetch_history.py` | scrape historical weeks from forexfactory.com (embedded JSON) |
| `align.py` | gold M1 loader, ET+7h server-time rule, event window extraction, alignment check |
| `drift.py` | impulse/drift decomposition, per-event table, spread cost model |
| `study.py` | H1/H2/H3 evaluation, per-year/per-event tables, locked holdout |
| `selftest.py` | end-to-end test with a planted drift |

## Current data caveat

The only M1 export on hand spans **2026-04-08 → 2026-07-20 (~3.5 months)** —
roughly 3 occurrences of each monthly indicator. That is enough to wire and
smoke-test the study, **not** enough to conclude anything. Export a longer M1
history from MT5 (View → Symbols → Bars, or a chart export with
`Max bars in chart` raised) and re-run; the calendar side already covers
2025-01 onward and extends cheaply with `fetch_history --start 2019-01-07`.
