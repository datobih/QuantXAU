"""
Thin runner for the post-news drift study. All logic lives in news_drift/.

Usage (from the repo root):
    python ml_test/run_news_drift.py --calendar data/raw/ff_history.csv
    python ml_test/run_news_drift.py --calendar data/raw/ff_history.csv --check-alignment

Inputs:
    data/raw/XAUUSD1.csv     MT5 M1 export (tab-separated) — you have this.
    --calendar <file>        ForexFactory history CSV (community dump; columns are
                             auto-detected). Get one via ml_test/news_drift/README.md.

The holdout cutoff is created on first run and persisted to
data/processed/news_drift_holdout.json — after that, exploration only ever sees
events before the cutoff. Do not delete that file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from news_drift import align, drift, ff_calendar, study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calendar", required=True, help="ForexFactory history CSV")
    ap.add_argument("--gold", default="data/raw/XAUUSD1.csv")
    ap.add_argument("--calendar-tz", default="America/New_York",
                    help="Timezone of the calendar dump's timestamps (default ET)")
    ap.add_argument("--impacts", default="High", help="Comma list, e.g. High,Medium")
    ap.add_argument("--server-rule", default=align.DEFAULT_RULE, choices=["utc", "et+7"],
                    help="Bar-clock rule; verify with --check-alignment")
    ap.add_argument("--k", type=int, default=drift.DEFAULT_K, help="Impulse minutes (not traded)")
    ap.add_argument("--check-alignment", action="store_true",
                    help="Print the timezone-offset volatility check and exit")
    args = ap.parse_args()

    print("loading calendar...")
    cal = ff_calendar.load_history_csv(args.calendar, source_tz=args.calendar_tz)
    ev = ff_calendar.usd_events(cal, impacts=tuple(args.impacts.split(",")))
    ev = ff_calendar.dedupe_simultaneous(ev)
    ev = align.attach_server_time(ev, rule=args.server_rule)
    print(f"  {len(cal)} calendar rows -> {len(ev)} core USD events "
          f"({ev['ts_utc'].min():%Y-%m-%d} .. {ev['ts_utc'].max():%Y-%m-%d})")
    print(ev["event_key"].value_counts().to_string())

    print("\nloading gold bars...")
    bars = align.load_gold_m1(args.gold)
    print(f"  {len(bars):,} M1 bars ({bars.index.min()} .. {bars.index.max()})")

    # clip events to the bar data span
    ev = ev[(ev["ts_server"] >= bars.index.min()) & (ev["ts_server"] <= bars.index.max())]
    print(f"  {len(ev)} events inside the bar span")

    if args.check_alignment:
        print("\ntimezone alignment check (offset 0 should have the biggest ratio):")
        print(align.alignment_report(bars, ev).to_string(index=False))
        return

    print("\nextracting event windows...")
    paths, ev = align.extract_paths(bars, ev)
    point = align.infer_point(args.gold)
    print(f"  broker point size inferred: {point}")
    tab = drift.build_event_table(paths, ev, k=args.k, point=point)
    print(f"  {len(tab)} events with usable price paths, "
          f"{int(tab['cluster_primary'].sum())} cluster primaries")

    out = Path("data/processed/news_drift_events.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    print(f"  event table -> {out}\n")

    study.run_exploration(tab)


if __name__ == "__main__":
    main()
