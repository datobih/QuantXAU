"""
End-to-end self-test on synthetic data with a PLANTED drift.

Builds 2 years of fake M1 gold bars + a fake monthly calendar where, after each
release, price impulses in the surprise direction for 3 minutes and then drifts
the same way for 60 more (plus noise). If the pipeline is wired correctly:
  - the alignment check picks offset 0,
  - H1 (confirmation) shows a strongly positive t-stat at the 60m horizon,
  - H2 (fade) shows nothing,
  - the holdout split locks and refuses casual access.

Run:  python -m news_drift.selftest   (from inside ml_test/)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import align, drift, study
from .ff_calendar import _finalize


def make_synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", "2025-12-31 23:59", freq="min")
    idx = idx[idx.dayofweek < 5]
    n = len(idx)
    ret = rng.normal(0, 0.05, n)  # ~$0.05/min noise

    # monthly events: first business day at 15:30 server (8:30 ET winter)
    ev_rows = []
    months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
    pos_of = pd.Series(np.arange(n), index=idx)
    for m in months:
        t0 = m.replace(hour=15, minute=30)
        while t0 not in pos_of.index:
            t0 += pd.Timedelta(days=1)
        p = pos_of[t0]
        surprise = rng.normal(0, 50)                    # fake NFP surprise, thousands
        d = -np.sign(surprise)                          # gold moves against USD surprise
        ret[p:p + 3] += d * 1.0                         # impulse: $3 over 3 min
        ret[p + 3:p + 63] += d * 0.05                   # planted drift: $3 over next 60
        ev_rows.append({"ts": t0, "actual": 200 + surprise, "forecast": 200.0})

    price = 2400 + np.cumsum(ret)
    bars = pd.DataFrame({
        "Open": price - ret, "Close": price,
        "High": np.maximum(price, price - ret) + 0.02,
        "Low": np.minimum(price, price - ret) - 0.02,
        "TickVol": 100.0, "Spread": 15.0,
    }, index=idx)

    ev = pd.DataFrame(ev_rows)
    # ts is server time = ET + 7h -> reconstruct the UTC the calendar would carry
    et = ev["ts"] - pd.Timedelta(hours=7)
    cal = pd.DataFrame({
        "ts_utc": et.dt.tz_localize("America/New_York").dt.tz_convert("UTC"),
        "currency": "USD", "event": "Non-Farm Employment Change",
        "impact": "High", "actual": ev["actual"], "forecast": ev["forecast"],
        "previous": np.nan,
    })
    return bars, _finalize(cal)


def main() -> None:
    bars, cal = make_synthetic()
    from .ff_calendar import dedupe_simultaneous, usd_events
    # synthetic bars were built on the ET+7 clock, so exercise that rule here
    ev = align.attach_server_time(dedupe_simultaneous(usd_events(cal)), rule="et+7")
    print(f"synthetic: {len(bars):,} bars, {len(ev)} events (z needs 8 prior surprises)")

    rep = align.alignment_report(bars, ev, n_check=50)
    print("\nalignment check (offset 0 must win):")
    print(rep.to_string(index=False))
    best = rep.loc[rep["median_range_ratio"].idxmax(), "offset_hours"]
    assert best == 0, f"alignment picked offset {best}, expected 0"

    paths, ev = align.extract_paths(bars, ev)
    tab = drift.build_event_table(paths, ev)
    print(f"\nevent table: {len(tab)} rows")

    # study on the full synthetic table (no holdout manifest games in a test:
    # point the manifest somewhere disposable)
    import tempfile
    from pathlib import Path
    study.HOLDOUT_MANIFEST = Path(tempfile.mkdtemp()) / "holdout.json"
    study.run_exploration(tab)

    explore, _ = study.split_events(tab)
    t60 = study.evaluate(explore, study.h1_direction(explore))["t"].loc[60]
    assert t60 > 2.0, f"planted drift not detected: t(60m)={t60}"
    try:
        study.run_holdout(tab, study.h1_direction, 60, confirm="nope")
        raise AssertionError("holdout ran without confirmation!")
    except RuntimeError:
        pass
    print(f"\nSELFTEST PASS — planted drift detected (t(60m)={t60}), "
          "holdout lock enforced, alignment rule verified.")


if __name__ == "__main__":
    main()
