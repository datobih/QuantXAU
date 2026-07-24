"""
Forward paper-trade logger — the ONLY legitimate validator left for CPI ideas.

FROZEN SPEC (registered 2026-07-22, discovered on 2025-01..2026-07 in-sample;
that window is spent and can never validate anything again):

    CPI-FADE: on every US CPI release day (cluster containing any of core CPI
    m/m, CPI m/m, CPI y/y; direction number = core m/m, fallback m/m, then y/y;
    skip if that number's surprise is zero):
      - at minute 3 after release, enter AGAINST the surprise direction
        (hot CPI -> LONG gold, cool CPI -> SHORT gold)
      - exit at minute 30
      - disaster stop $25/oz, frozen 2026-07-22: insurance only, beyond every
        observed path (worst in-sample MAE -14.15); MAE/MFE are logged per
        trade so any tighter-stop decision is made on FORWARD data only
      - cost = entry-bar spread * 1.5
    In-sample 2025-26 benchmark, CORRECTED 2026-07-22 (the first-reported
    n=25/t=4.73 double-counted core-y/y rows against the same price path): at
    honest day level it is n=17 days, mean +3.65 $/oz, hit 0.59, t=1.75 — and
    horizon-fragile (15m/45m/60m all ~zero). A weak candidate; the forward
    record is kept because logging is free, not because the case is strong.

    Read-out (committed in advance): after >=10 forward CPI days —
      mean > 0 and hit >= 0.6 -> regime effect supported, consider small size;
      otherwise -> dead, and this file is the tombstone.

Only releases AFTER FREEZE_DATE count toward the record. Run any time (weekly,
or the evening after each CPI print): it extends ff_history.csv, finds new CPI
days, pulls the bar window straight from the MT5 terminal, and appends to
data/processed/forward_cpi_log.csv (idempotent — already-logged days skipped).

    python -m news_drift.forward_log            # from ml_test/
    python -m news_drift.forward_log --dry-run  # compute, print, don't write
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import align, ff_calendar
from .fetch_history import fetch_range

FREEZE_DATE = pd.Timestamp("2026-07-22")
K, T_EXIT = 3, 30
DISASTER_STOP = 25.0   # $/oz, frozen 2026-07-22 — insurance only, never optimized
PRIORITY = ["CORE_CPI_MM", "CPI_MM", "CPI_YY"]
HISTORY_CSV = Path("../data/raw/ff_history.csv")
LOG_CSV = Path("../data/processed/forward_cpi_log.csv")
SPREAD_MULT = 1.5


def _bars_from_mt5(symbol: str, t0_utc: pd.Timestamp) -> pd.DataFrame | None:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        return None
    try:
        lo = (t0_utc - pd.Timedelta(minutes=5)).to_pydatetime().replace(tzinfo=timezone.utc)
        hi = (t0_utc + pd.Timedelta(minutes=T_EXIT + 10)).to_pydatetime().replace(tzinfo=timezone.utc)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, lo, hi)
    finally:
        mt5.shutdown()
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df.index = pd.to_datetime(df["time"], unit="s")   # server clock == UTC (verified)
    return df


def pick_direction_row(cluster: pd.DataFrame) -> pd.Series | None:
    for key in PRIORITY:
        cand = cluster[(cluster["event_key"] == key) & (cluster["surprise"] != 0)]
        if len(cand):
            return cand.iloc[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--since", default=None,
                    help="Override start date (TESTING ONLY; official record starts at freeze)")
    args = ap.parse_args()
    since = pd.Timestamp(args.since) if args.since else FREEZE_DATE
    if args.since:
        print(f"*** --since override: rows before {FREEZE_DATE.date()} are NOT part of the record ***")

    print("updating calendar...")
    fetch_range(date.today() - timedelta(days=21), date.today(), HISTORY_CSV)
    cal = ff_calendar.load_history_csv(HISTORY_CSV, source_tz="UTC")
    ev = ff_calendar.dedupe_simultaneous(ff_calendar.usd_events(cal))
    ev = align.attach_server_time(ev)   # server == UTC for this broker

    logged: set[str] = set()
    log = pd.DataFrame()
    if LOG_CSV.exists():
        log = pd.read_csv(LOG_CSV)
        logged = set(log["date"].astype(str))

    new_rows = []
    for _, cluster in ev[ev["ts_server"] >= since].groupby("cluster_id"):
        if not cluster["event_key"].isin(PRIORITY).any():
            continue
        pick = pick_direction_row(cluster)
        day = str(cluster["ts_server"].iloc[0].date())
        if day in logged:
            continue
        if pick is None:
            new_rows.append({"date": day, "note": "zero surprise - no trade"})
            continue
        t0 = pick["ts_server"].floor("min")
        bars = _bars_from_mt5(args.symbol, pick["ts_utc"])
        if bars is None or t0 + pd.Timedelta(minutes=T_EXIT - 1) not in bars.index:
            new_rows.append({"date": day, "note": "bars unavailable - rerun later"})
            continue
        fade_dir = int(pick["usd_polarity"] * np.sign(pick["surprise"]))  # AGAINST surprise
        entry_bar = bars.loc[t0 + pd.Timedelta(minutes=K - 1)]
        exit_bar = bars.loc[t0 + pd.Timedelta(minutes=T_EXIT - 1)]
        seg = bars.loc[t0 + pd.Timedelta(minutes=K):t0 + pd.Timedelta(minutes=T_EXIT - 1)]
        point = 10.0 ** -max(str(entry_bar["close"])[::-1].find("."), 2)
        cost = entry_bar["spread"] * point * SPREAD_MULT
        mae = (fade_dir * ((seg["low"] if fade_dir > 0 else seg["high"]) - entry_bar["close"])).min()
        mfe = (fade_dir * ((seg["high"] if fade_dir > 0 else seg["low"]) - entry_bar["close"])).max()
        if mae <= -DISASTER_STOP:
            pnl = -DISASTER_STOP - cost
            note = f"disaster stop -{DISASTER_STOP} hit"
        else:
            pnl = fade_dir * (exit_bar["close"] - entry_bar["close"]) - cost
            note = ""
        new_rows.append({
            "date": day, "src": pick["event_key"], "surprise": pick["surprise"],
            "surprise_z": round(pick["surprise_z"], 2), "fade_dir": fade_dir,
            "entry": entry_bar["close"], "exit": exit_bar["close"],
            "mae": round(mae, 2), "mfe": round(mfe, 2),
            "cost": round(cost, 2), "pnl": round(pnl, 2), "note": note,
        })

    if not new_rows:
        print("no new CPI days since last run.")
    else:
        add = pd.DataFrame(new_rows)
        print("\nnew rows:")
        print(add.to_string(index=False))
        if not args.dry_run:
            log = pd.concat([log, add], ignore_index=True)
            LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
            log.to_csv(LOG_CSV, index=False)
            print(f"appended -> {LOG_CSV}")

    rec = log[(log.get("note", "") == "") | log.get("note", pd.Series(dtype=str)).isna()] \
        if len(log) else pd.DataFrame()
    rec = rec[pd.to_datetime(rec["date"]) >= FREEZE_DATE] if len(rec) else rec
    if len(rec):
        p = rec["pnl"].astype(float)
        print(f"\nOFFICIAL FORWARD RECORD: n={len(p)}  mean={p.mean():+.2f}  "
              f"hit={(p > 0).mean():.2f}  total={p.sum():+.2f}"
              + (f"  t={p.mean() / (p.std(ddof=1) / np.sqrt(len(p))):+.2f}" if len(p) > 2 else ""))
        print("verdict rule: >=10 days, mean>0 and hit>=0.6 -> supported; else dead.")
    else:
        print("\nOFFICIAL FORWARD RECORD: empty — next CPI print starts the clock.")


if __name__ == "__main__":
    main()
