"""
Rolling tick archiver — because Exness only serves ~2-3 weeks of tick history,
the only way to ever have a tick-level backtest is to start hoarding now.

Each run pulls whatever bid/ask tick history the server still has (day by day,
UTC), and writes one file per day to data/raw/ticks/XAUUSDm_YYYY-MM-DD.parquet
(falls back to .csv.gz without pyarrow). Already-archived days are skipped, so
run it any time — weekly at minimum, since the server window slides.

    python -m news_drift.tick_archive                # from ml_test/
    python -m news_drift.tick_archive --symbol XAUUSDm --days 30

After a few months this becomes the dataset for honest spike-strategy studies:
exact spread at every quote, gap sizes, real stop-fill simulation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

OUT_DIR = Path("../data/raw/ticks")


def _write(df: pd.DataFrame, base: Path) -> Path:
    try:
        p = base.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = base.with_suffix(".csv.gz")
        df.to_csv(p, index=False, compression="gzip")
        return p


def archive(symbol: str, days: int) -> None:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")
    try:
        mt5.symbol_select(symbol, True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        have = {p.stem for p in OUT_DIR.glob(f"{symbol}_*")}
        today = datetime.now(timezone.utc).date()
        n_new = n_ticks = 0
        for back in range(days, 0, -1):   # oldest first — those expire soonest
            day = today - timedelta(days=back)
            if day.weekday() >= 5:
                continue
            stem = f"{symbol}_{day.isoformat()}"
            if stem in have:
                continue
            t0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            ticks = mt5.copy_ticks_range(symbol, t0, t0 + timedelta(days=1),
                                         mt5.COPY_TICKS_INFO)
            if ticks is None or len(ticks) == 0:
                continue
            df = pd.DataFrame(ticks)[["time_msc", "bid", "ask"]]
            path = _write(df, OUT_DIR / stem)
            n_new += 1
            n_ticks += len(df)
            print(f"  {day}  {len(df):8,d} ticks -> {path.name}")
        print(f"archived {n_new} new days, {n_ticks:,} ticks "
              f"({len(list(OUT_DIR.glob(f'{symbol}_*')))} days total on disk)")
    finally:
        mt5.shutdown()


def load_day(day: str, symbol: str = "XAUUSDm", root: "Path | None" = None) -> pd.DataFrame:
    """Load one archived day; adds dt (UTC-naive, server clock) and spread.
    root defaults to the Exness archive dir; pass data/raw/ticks_duka for Dukascopy."""
    base = (root or OUT_DIR) / f"{symbol}_{day}"
    p = base.with_suffix(".parquet")
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(base.with_suffix(".csv.gz"))
    df["dt"] = pd.to_datetime(df["time_msc"], unit="ms")
    df["spread"] = df["ask"] - df["bid"]
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--days", type=int, default=25,
                    help="how far back to attempt (server keeps ~2-3 weeks)")
    a = ap.parse_args()
    archive(a.symbol, a.days)
