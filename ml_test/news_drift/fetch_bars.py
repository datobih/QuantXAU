"""
Pull M1 bars programmatically from the running MT5 terminal — no manual chart
exports. Writes the repo-standard tab-separated CSV so align.load_gold_m1 (and
everything else in this repo) reads it unchanged.

How deep the history goes is decided by the broker's server, not by us: we page
backwards in 60-day chunks until the server returns nothing three chunks in a
row. Bar `time` from the API is the bar-open in *server* clock (verified UTC+0
for Exness via check_server_clock), so timestamps land on the same clock as the
terminal's own exports.

Run (terminal must be running and logged in):
    python -m news_drift.fetch_bars --symbol XAUUSDm --out ../data/raw/XAUUSD1.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_DAYS = 60
MAX_EMPTY_CHUNKS = 3


def fetch_all_m1(symbol: str, end: datetime | None = None) -> pd.DataFrame:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()} — terminal running?")
    try:
        term = mt5.terminal_info()
        if term is not None and not term.connected:
            raise RuntimeError("terminal running but not connected to the broker")
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select({symbol!r}) failed: {mt5.last_error()}")

        end = end or datetime.now(timezone.utc) + timedelta(days=1)
        floor = datetime(2000, 1, 1, tzinfo=timezone.utc)
        chunks, empty = [], 0
        hi = end
        oldest = None   # oldest bar time seen so far; chunks must push this back
        while empty < MAX_EMPTY_CHUNKS and hi > floor:
            lo = hi - timedelta(days=CHUNK_DAYS)
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, lo, hi)
            if rates is None or len(rates) == 0:
                empty += 1
            elif oldest is not None and rates["time"].min() >= oldest:
                # server re-served bars we already have -> start of history reached
                # (MT5 answers out-of-history requests with the boundary bar)
                break
            else:
                empty = 0
                oldest = int(rates["time"].min())
                chunks.append(pd.DataFrame(rates))
                print(f"  {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}: {len(rates):6d} bars")
            hi = lo
        if not chunks:
            raise RuntimeError(f"no M1 history returned for {symbol}")
    finally:
        mt5.shutdown()

    df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset="time")
    df = df.sort_values("time").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["time"], unit="s")
    return df


def write_repo_csv(df: pd.DataFrame, out: Path) -> None:
    """Repo-standard MT5 export layout: tab-separated, <DATE> header style."""
    dec = 3 if (np.round(df["close"] * 100, 6) % 1 > 0).any() else 2
    fmt = f"{{:.{dec}f}}".format
    out_df = pd.DataFrame({
        "<DATE>": df["dt"].dt.strftime("%Y.%m.%d"),
        "<TIME>": df["dt"].dt.strftime("%H:%M:%S"),
        "<OPEN>": df["open"].map(fmt),
        "<HIGH>": df["high"].map(fmt),
        "<LOW>": df["low"].map(fmt),
        "<CLOSE>": df["close"].map(fmt),
        "<TICKVOL>": df["tick_volume"].astype(int),
        "<VOL>": df["real_volume"].astype(int),
        "<SPREAD>": df["spread"].astype(int),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--out", default="../data/raw/XAUUSD1.csv")
    args = ap.parse_args()

    print(f"fetching all available M1 history for {args.symbol}...")
    df = fetch_all_m1(args.symbol)
    print(f"total: {len(df):,} bars, {df['dt'].min()} .. {df['dt'].max()}")
    write_repo_csv(df, Path(args.out))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
