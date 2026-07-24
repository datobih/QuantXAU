"""
Dukascopy historical tick fetcher — gold bid/ask ticks back to ~2003, FREE.

This breaks the Exness ~3-week tick wall for RESEARCH. The price move around a
release is broker-independent, so Dukascopy ticks let us replay hundreds of past
releases to test the impulse-continuation edge at scale. (Fill/spread quality is
still Exness-specific — validate that separately on the demo.)

Raw scheme (no auth):
  https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5
  MM0 = month 0-indexed (Jan=00). File is LZMA-compressed; 20 bytes/record,
  big-endian: uint32 ms-since-hour, uint32 ask*pf, uint32 bid*pf, f32 askvol,
  f32 bidvol. XAUUSD price factor = 1000. Timestamps are UTC/GMT.

Output matches tick_archive exactly (per-day parquet: time_msc, bid, ask) but in
a separate dir so Dukascopy and Exness archives never mix:
  data/raw/ticks_duka/XAUUSD_YYYY-MM-DD.parquet
so tick_archive.load_day(..., root=DUKA_DIR) and every existing EDA script read
it unchanged.

Usage (from ml_test/):
  python -m news_drift.duka_fetch --symbol XAUUSD --start 2020-01-01 --end 2024-12-31
  python -m news_drift.duka_fetch --symbol XAUUSD --releases ../data/raw/ff_history.csv
      # ^ fetch ONLY the days that have a high-impact USD release (fast, targeted)
"""

from __future__ import annotations

import argparse
import lzma
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m0:02d}/{d:02d}/{h:02d}h_ticks.bi5"
PRICE_FACTOR = {"XAUUSD": 1000.0, "XAGUSD": 1000.0}   # default 1e5 for FX majors
DUKA_DIR = Path("../data/raw/ticks_duka")
_REC = struct.Struct(">IIIff")


def _price_factor(sym: str) -> float:
    return PRICE_FACTOR.get(sym.upper(), 1e5)


def fetch_hour(sym: str, dt_hour: datetime, retries: int = 3) -> list[tuple[int, float, float]]:
    """Return [(time_msc_epoch, bid, ask)] for one UTC hour; [] if no ticks."""
    pf = _price_factor(sym)
    url = URL.format(sym=sym, y=dt_hour.year, m0=dt_hour.month - 1,
                     d=dt_hour.day, h=dt_hour.hour)
    hour_ms = int(dt_hour.replace(tzinfo=timezone.utc).timestamp() * 1000)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []          # no data for this hour (market closed)
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    if not raw:
        return []
    data = lzma.decompress(raw)
    out = []
    for off in range(0, len(data), 20):
        ms, ask, bid, _av, _bv = _REC.unpack_from(data, off)
        out.append((hour_ms + ms, bid / pf, ask / pf))
    return out


def fetch_day(sym: str, day: date, hours: range = range(24),
              workers: int = 8) -> pd.DataFrame:
    """Fetch a day's hourly bi5 files concurrently (polite pool)."""
    base = datetime(day.year, day.month, day.day)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_hour, sym, base + timedelta(hours=h)): h for h in hours}
        for fut in as_completed(futs):
            rows.extend(fut.result())
    if not rows:
        return pd.DataFrame(columns=["time_msc", "bid", "ask"])
    df = pd.DataFrame(rows, columns=["time_msc", "bid", "ask"])
    return df.sort_values("time_msc").reset_index(drop=True)


def _target_days(args) -> list[date]:
    if args.releases:
        cal = pd.read_csv(args.releases)
        m = pd.Series(True, index=cal.index)
        if "currency" in cal and not args.all_currencies:
            m &= cal["currency"].astype(str).str.upper().eq("USD")
        if "impact" in cal and args.impact:
            m &= cal["impact"].astype(str).str.title().isin(args.impact.split(","))
        ts = pd.to_datetime(cal.loc[m, "datetime"], utc=True, errors="coerce")
        days = sorted({t.date() for t in ts.dropna()
                       if args.start_d <= t.date() <= args.end_d})
        return [d for d in days if d.weekday() < 5]
    d, out = args.start_d, []
    while d <= args.end_d:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--releases", default=None,
                    help="ff_history.csv — fetch only days with a matching release (targeted)")
    ap.add_argument("--impact", default="High",
                    help="comma list of impacts to target with --releases (default High)")
    ap.add_argument("--all-currencies", action="store_true",
                    help="with --releases, don't restrict to USD")
    ap.add_argument("--hours", default="0-23",
                    help="UTC hour range to fetch per day, e.g. 12-19 (release windows). Default all")
    ap.add_argument("--delay", type=float, default=0.3)
    a = ap.parse_args()
    a.start_d = date.fromisoformat(a.start)
    a.end_d = date.fromisoformat(a.end)
    h0, h1 = (int(x) for x in a.hours.split("-"))
    a.hour_range = range(h0, h1 + 1)

    DUKA_DIR.mkdir(parents=True, exist_ok=True)
    days = _target_days(a)
    have = {p.stem for p in DUKA_DIR.glob(f"{a.symbol}_*.parquet")}
    todo = [d for d in days if f"{a.symbol}_{d.isoformat()}" not in have]
    print(f"{a.symbol}: {len(days)} target days, {len(todo)} to fetch "
          f"({len(days)-len(todo)} already on disk)")

    n_ticks = 0
    for i, d in enumerate(todo, 1):
        df = fetch_day(a.symbol, d, hours=a.hour_range)
        if len(df):
            df.to_parquet(DUKA_DIR / f"{a.symbol}_{d.isoformat()}.parquet", index=False)
            n_ticks += len(df)
        if i % 20 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {d}  {len(df):7d} ticks  (cum {n_ticks:,})")
        time.sleep(a.delay)
    print(f"done: {n_ticks:,} ticks over {len(todo)} days -> {DUKA_DIR}")


if __name__ == "__main__":
    main()
