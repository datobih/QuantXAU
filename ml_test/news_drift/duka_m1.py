"""
Dukascopy M1 candle fetcher — one bi5 file per day (fast), back to ~2003.

Record format (verified against tick-derived M1, exact match):
  BID_candles_min_1.bi5, LZMA, 24-byte big-endian records:
  uint32 sec-of-day, uint32 open*1000, uint32 close*1000, uint32 low*1000,
  uint32 high*1000, float32 volume   (XAUUSD price factor 1000; UTC timestamps)

Writes per-month parquet (resumable): data/raw/m1_duka/XAUUSD_YYYY-MM.parquet
with columns dt, open, high, low, close, vol.

  python -m news_drift.duka_m1 --start 2014-10-01 --end 2021-12-31
"""

from __future__ import annotations

import argparse
import lzma
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

OUT = Path("../data/raw/m1_duka")
PF = 1000.0


def fetch_day(day: date, retries: int = 5) -> pd.DataFrame | None:
    url = (f"https://datafeed.dukascopy.com/datafeed/XAUUSD/{day.year:04d}/"
           f"{day.month-1:02d}/{day.day:02d}/BID_candles_min_1.bi5")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = None
    for attempt in range(retries):
        try:
            raw = urllib.request.urlopen(req, timeout=30).read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))       # conn reset = rate limit; back off
    if raw is None:
        raise RuntimeError(f"gave up on {day}")
    if not raw:
        return None
    data = lzma.decompress(raw)
    n = len(data) // 24
    recs = [struct.unpack(">IIIIIf", data[i*24:(i+1)*24]) for i in range(n)]
    df = pd.DataFrame(recs, columns=["sec", "o", "c", "l", "h", "vol"])
    df = df[df["vol"] > 0]                       # drop empty filler minutes
    if df.empty:
        return None
    out = pd.DataFrame({
        "dt": pd.Timestamp(day) + pd.to_timedelta(df["sec"], unit="s"),
        "open": df["o"] / PF, "high": df["h"] / PF,
        "low": df["l"] / PF, "close": df["c"] / PF, "vol": df["vol"],
    })
    return out


def month_days(y: int, m: int, start: date, end: date) -> list[date]:
    d = date(y, m, 1)
    out = []
    while d.month == m:
        if d.weekday() < 5 and start <= d <= end:
            out.append(d)
        d += timedelta(days=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    OUT.mkdir(parents=True, exist_ok=True)

    months = sorted({(d.year, d.month) for d in
                     (start + timedelta(days=i) for i in range((end - start).days + 1))})
    total = 0
    for (y, m) in months:
        p = OUT / f"XAUUSD_{y:04d}-{m:02d}.parquet"
        if p.exists():
            continue
        days = month_days(y, m, start, end)
        frames = []
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(fetch_day, d): d for d in days}
            for f in as_completed(futs):
                r = f.result()
                if r is not None:
                    frames.append(r)
        if frames:
            mdf = pd.concat(frames).sort_values("dt").reset_index(drop=True)
            mdf.to_parquet(p, index=False)
            total += len(mdf)
            print(f"  {y}-{m:02d}: {len(mdf):6d} bars")
        time.sleep(1.0)                          # politeness gap between months
    print(f"done, {total:,} new bars -> {OUT}")


if __name__ == "__main__":
    main()
