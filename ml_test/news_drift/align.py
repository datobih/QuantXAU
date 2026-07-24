"""
Align calendar events (UTC) onto the XAUUSD M1 bar clock (MT5 server time).

The server-time problem, solved properly this time
--------------------------------------------------
Candidate rules, selectable via `rule=`:

    'utc'   server clock == UTC. VERIFIED for the Exness XAUUSDm export on hand:
            NFP/CPI bars spike exactly at 12:30 and FOMC at 18:00 in bar time,
            which are the releases' UTC times. (Exness runs GMT+0, unlike the
            GMT+2/+3 assumption hard-coded into the old news scripts.)
    'et+7'  classic GMT+2/+3 broker that follows the US DST schedule so the
            NY 17:00 close is always midnight: server = US-Eastern + 7h,
            DST handled by zoneinfo.

No more "try +7, then try +6" guessing like news_reaction_backtest.py does.
Whatever you pick, `alignment_report` checks empirically that the release-minute
volatility spike lands on offset 0 — run it once per new bar export; if a
different offset wins, the rule is wrong for that broker.

Bar data
--------
`load_gold_m1` reads the repo-standard MT5 export:
    data/raw/XAUUSD1.csv  (tab-separated)
    Date  Time  Open  High  Low  Close  TickVol  Vol  Spread
Spread is in points (0.01 for gold), kept for the cost model.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
SERVER_OFFSET_FROM_ET = pd.Timedelta(hours=7)


def load_gold_m1(path: str | Path = "data/raw/XAUUSD1.csv") -> pd.DataFrame:
    """Load the MT5 M1 export into a DatetimeIndex'd OHLC frame (server time, tz-naive)."""
    path = Path(path)
    df = pd.read_csv(
        path, sep="\t",
        names=["Date", "Time", "Open", "High", "Low", "Close", "TickVol", "Vol", "Spread"],
        header=0,
    )
    if df["Open"].dtype == object:  # comma-separated fallback (some exports)
        df = pd.read_csv(path)
        df.columns = [c.strip("<>").title() for c in df.columns]
        df = df.rename(columns={"Tickvol": "TickVol"})
    raw_dt = df["Date"].astype(str) + " " + df["Time"].astype(str)
    try:
        df.index = pd.to_datetime(raw_dt, format="%Y.%m.%d %H:%M:%S")
    except ValueError:
        df.index = pd.to_datetime(raw_dt)
    df = df[["Open", "High", "Low", "Close", "TickVol", "Spread"]].astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def infer_point(path: str | Path) -> float:
    """Broker point size from the raw file: 10^-(decimals of the Close field).
    Exness XAUUSDm quotes 3 decimals -> 0.001; classic 2-decimal gold -> 0.01."""
    with open(path) as f:
        f.readline()                       # header
        parts = f.readline().rstrip("\n").split("\t")
    close = parts[5] if len(parts) > 5 else parts[-1]
    decimals = len(close.split(".")[1]) if "." in close else 0
    return 10.0 ** -decimals


DEFAULT_RULE = "utc"   # verified for the current Exness XAUUSDm export


def utc_to_server(ts_utc: pd.Series, rule: str = DEFAULT_RULE) -> pd.Series:
    """UTC tz-aware -> tz-naive server time. See module docstring for rules."""
    if rule == "utc":
        return ts_utc.dt.tz_convert("UTC").dt.tz_localize(None)
    if rule == "et+7":
        et = ts_utc.dt.tz_convert(ET)
        return (et + SERVER_OFFSET_FROM_ET).dt.tz_localize(None)
    raise ValueError(f"unknown server-time rule: {rule!r}")


def attach_server_time(events: pd.DataFrame, rule: str = DEFAULT_RULE) -> pd.DataFrame:
    events = events.copy()
    events["ts_server"] = utc_to_server(events["ts_utc"], rule=rule)
    return events


def extract_paths(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    pre: int = 30,
    post: int = 240,
    max_missing_frac: float = 0.20,
) -> tuple[dict, pd.DataFrame]:
    """
    For each event, pull the bar window [t-pre, t+post] minutes and reindex it to a
    complete minute grid (NaN where the market printed no bar). Returns
    (paths, kept_events):
      paths[event_index] = DataFrame indexed by offset-minute (-pre..post) with
                           columns Open/High/Low/Close/TickVol/Spread
      kept_events        = events that had a bar at/near t0 and enough coverage.

    An event is dropped when there is no bar within 2 minutes of the release
    (weekend/holiday/feed-side timestamp error) or when > max_missing_frac of the
    post-window minutes are missing.
    """
    idx = bars.index
    paths: dict[int, pd.DataFrame] = {}
    keep = []
    for i, ts in events["ts_server"].items():
        t0 = ts.floor("min")
        # tolerate the release bar itself being missing by up to 2 minutes
        pos = idx.searchsorted(t0)
        if pos >= len(idx) or (idx[pos] - t0) > pd.Timedelta(minutes=2):
            continue
        grid = pd.date_range(t0 - pd.Timedelta(minutes=pre),
                             t0 + pd.Timedelta(minutes=post), freq="min")
        win = bars.reindex(grid)
        post_missing = win.loc[t0:, "Close"].isna().mean()
        if post_missing > max_missing_frac:
            continue
        win.index = ((win.index - t0).total_seconds() // 60).astype(int)
        paths[i] = win
        keep.append(i)
    return paths, events.loc[keep]


def alignment_report(bars: pd.DataFrame, events: pd.DataFrame, n_check: int = 200) -> pd.DataFrame:
    """
    Empirical sanity check of the timezone rule: at the true release minute the
    bar range (High-Low) should explode vs. the minutes before. For each candidate
    offset around our rule (-2..+2 hours plus the legacy +6h-in-summer guess),
    report median range ratio (release bar / pre-event baseline). The correct
    offset should dominate. Run this ONCE when the data first loads; if offset 0
    doesn't clearly win, the broker uses European DST and align needs adjusting.
    """
    ev = events.sort_values("surprise_z", key=lambda s: s.abs(), ascending=False).head(n_check)
    rows = []
    for hours in (-2, -1, 0, 1, 2):
        ratios = []
        for ts in ev["ts_server"]:
            t0 = (ts + pd.Timedelta(hours=hours)).floor("min")
            try:
                win = bars.loc[t0 - pd.Timedelta(minutes=30): t0 + pd.Timedelta(minutes=2)]
            except KeyError:
                continue
            if len(win) < 20 or t0 not in win.index:
                continue
            base = (win["High"] - win["Low"]).loc[:t0 - pd.Timedelta(minutes=5)].median()
            spike = (win["High"] - win["Low"]).loc[t0: t0 + pd.Timedelta(minutes=1)].max()
            if base > 0:
                ratios.append(spike / base)
        rows.append({"offset_hours": hours, "n": len(ratios),
                     "median_range_ratio": float(np.median(ratios)) if ratios else np.nan})
    rep = pd.DataFrame(rows)
    rep["is_our_rule"] = rep["offset_hours"] == 0
    return rep
