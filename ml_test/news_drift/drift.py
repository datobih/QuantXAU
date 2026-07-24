"""
Impulse / drift decomposition — the core measurement of the study.

For each aligned event (minute grid, 0 = release minute):

    impulse  = Close[k-1]  - Open[0]        (the first k minutes; NOT traded)
    drift(T) = Close[T-1]  - Close[k-1]     (entered after the impulse settles)

Everything downstream trades the DRIFT phase only: entry at the close of minute
k-1 (i.e. k minutes after the release), which sidesteps the release-instant
spread blowout and slippage that made the old news_reaction_backtest.py entries
untradeable.

The event table produced by `build_event_table` has one row per event:
    surprise_z      standardized real surprise (actual - consensus)
    usd_polarity    +1 / -1 (see ff_calendar)
    exp_dir         expected GOLD direction from the surprise:
                        exp_dir = -usd_polarity * sign(surprise)
                    (USD-positive surprise => gold down). 0 when surprise == 0.
    impulse         signed price move over minutes [0, k)
    impulse_dir     sign(impulse)
    agree           impulse_dir == exp_dir (market moved the "textbook" way)
    drift_15/30/60/120  signed drift over minutes [k, T)
    entry_spread    Spread (points) on the entry bar -> cost model
    pre_atr         mean 1-min High-Low over the 30 pre-event minutes (vol regime)

Cost model
----------
round_trip_cost = entry_spread_points * point * SPREAD_WIDEN_MULT
`point` is the broker's price increment — infer it from the CSV with
align.infer_point (Exness XAUUSDm quotes 3 decimals => point = 0.001; a spread
of 280 points is $0.28, not $2.80). Spread on the entry bar is already the
*observed* post-news spread from the MT5 export; SPREAD_WIDEN_MULT (default
1.5) adds slippage headroom on top. Net per-trade PnL in $ per oz = |drift| in
the trade direction minus cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPREAD_WIDEN_MULT = 1.5      # slippage headroom on top of observed spread
DEFAULT_K = 3                # impulse length, minutes
DEFAULT_HORIZONS = (15, 30, 60, 120)


def build_event_table(
    paths: dict,
    events: pd.DataFrame,
    k: int = DEFAULT_K,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    point: float = 0.01,
) -> pd.DataFrame:
    """One row per event: surprise features + impulse + drift outcomes + costs."""
    rows = []
    for i, ev in events.iterrows():
        w = paths[i]
        try:
            open0 = w.at[0, "Open"]
            entry_close = w.at[k - 1, "Close"]
        except KeyError:
            continue
        if np.isnan(open0) or np.isnan(entry_close):
            continue

        surprise = ev["surprise"]
        pol = ev["usd_polarity"]
        exp_dir = int(-pol * np.sign(surprise)) if pol != 0 and surprise != 0 else 0

        impulse = entry_close - open0
        row = {
            "ts_utc": ev["ts_utc"],
            "ts_server": ev["ts_server"],
            "event_key": ev["event_key"],
            "cluster_id": ev.get("cluster_id", i),
            "cluster_primary": bool(ev.get("cluster_primary", True)),
            "surprise": surprise,
            "surprise_z": ev["surprise_z"],
            "usd_polarity": pol,
            "exp_dir": exp_dir,
            "impulse": impulse,
            "impulse_dir": int(np.sign(impulse)),
            "entry_price": entry_close,
            "entry_spread": w.at[k - 1, "Spread"],
            "pre_atr": float((w.loc[-30:-1, "High"] - w.loc[-30:-1, "Low"]).mean()),
        }
        row["agree"] = row["impulse_dir"] != 0 and row["impulse_dir"] == exp_dir
        for T in horizons:
            try:
                cT = w.at[T - 1, "Close"]
            except KeyError:
                cT = np.nan
            row[f"drift_{T}"] = cT - entry_close if not np.isnan(cT) else np.nan
        rows.append(row)

    tab = pd.DataFrame(rows)
    if len(tab):
        tab["year"] = tab["ts_server"].dt.year
        tab["round_trip_cost"] = tab["entry_spread"].fillna(
            tab["entry_spread"].median()) * point * SPREAD_WIDEN_MULT
    return tab.reset_index(drop=True)


def trade_pnl(tab: pd.DataFrame, direction: pd.Series, horizon: int) -> pd.Series:
    """Net $/oz PnL of holding `direction` (+1/-1/0) from minute k to minute T."""
    gross = direction * tab[f"drift_{horizon}"]
    return gross - tab["round_trip_cost"].where(direction != 0, 0.0)
