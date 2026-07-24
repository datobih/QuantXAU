"""
The drift study itself: three hypotheses, honest statistics, locked holdout.

Hypotheses
----------
H1 CONFIRMATION DRIFT: when the impulse direction AGREES with the surprise sign
   (exp_dir), enter WITH the impulse at minute k; drift continues to T.
H2 FADE: when the impulse DISAGREES with the surprise sign, the impulse was
   noise/positioning — enter AGAINST the impulse (i.e. with exp_dir).
H3 MAGNITUDE CONDITIONING: H1 but only for |surprise_z| >= z_min — small
   surprises carry no information, so drift should concentrate in the tail.

Honesty controls (the ways the rest of this repo burned itself, closed off)
---------------------------------------------------------------------------
- split_events(): time-ordered split; the most recent HOLDOUT_FRAC of events is
  written to disk and NEVER read by run_exploration(). run_holdout() refuses to
  run unless called with confirm="I ACCEPT THIS IS FINAL", and is meant to be
  executed exactly once, on the frozen spec, at the end.
- Events are naturally spaced days apart -> per-event returns are (near)
  independent, so plain t-stats are legitimate here, unlike the overlapping
  bar-label backtests elsewhere in ml_test/. Simultaneous releases are collapsed
  to cluster primaries first so NFP doesn't count 3x.
- Costs (observed post-news spread x widen multiplier) are inside every number.
- Per-year breakdown is printed for every cell; n is printed everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .drift import DEFAULT_HORIZONS, trade_pnl

HOLDOUT_FRAC = 0.30
HOLDOUT_MANIFEST = Path("data/processed/news_drift_holdout.json")
CONFIRM_PHRASE = "I ACCEPT THIS IS FINAL"


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_events(tab: pd.DataFrame, holdout_frac: float = HOLDOUT_FRAC
                 ) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Time-ordered split. Returns (exploration_table, cutoff). Rows at/after the
    cutoff are the holdout; the cutoff is persisted to a manifest so it can't
    quietly move between runs. If a manifest already exists, its cutoff WINS
    (you don't get a fresh split by re-running).
    """
    tab = tab.sort_values("ts_server").reset_index(drop=True)
    if HOLDOUT_MANIFEST.exists():
        cutoff = pd.Timestamp(json.loads(HOLDOUT_MANIFEST.read_text())["cutoff_server"])
    else:
        cut_i = int(len(tab) * (1 - holdout_frac))
        cutoff = tab["ts_server"].iloc[cut_i]
        HOLDOUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        HOLDOUT_MANIFEST.write_text(json.dumps({
            "cutoff_server": str(cutoff),
            "holdout_frac": holdout_frac,
            "n_total_at_creation": len(tab),
        }, indent=2))
    return tab[tab["ts_server"] < cutoff].copy(), cutoff


# ---------------------------------------------------------------------------
# Hypothesis directions
# ---------------------------------------------------------------------------

def h1_direction(tab: pd.DataFrame) -> pd.Series:
    """Confirmation: trade impulse direction only when it agrees with the surprise."""
    return pd.Series(np.where(tab["agree"], tab["impulse_dir"], 0), index=tab.index)


def h2_direction(tab: pd.DataFrame) -> pd.Series:
    """Fade: when impulse disagrees with the surprise, trade the surprise (against impulse)."""
    disagree = (~tab["agree"]) & (tab["exp_dir"] != 0) & (tab["impulse_dir"] != 0)
    return pd.Series(np.where(disagree, tab["exp_dir"], 0), index=tab.index)


def h3_direction(tab: pd.DataFrame, z_min: float) -> pd.Series:
    return h1_direction(tab).where(tab["surprise_z"].abs() >= z_min, 0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _cell(pnl: pd.Series) -> dict:
    pnl = pnl.dropna()
    n = len(pnl)
    if n < 5:
        return {"n": n, "mean": np.nan, "t": np.nan, "hit": np.nan, "total": np.nan}
    mean, sd = pnl.mean(), pnl.std(ddof=1)
    return {
        "n": n,
        "mean": round(float(mean), 3),
        "t": round(float(mean / (sd / np.sqrt(n))), 2) if sd > 0 else np.nan,
        "hit": round(float((pnl > 0).mean()), 3),
        "total": round(float(pnl.sum()), 2),
    }


def evaluate(tab: pd.DataFrame, direction: pd.Series,
             horizons=DEFAULT_HORIZONS) -> pd.DataFrame:
    """Per-horizon stats over TRADED events only (direction != 0), cluster primaries."""
    m = (direction != 0) & tab["cluster_primary"]
    rows = {}
    for T in horizons:
        rows[T] = _cell(trade_pnl(tab[m], direction[m], T))
    return pd.DataFrame(rows).T.rename_axis("horizon_min")


def evaluate_by_year(tab: pd.DataFrame, direction: pd.Series, horizon: int) -> pd.DataFrame:
    m = (direction != 0) & tab["cluster_primary"]
    pnl = trade_pnl(tab[m], direction[m], horizon)
    return pd.DataFrame({y: _cell(g) for y, g in pnl.groupby(tab.loc[m, "year"])}).T \
             .rename_axis("year")


def evaluate_by_event(tab: pd.DataFrame, direction: pd.Series, horizon: int,
                      min_n: int = 8) -> pd.DataFrame:
    m = (direction != 0) & tab["cluster_primary"]
    pnl = trade_pnl(tab[m], direction[m], horizon)
    out = pd.DataFrame({k: _cell(g) for k, g in pnl.groupby(tab.loc[m, "event_key"])}).T
    return out[out["n"] >= min_n].sort_values("t", ascending=False).rename_axis("event_key")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_exploration(tab: pd.DataFrame, z_grid=(0.5, 1.0, 1.5)) -> dict:
    """The full exploration report on the pre-cutoff events only."""
    explore, cutoff = split_events(tab)
    n_hold = int((tab["ts_server"] >= cutoff).sum())
    print(f"events: {len(explore)} exploration | {n_hold} HOLDOUT (locked, cutoff={cutoff})")
    print(f"exploration span: {explore['ts_server'].min():%Y-%m-%d} .. "
          f"{explore['ts_server'].max():%Y-%m-%d}\n")

    results = {}
    for name, dirs in [("H1_confirmation", h1_direction(explore)),
                       ("H2_fade", h2_direction(explore))]:
        print(f"=== {name} ===")
        r = evaluate(explore, dirs)
        print(r.to_string(), "\n")
        results[name] = r
    for z in z_grid:
        name = f"H3_confirmation_z>={z}"
        print(f"=== {name} ===")
        r = evaluate(explore, h3_direction(explore, z))
        print(r.to_string(), "\n")
        results[name] = r
    return results


def run_holdout(tab: pd.DataFrame, direction_fn, horizon: int, confirm: str = "") -> pd.DataFrame:
    """
    The one-shot final test. `direction_fn` is the FROZEN spec chosen during
    exploration (e.g. lambda t: h3_direction(t, 1.0)). Refuses to run casually.
    """
    if confirm != CONFIRM_PHRASE:
        raise RuntimeError(
            f"Holdout is single-use. Call with confirm={CONFIRM_PHRASE!r} only when "
            "the strategy spec is final. Running it repeatedly turns it into a test set.")
    if not HOLDOUT_MANIFEST.exists():
        raise RuntimeError("No holdout manifest — run exploration first.")
    cutoff = pd.Timestamp(json.loads(HOLDOUT_MANIFEST.read_text())["cutoff_server"])
    hold = tab[tab["ts_server"] >= cutoff].copy()
    dirs = direction_fn(hold)
    print(f"HOLDOUT: {len(hold)} events from {cutoff}")
    r = evaluate(hold, dirs, horizons=(horizon,))
    print(r.to_string())
    print("\nby year:")
    print(evaluate_by_year(hold, dirs, horizon).to_string())
    return r
