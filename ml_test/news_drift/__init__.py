"""
news_drift — a clean, reusable pipeline for studying post-news DRIFT on XAUUSD.

This is deliberately NOT another 500-line copy-paste script. The repo already has
28 copies of create_microstructure_features(); this track keeps its logic in one
place and puts thin experiment scripts on top.

Modules
-------
ff_calendar : fetch + parse the ForexFactory economic calendar into a normalized
              events table, and compute a REAL surprise = actual - consensus_forecast
              (standardized per indicator). This is the key upgrade over the existing
              news_reaction_backtest.py, which used change-vs-previous-release as a
              proxy for surprise.
align       : DST-aware alignment of release timestamps to the gold bar clock, and
              extraction of the price path around each event.
drift       : decompose each event into IMPULSE (0..k min) and DRIFT (k..T min) phases,
              and build the per-event feature/outcome table the study consumes.
study       : the analysis itself, with a STRICT out-of-sample split (explore on the
              earliest events, lock the most recent ~30% as a holdout touched once).

Conventions
-----------
- All timestamps internal to the pipeline are tz-naive in *gold server time* (the same
  clock as the XAUUSD1.csv bars), matching the rest of the repo.
- "surprise_z" is (actual - forecast) standardized by that indicator's own rolling std
  of (actual - forecast). Sign is in raw economic units (higher actual => positive z).
- "polarity" maps an indicator's positive surprise to a USD/gold direction so that a
  direction-aware test can be run without re-deriving it everywhere.
"""

from . import ff_calendar, align, drift, study  # noqa: F401

__all__ = ["ff_calendar", "align", "drift", "study"]
