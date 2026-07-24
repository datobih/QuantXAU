"""
ForexFactory calendar: fetch, parse, normalize, and compute real surprises.

Two sources feed the same normalized table:

1. Weekly JSON feed (free, official-ish, no auth):
       https://nfs.faireconomy.media/ff_calendar_thisweek.json
   Also ff_calendar_lastweek.json / ff_calendar_nextweek.json. Only covers the
   current window, so it is for *keeping the dataset fresh*, not for history.

2. Historical CSV dumps (community scrapes of the FF calendar, e.g. from GitHub
   or Kaggle). Column names vary between dumps, so `load_history_csv` does
   fuzzy column mapping. Point it at whatever file you download.

Normalized schema (one row per release):
    ts_utc      : tz-aware UTC timestamp of the release
    currency    : 'USD', 'EUR', ...
    event       : raw event title from FF
    event_key   : canonicalized key (e.g. 'NFP', 'CPI_MM') where recognized, else
                  a cleaned-up version of the title
    impact      : 'High' / 'Medium' / 'Low' / 'Non-Economic'
    actual, forecast, previous : floats (K/M/B/% suffixes parsed; sign preserved)
    surprise    : actual - forecast (NaN when either side missing)
    surprise_z  : surprise standardized by that event_key's own expanding std of
                  surprises (min 8 prior observations) — comparable across indicators
    usd_polarity: +1 if a positive surprise is USD-positive (NFP, CPI, retail sales),
                  -1 if USD-negative (unemployment rate, jobless claims), 0 unknown
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_{week}.json"  # thisweek|lastweek|nextweek

# ---------------------------------------------------------------------------
# Canonical event keys + polarity (is a positive surprise USD-positive?)
# Matched by regex against the FF event title, first hit wins.
# ---------------------------------------------------------------------------
EVENT_MAP: list[tuple[str, str, int]] = [
    (r"non[- ]?farm|nfp",                       "NFP",            +1),
    (r"unemployment rate",                      "UNEMP_RATE",     -1),
    (r"unemployment claims|jobless claims",     "CLAIMS",         -1),
    (r"average hourly earnings",                "AHE_MM",         +1),
    (r"core cpi.*y/y",                          "CORE_CPI_YY",    +1),
    (r"core cpi",                               "CORE_CPI_MM",    +1),
    (r"\bcpi\b.*y/y",                           "CPI_YY",         +1),
    (r"\bcpi\b",                                "CPI_MM",         +1),
    (r"core pce",                               "CORE_PCE_MM",    +1),
    (r"core ppi",                               "CORE_PPI_MM",    +1),
    (r"\bppi\b",                                "PPI_MM",         +1),
    (r"core retail sales",                      "CORE_RETAIL_MM", +1),
    (r"retail sales",                           "RETAIL_MM",      +1),
    (r"ism manufacturing pmi",                  "ISM_MFG",        +1),
    (r"ism services pmi|ism non-manufacturing", "ISM_SVC",        +1),
    (r"adp non-farm",                           "ADP",            +1),
    (r"advance gdp|prelim gdp|final gdp|\bgdp\b", "GDP_QQ",       +1),
    (r"federal funds rate",                     "FED_RATE",       +1),
    (r"jolts",                                  "JOLTS",          +1),
    (r"cb consumer confidence",                 "CB_CONF",        +1),
    (r"uom.*consumer sentiment|prelim uom",     "UOM_SENT",       +1),
    (r"durable goods",                          "DURABLES_MM",    +1),
    (r"building permits",                       "PERMITS",        +1),
    (r"housing starts",                         "HOUSING_STARTS", +1),
    (r"philly fed|philadelphia fed",            "PHILLY_FED",     +1),
    (r"empire state",                           "EMPIRE_STATE",   +1),
    (r"flash manufacturing pmi",                "SP_PMI_MFG",     +1),
    (r"flash services pmi",                     "SP_PMI_SVC",     +1),
    (r"trade balance",                          "TRADE_BAL",      +1),
]

# The releases the drift study actually cares about (scheduled, market-moving,
# consistently 8:30/10:00 ET style events with real consensus numbers).
CORE_EVENT_KEYS = {
    "NFP", "UNEMP_RATE", "AHE_MM", "CLAIMS", "ADP",
    "CPI_MM", "CORE_CPI_MM", "CPI_YY", "PPI_MM", "CORE_PPI_MM",
    "CORE_PCE_MM", "RETAIL_MM", "CORE_RETAIL_MM",
    "ISM_MFG", "ISM_SVC", "GDP_QQ", "FED_RATE", "JOLTS",
    "DURABLES_MM", "UOM_SENT", "CB_CONF", "PHILLY_FED", "EMPIRE_STATE",
}


def canonical_event(title: str) -> tuple[str, int]:
    """Map a raw FF event title to (event_key, usd_polarity)."""
    t = str(title).lower()
    for pat, key, pol in EVENT_MAP:
        if re.search(pat, t):
            return key, pol
    clean = re.sub(r"[^a-z0-9]+", "_", t).strip("_").upper()[:40]
    return clean, 0


_NUM_RE = re.compile(r"^\s*(<)?\s*(-?\d+(?:\.\d+)?)\s*([kmbt%])?\s*$", re.IGNORECASE)


def parse_ff_number(x) -> float:
    """Parse FF-style values: '3.2%', '227K', '-0.1%', '1.02M', '', 'nan'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in ("", "nan", "None", "-", "n/a", "N/A"):
        return np.nan
    m = _NUM_RE.match(s)
    if not m:
        return np.nan
    val = float(m.group(2))
    suf = (m.group(3) or "").lower()
    mult = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12, "%": 1.0, "": 1.0}[suf]
    return val * mult


# ---------------------------------------------------------------------------
# Source 1: weekly JSON feed
# ---------------------------------------------------------------------------

def fetch_week(week: str = "thisweek", timeout: int = 30) -> pd.DataFrame:
    """Fetch one week from the faireconomy feed and return normalized rows."""
    url = FEED_URL.format(week=week)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        rows = json.loads(r.read().decode("utf-8"))
    df = pd.DataFrame(rows)
    # Feed 'date' field is ISO8601 with offset, e.g. '2026-07-21T08:30:00-04:00'
    out = pd.DataFrame({
        "ts_utc": pd.to_datetime(df["date"], utc=True),
        "currency": df["country"].astype(str).str.upper(),
        "event": df["title"].astype(str),
        "impact": df["impact"].astype(str),
        "actual": df.get("actual", pd.Series(dtype=object)).map(parse_ff_number),
        "forecast": df.get("forecast", pd.Series(dtype=object)).map(parse_ff_number),
        "previous": df.get("previous", pd.Series(dtype=object)).map(parse_ff_number),
    })
    return _finalize(out)


# ---------------------------------------------------------------------------
# Source 2: historical CSV dumps (fuzzy column mapping)
# ---------------------------------------------------------------------------

_COL_ALIASES = {
    "ts": ["datetime", "date_time", "timestamp", "start", "dateutc", "date_utc"],
    "date": ["date", "day"],
    "time": ["time", "time_eastern", "time_et"],
    "currency": ["currency", "country", "cur"],
    "event": ["event", "title", "name", "event_name"],
    "impact": ["impact", "importance", "vol", "volatility"],
    "actual": ["actual", "actual_value"],
    "forecast": ["forecast", "consensus", "forecast_value", "expected"],
    "previous": ["previous", "prior", "previous_value"],
}


def _find_col(cols: list[str], names: list[str]) -> str | None:
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def load_history_csv(path: str | Path, source_tz: str = "America/New_York") -> pd.DataFrame:
    """
    Load a community FF-history CSV (any of the common column layouts) into the
    normalized schema. `source_tz` is the timezone the dump's timestamps are in —
    community FF scrapes are almost always US-Eastern ('America/New_York');
    pass 'UTC' if your dump says so.
    """
    path = Path(path)
    df = pd.read_csv(path)
    cols = list(df.columns)

    c_ts = _find_col(cols, _COL_ALIASES["ts"])
    if c_ts is not None:
        ts = pd.to_datetime(df[c_ts], errors="coerce")
    else:
        c_d = _find_col(cols, _COL_ALIASES["date"])
        c_t = _find_col(cols, _COL_ALIASES["time"])
        if c_d is None:
            raise ValueError(f"Can't find a timestamp column in {path.name}; columns={cols}")
        raw = df[c_d].astype(str) + (" " + df[c_t].astype(str) if c_t else "")
        ts = pd.to_datetime(raw, errors="coerce")

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
    ts_utc = ts.dt.tz_convert("UTC")

    def col(key, default=np.nan):
        c = _find_col(cols, _COL_ALIASES[key])
        return df[c] if c is not None else pd.Series(default, index=df.index)

    out = pd.DataFrame({
        "ts_utc": ts_utc,
        "currency": col("currency", "").astype(str).str.upper().str.strip(),
        "event": col("event", "").astype(str).str.strip(),
        "impact": col("impact", "").astype(str).str.strip().str.title(),
        "actual": col("actual").map(parse_ff_number),
        "forecast": col("forecast").map(parse_ff_number),
        "previous": col("previous").map(parse_ff_number),
    })
    out = out[out["ts_utc"].notna()]
    return _finalize(out)


# ---------------------------------------------------------------------------
# Normalization + surprise computation
# ---------------------------------------------------------------------------

def _finalize(out: pd.DataFrame) -> pd.DataFrame:
    out = out.drop_duplicates(subset=["ts_utc", "event"], keep="first")
    keys = out["event"].map(canonical_event)
    out = out.copy()
    out["event_key"] = keys.map(lambda kp: kp[0])
    out["usd_polarity"] = keys.map(lambda kp: kp[1])
    out["surprise"] = out["actual"] - out["forecast"]
    out = out.sort_values("ts_utc").reset_index(drop=True)
    out["surprise_z"] = _surprise_z(out)
    return out


def _surprise_z(df: pd.DataFrame, min_hist: int = 8) -> pd.Series:
    """Expanding, PAST-ONLY std per event_key. Uses shift(1) so the z of event i
    never includes event i itself (no lookahead)."""
    def per_key(s: pd.Series) -> pd.Series:
        std = s.expanding(min_periods=min_hist).std().shift(1)
        return s / std

    surprise = pd.to_numeric(df["surprise"], errors="coerce")
    z = surprise.groupby(df["event_key"]).transform(per_key)
    return z.replace([np.inf, -np.inf], np.nan)


def usd_events(df: pd.DataFrame, impacts=("High",), core_only: bool = True) -> pd.DataFrame:
    """Filter to the tradeable USD release set: has actual+forecast, wanted impact."""
    m = (
        (df["currency"] == "USD")
        & df["impact"].isin(impacts)
        & df["actual"].notna()
        & df["forecast"].notna()
    )
    if core_only:
        m &= df["event_key"].isin(CORE_EVENT_KEYS)
    return df[m].reset_index(drop=True)


def dedupe_simultaneous(df: pd.DataFrame, tol_minutes: int = 1) -> pd.DataFrame:
    """
    NFP day publishes NFP + unemployment rate + AHE at the same second. For the
    *price-path* side these are ONE market event. Keep every row (each has its own
    surprise) but tag rows that share a release minute with a `cluster_id` and a
    `cluster_primary` flag (the highest-|surprise_z| row wins primary).
    """
    df = df.sort_values("ts_utc").reset_index(drop=True)
    gap = df["ts_utc"].diff().dt.total_seconds().div(60).fillna(np.inf)
    df["cluster_id"] = (gap > tol_minutes).cumsum()
    absz = df["surprise_z"].abs().fillna(-1.0)
    df["cluster_primary"] = absz.eq(absz.groupby(df["cluster_id"]).transform("max"))
    # break ties (identical |z|) by keeping only the first flagged row per cluster
    dup = df["cluster_primary"] & df.duplicated(subset=["cluster_id", "cluster_primary"])
    df.loc[dup, "cluster_primary"] = False
    return df


if __name__ == "__main__":
    # Smoke test against the live weekly feed.
    wk = fetch_week("thisweek")
    print(f"thisweek feed: {len(wk)} rows, "
          f"{(wk['currency'] == 'USD').sum()} USD, "
          f"{((wk['currency'] == 'USD') & (wk['impact'] == 'High')).sum()} USD-High")
    print(wk[wk["currency"] == "USD"].head(12).to_string())
