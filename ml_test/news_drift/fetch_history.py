"""
Fetch historical ForexFactory calendar weeks directly from forexfactory.com.

Each calendar page embeds its full week as JSON in
`window.calendarComponentStates[N] = { days: [...] }`; every event carries a
unix `dateline` (UTC), currency, impactName, and actual/forecast/previous
strings — exactly what the study needs, no HTML parsing games.

The fetcher is polite (one request per ~2s, plain browser UA, retries with
backoff) and resumable: already-fetched week starts found in the output CSV are
skipped, so re-running after an interruption continues where it stopped.

Output CSV columns (datetime is UTC) are chosen so that
    ff_calendar.load_history_csv(path, source_tz="UTC")
consumes the file directly.

Run (from ml_test/):
    python -m news_drift.fetch_history --start 2025-01-06 --out ../data/raw/ff_history.csv
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]


def week_url(d: date) -> str:
    return f"https://www.forexfactory.com/calendar?week={MONTHS[d.month - 1]}{d.day}.{d.year}"


def _extract_days(html: str) -> list[dict]:
    """Pull the first non-empty `days: [...]` array out of the page via bracket matching."""
    start = 0
    while True:
        i = html.find("window.calendarComponentStates[", start)
        if i < 0:
            return []
        j = html.find("days: [", i)
        if j < 0:
            return []
        depth, k = 0, j + 6
        for k in range(j + 6, len(html)):
            if html[k] == "[":
                depth += 1
            elif html[k] == "]":
                depth -= 1
                if depth == 0:
                    break
        try:
            days = json.loads(html[j + 6:k + 1])
        except json.JSONDecodeError:
            days = []
        if days:
            return days
        start = k


def fetch_week_page(d: date, retries: int = 4, timeout: int = 40) -> list[dict]:
    """Fetch one calendar week; returns flat event rows."""
    url = week_url(d)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                html = r.read().decode("utf-8", "replace")
            days = _extract_days(html)
            rows = []
            for day in days:
                for e in day.get("events", []):
                    if not e.get("dateline"):
                        continue
                    rows.append({
                        "datetime": pd.Timestamp(int(e["dateline"]), unit="s", tz="UTC"),
                        "currency": e.get("currency", ""),
                        "event": e.get("name", ""),
                        "impact": str(e.get("impactName", "")).title(),
                        "actual": e.get("actual", ""),
                        "forecast": e.get("forecast", ""),
                        "previous": e.get("previous", ""),
                        "week_start": d.isoformat(),
                    })
            return rows
        except Exception as err:  # noqa: BLE001 - network fetch, retry anything
            last_err = err
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} tries: {url}: {last_err}")


def fetch_range(start: date, end: date, out: Path, delay: float = 2.0) -> pd.DataFrame:
    """Fetch every week in [start, end], appending to `out` (resumable)."""
    start -= timedelta(days=start.weekday())  # snap to Monday
    done: set[str] = set()
    frames = []
    if out.exists():
        prev = pd.read_csv(out)
        done = set(prev["week_start"].astype(str))
        frames.append(prev)
        print(f"resume: {len(prev)} rows already fetched ({len(done)} weeks)")

    d = start
    n_weeks = 0
    while d <= end:
        if d.isoformat() not in done:
            rows = fetch_week_page(d)
            frames.append(pd.DataFrame(rows))
            n_weeks += 1
            print(f"  {d}  {len(rows):3d} events")
            df = pd.concat(frames, ignore_index=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)   # checkpoint every week
            time.sleep(delay)
        d += timedelta(days=7)

    df = pd.concat(frames, ignore_index=True)
    print(f"done: {len(df)} rows, {n_weeks} new weeks -> {out}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (snapped back to Monday)")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out", default="../data/raw/ff_history.csv")
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()
    fetch_range(date.fromisoformat(args.start), date.fromisoformat(args.end),
                Path(args.out), delay=args.delay)


if __name__ == "__main__":
    main()
