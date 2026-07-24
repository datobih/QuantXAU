"""
Measure the broker server clock offset directly from the live MT5 terminal.

The last tick's `time` is stamped by the *server* clock; comparing it to this
machine's UTC clock (which Windows syncs via NTP) gives the server-vs-UTC
offset to within seconds whenever the market is trading. Rounded to the nearest
half hour that IS the server timezone — no volatility inference needed.

Caveats handled:
- Weekends/holidays: the last tick is hours old, which would fake a huge
  offset. Gold ticks near-continuously during the week, so we require the tick
  to be < 120s stale after removing the candidate offset; otherwise we say so
  instead of reporting garbage.
- DST: run this once in summer and once in winter (or after any suspicious
  --check-alignment result). Exness is expected to be GMT+0 year-round, but
  that expectation is exactly what this script exists to test.

Run (MT5 terminal must be running and logged in):
    python -m news_drift.check_server_clock            # from ml_test/
    python -m news_drift.check_server_clock XAUUSDz    # other symbol variant
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

CANDIDATE_SYMBOLS = ["XAUUSDm", "XAUUSD", "XAUUSDz", "XAUUSDc", "XAUUSDb"]


def main() -> int:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package not installed: python -m pip install MetaTrader5")
        return 1

    if not mt5.initialize():
        print(f"mt5.initialize() failed: {mt5.last_error()}\n"
              "Is the MT5 terminal running and logged in?")
        return 1
    try:
        term = mt5.terminal_info()
        if term is not None and not term.connected:
            print("MT5 terminal is running but NOT connected to the broker "
                  "(log in / check connection), so tick times are unavailable.")
            return 1

        symbols = sys.argv[1:] or CANDIDATE_SYMBOLS
        tick, sym = None, None
        for cand in symbols:
            if mt5.symbol_select(cand, True):
                t = mt5.symbol_info_tick(cand)
                if t is not None and t.time > 0:
                    tick, sym = t, cand
                    break
        if tick is None:
            print(f"no live tick for any of {symbols} — terminal connected but "
                  "no quote data (market closed for days, or symbols hidden).")
            return 1

        now_utc = datetime.now(timezone.utc).timestamp()
        raw = tick.time - now_utc                       # server_clock - utc, plus staleness
        offset_h = round(raw / 1800) / 2                # nearest 0.5h
        residual = raw - offset_h * 3600                # tick staleness after removing offset

        print(f"symbol            : {sym}")
        print(f"last tick (server): {datetime.fromtimestamp(tick.time, timezone.utc):%Y-%m-%d %H:%M:%S}")
        print(f"now (UTC)         : {datetime.fromtimestamp(now_utc, timezone.utc):%Y-%m-%d %H:%M:%S}")
        print(f"raw difference    : {raw:+.0f}s  ->  server = UTC{offset_h:+g}h "
              f"(tick {abs(residual):.0f}s stale)")

        if abs(residual) > 120:
            print("\nUNRELIABLE: last tick is minutes old (market closed?). "
                  "Re-run while gold is actively trading.")
            return 1
        rule = "utc" if offset_h == 0 else ("et+7" if offset_h in (2, 3) else None)
        print(f"\nserver clock = UTC{offset_h:+g}h  ->  news_drift align rule: "
              f"{rule or 'NONE OF THE BUILT-INS — add one in align.py'}")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
