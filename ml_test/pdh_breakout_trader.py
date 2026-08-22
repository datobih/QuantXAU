"""
Prior-Day-High breakout live trader (XAUUSD) — DEMO-FIRST, risk-controlled.

STRATEGY (as researched; see breakout-track memory) — TWO validated level streams:
  Daily setup (evaluated once per server day, from completed daily bars):
    - PDH  = prior day's HIGH                  (validated +$1.58/oz, t~3.4)
    - PWH  = prior COMPLETED ISO-week's HIGH   (validated +$2.55/oz, t=2.65)
    - trend filter = prior day's CLOSE > 10-day SMA of daily closes  (uptrend)
  Entry (long only; one PDH trade per day, one PWH trade per week):
    - PDH stream: gated DAILY — uptrend day and current price below the level.
    - PWH stream: gated ONCE PER WEEK, at the week's first day (uptrend at week
      start and week's opening price below the level) — the order then rests
      until it fills or the ISO week ends. (Fixed 2026-08-01: the original code
      re-gated the weekly stream every day, which changes 25% of the validated
      trade population — 25 unvalidated trades in, 9 validated trades out.)
    - a BUY-STOP pending order rests at each armed level (fills on the break).
      When PWH == PDH (~27% of weeks) both streams fire at the same level —
      that's 2x volume on those breaks, matching the backtest's separate streams.
    - short-session guard: a UTC day with fewer than 62 minutes of session left
      at arming time (the winter Sunday reopen) is skipped for the PDH stream —
      the backtest excludes day-blocks under 62 M1 bars.
  Exit — two modes:
    - "time" (DEFAULT — the VALIDATED spec): script closes the position 60 min
      after the fill; a broker-side disaster-SL (--protect, default $12) is the
      only other exit. Needs the script running for the time exit.
      WHY $12 (changed from an arbitrary $25 on 2026-07-24): tested bar-accurately
      across stop levels and three fill models. At $12 it triggers on 29/470
      backtest trades (6.2%), kills only 2 winners in 5 years, and lifts
      profit-per-drawdown from 8.98 (no stop) / 10.6 ($25) to 12.3-17.3 depending
      on fill quality — i.e. it survives even a worst-case fill at the breaching
      bar's low. Walk-forward: threshold chosen on 2021-23, out-of-sample 2024-26
      prof/DD 9.74 vs 7.07 unstopped (pessimistic fills). Bootstrap: improvement
      positive in 94% of 2000 resampled equity paths. This is a RISK improvement,
      not a return one — the P&L difference is not significant (paired t=1.09).
      There is deliberately NO take-profit: the payoff is right-tailed (MFE ~2x
      MAE) and a fixed TP caps exactly the trend-day winners that carry the
      strategy's profit. A TP/SL bracket variant was tested and removed.

DAY BOUNDARY (added 2026-08-01): the validated spec defines "day" in UTC.
Exness's server clock IS UTC, so the original code could read levels straight
from the terminal's D1 bars. Most brokers (FTMO included) run GMT+2/+3
(New-York-close convention), where the server day rolls 2-3 h before the UTC
day — and the strategy's profit bursts fire 00:00-01:00 UTC, right after the
UTC roll. Backtested on FTMO's own feed 2021-2026: UTC-day levels earn
+$1.42/trade (t=3.89) vs +$1.14 (t=3.00) with server-day levels.
Default --day-boundary utc therefore:
  - measures the server-UTC offset from live ticks (rounded to the hour).
    HARDENED 2026-08-01 after an adversarial audit: the FIRST tick seen by a
    fresh process only PRIMES the freshness gate and is never measured — a
    stale cached tick whose age is near a whole hour would otherwise latch a
    wrong offset (~1 in 6 restart timings; reproduced end-to-end). A
    measurement now requires a tick that ARRIVED while we were watching.
    Re-checked every poll => the broker's DST switch self-corrects.
  - the measured offset is part of the arming key: if the offset ever changes
    (DST switch, or a corrected mis-measurement), our pending orders are
    cancelled and the day re-armed at the corrected levels.
  - offsets outside {0, +2/+3-per-US-DST} are accepted but logged loudly —
    they usually mean the PC clock is wrong by a whole hour.
  - builds PDH / SMA10 / PWH from H1 bars bucketed into UTC calendar days
    (identical aggregation to the validated backtest)
  - keys one-trade-per-day / per-week to UTC dates
  - PDH orders expire at UTC midnight, PWH orders at the ISO-week end, via
    ORDER_TIME_SPECIFIED; falls back to GTC + cancel-on-roll if unsupported.
On a UTC broker the offset measures 0 and behavior matches the original spec.
--day-boundary server restores the old terminal-D1 behavior for A/B.

SAFETY (things the repo's other traders got wrong):
  - refuses to trade a REAL account unless --allow-real is passed
  - verifies terminal is connected and the symbol is tradable
  - respects broker min stop-distance and volume min/step
  - never double-places: checks existing orders/positions by magic number
  - pending order expires end-of-day (see DAY BOUNDARY) so nothing lingers
  - --dry-run computes and logs the decision without sending anything

THIS IS NOT VALIDATED OUT-OF-SAMPLE. It is a forward-test harness. Run it on a
DEMO account to collect honest out-of-sample trades before ever risking money.

Usage (MT5 terminal running + logged in, Algo Trading enabled):
  python pdh_breakout_trader.py --dry-run            # see today's decision
  python pdh_breakout_trader.py                      # trade on demo (validated spec)
  python pdh_breakout_trader.py --symbol XAUUSD      # broker-specific symbol name
                                                     # (default auto: XAUUSDm/XAUUSD)
  python pdh_breakout_trader.py --protect 25         # override the disaster stop
  python pdh_breakout_trader.py --allow-real         # ONLY after demo validation
For unattended running use run_pdh_trader.bat (auto-restart) and
install_pdh_task.bat (start at logon).
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

SYMBOL = "XAUUSDm"                     # resolved in connect(); --symbol auto tries these:
SYMBOL_CANDIDATES = ("XAUUSDm", "XAUUSD")   # Exness, FTMO/most brokers
VOLUME = 0.01
SMA_PERIOD = 10
PROTECT_USD = 12.0        # disaster-SL distance ($/oz) - see docstring for validation
HOLD_MIN = 60            # time-exit minutes for "time" mode
MAGIC = 770022            # daily PDH stream
MAGIC_W = 770023          # weekly PWH stream (validated separately: +$2.55/oz, t=2.65)
MAGICS = {MAGIC: "daily", MAGIC_W: "weekly"}
POLL_SEC = 20
LOG = Path("../data/processed/pdh_trader_log.csv")
LOG_FIELDS = ["ts_utc", "action", "server_day", "pdh", "sma10", "yday_close",
              "uptrend", "price", "order_price", "sl", "tp", "volume", "detail"]


def log(row: dict) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not LOG.exists()
        with LOG.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in LOG_FIELDS})
    except Exception as e:                          # logging must never kill the trader
        print(f"[log-error] {e}")
    try:
        print(f"[{row.get('ts_utc','')}] {row.get('action','')}: {row.get('detail','')}", flush=True)
    except Exception:
        pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _init(mt5, terminal_path):
    """mt5.initialize, optionally launching a specific terminal by path."""
    try:
        return mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()
    except Exception:
        return False


def try_reconnect(mt5, terminal_path=None) -> bool:
    """Best-effort re-establish the terminal connection. Never raises."""
    try:
        mt5.shutdown()
    except Exception:
        pass
    if not _init(mt5, terminal_path):
        return False
    t = mt5.terminal_info()
    if t is None or not t.connected:
        return False
    try:
        mt5.symbol_select(SYMBOL, True)
    except Exception:
        pass
    return True


def connect(mt5, allow_real: bool, require_trading: bool = True, terminal_path=None,
            symbol_arg: str = "auto"):
    if not _init(mt5, terminal_path):
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()} — is the terminal running?")
    term = mt5.terminal_info()
    if term is None or not term.connected:
        raise SystemExit("MT5 terminal not connected to broker.")
    if not term.trade_allowed:
        if require_trading:
            raise SystemExit("Algo Trading is disabled — enable the 'Algo Trading' toolbar button before live trading.")
        print("WARNING: Algo Trading is OFF — fine for --dry-run, but enable it before live trading.")
    ai = mt5.account_info()
    mode = {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(ai.trade_mode, "?")
    print(f"account {ai.login} [{mode}] {ai.balance} {ai.currency} on {ai.server}")
    if mode == "REAL" and not allow_real:
        mt5.shutdown()
        raise SystemExit("REAL account — refusing. Validate on demo first; pass --allow-real to override.")
    cands = [symbol_arg] if symbol_arg and symbol_arg != "auto" else list(SYMBOL_CANDIDATES)
    si = None
    for name in cands:
        s = mt5.symbol_info(name)
        if s is not None and mt5.symbol_select(name, True):
            globals()["SYMBOL"] = name
            si = s
            break
    if si is None:
        raise SystemExit(f"no tradable gold symbol found on this account (tried {cands}).")
    print(f"symbol: {SYMBOL} (digits {si.digits}, point {si.point}, "
          f"stops_level {si.trade_stops_level})")
    return si


def daily_setup(mt5, expect_day=None):
    """Daily + weekly levels from D1 bars. Returns dict or None.
    expect_day: the trading day (date) the caller is arming. If the feed's
    forming bar predates it, the terminal hasn't printed the new day yet —
    return None so the caller retries instead of evaluating YESTERDAY's
    levels as today's (2026-08-07 VPS incident: a poll 7s after midnight
    saw a stale feed, judged the old level "already touched", and silently
    burned the whole day).
    pwh = prior COMPLETED ISO-week's high; week_high = this week's high so far
    (completed days this week + the forming day) — the restart-proof
    one-trade-per-week guard."""
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, max(SMA_PERIOD + 3, 25))
    if rates is None or len(rates) < SMA_PERIOD + 2:
        return None
    if expect_day is not None:
        forming = datetime.fromtimestamp(int(rates[-1]["time"]), timezone.utc).date()
        if forming != expect_day:
            return None                         # new day's bar not printed yet — retry
    completed = rates[:-1]                      # drop the current forming day
    closes = [r["close"] for r in completed]
    pdh = float(completed[-1]["high"])          # yesterday's high
    yday_close = float(completed[-1]["close"])
    sma10 = sum(closes[-SMA_PERIOD:]) / SMA_PERIOD
    today_high = float(rates[-1]["high"])       # forming day's high so far

    def iso(ts):
        d = datetime.fromtimestamp(int(ts), timezone.utc).isocalendar()
        return (d[0], d[1])
    cur_week = iso(rates[-1]["time"])
    prev_weeks = [iso(r["time"]) for r in completed if iso(r["time"]) != cur_week]
    pwh = week_high = None
    pwh_gate = False
    if prev_weeks:
        last_prev = max(prev_weeks)
        pwh = max(float(r["high"]) for r in completed if iso(r["time"]) == last_prev)
        wk_rates = [r for r in rates if iso(r["time"]) == cur_week]
        week_high = max(float(r["high"]) for r in wk_rates) if wk_rates else today_high
        # weekly gate: evaluated ONCE at the week's first day (the validated
        # spec) — uptrend as of the week's open, and the week OPENED below the
        # level. Re-gating daily changes 25% of the trade population.
        if wk_rates:
            first = wk_rates[0]
            pre = [float(r["close"]) for r in completed if r["time"] < first["time"]]
            if len(pre) >= SMA_PERIOD + 1:
                sma_wk = sum(pre[-SMA_PERIOD:]) / SMA_PERIOD
                pwh_gate = pre[-1] > sma_wk and float(first["open"]) < pwh
    return {"pdh": pdh, "uptrend": yday_close > sma10, "sma10": sma10,
            "yday_close": yday_close, "today_high": today_high,
            "pwh": pwh, "week_high": week_high, "pwh_gate": pwh_gate}


def _us_dst(ts_utc: float) -> bool:
    """US daylight saving active at this UTC epoch? (2nd Sun of Mar 07:00 UTC
    -> 1st Sun of Nov 06:00 UTC — the switch NY-close broker clocks follow)."""
    d = datetime.fromtimestamp(ts_utc, timezone.utc)
    def nth_sunday(month, n):
        first = datetime(d.year, month, 1, tzinfo=timezone.utc)
        return datetime(d.year, month, 1 + (6 - first.weekday()) % 7 + 7 * (n - 1),
                        tzinfo=timezone.utc)
    return nth_sunday(3, 2).replace(hour=7) <= d < nth_sunday(11, 1).replace(hour=6)


def daily_setup_utc(mt5, offset_sec: int, expect_day=None):
    """Same levels as daily_setup, but from H1 bars bucketed into UTC calendar
    days (the validated backtest's aggregation) instead of the terminal's D1
    server days. offset_sec = server_clock - UTC, measured from live ticks.
    expect_day: see daily_setup — None is returned when the feed's latest
    UTC day predates the day being armed (midnight-roll race guard).
    Every UTC date with any bars counts as a day (incl. the short Sunday
    session) — exactly like the backtest's daily resample.

    Historical bars need PER-BAR offsets: an offset broker (GMT+2/+3) follows
    US DST, so bars from before the latest switch are off by an hour if shifted
    with today's offset (audit caught this: SMA10 drifted for ~2 weeks after
    each switch). offset 0 = fixed-UTC broker (Exness), no DST anywhere."""
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 800)
    if rates is None or len(rates) < (SMA_PERIOD + 3) * 20:
        return None
    if offset_sec == 0:
        bar_off = lambda ts: 0
    else:
        now_srv = int(rates[-1]["time"])
        winter = offset_sec - (3600 if _us_dst(now_srv - offset_sec) else 0)
        bar_off = lambda ts: winter + (3600 if _us_dst(ts - winter) else 0)
    days: dict = {}                    # date -> [high, close, last_ts, first_open, first_ts]
    for r in rates:
        rts = int(r["time"])
        ts = rts - bar_off(rts)
        d = datetime.fromtimestamp(ts, timezone.utc).date()
        rec = days.setdefault(d, [float(r["high"]), float(r["close"]), ts,
                                  float(r["open"]), ts])
        rec[0] = max(rec[0], float(r["high"]))
        if ts >= rec[2]:
            rec[1], rec[2] = float(r["close"]), ts
        if ts < rec[4]:
            rec[3], rec[4] = float(r["open"]), ts
    dl = sorted(days)
    if expect_day is not None and dl[-1] != expect_day:
        return None                             # new day's bar not printed yet — retry
    today, completed = dl[-1], dl[:-1]
    if len(completed) < SMA_PERIOD + 1:
        return None
    closes = [days[d][1] for d in completed]
    pdh = days[completed[-1]][0]
    yday_close = days[completed[-1]][1]
    sma10 = sum(closes[-SMA_PERIOD:]) / SMA_PERIOD
    today_high = days[today][0]
    cur_week = today.isocalendar()[:2]
    prev_weeks = sorted({d.isocalendar()[:2] for d in completed
                         if d.isocalendar()[:2] != cur_week})
    pwh = week_high = None
    pwh_gate = False
    if prev_weeks:
        last_prev = prev_weeks[-1]
        pwh = max(days[d][0] for d in completed if d.isocalendar()[:2] == last_prev)
        week_days = [d for d in dl if d.isocalendar()[:2] == cur_week]
        week_high = max(days[d][0] for d in week_days) if week_days else today_high
        # weekly gate: fixed at the week's FIRST day (validated spec — see
        # daily_setup): uptrend as of the week open + week opened below level
        if week_days:
            wf = week_days[0]
            pre = [days[d][1] for d in completed if d < wf]
            if len(pre) >= SMA_PERIOD + 1:
                sma_wk = sum(pre[-SMA_PERIOD:]) / SMA_PERIOD
                pwh_gate = pre[-1] > sma_wk and days[wf][3] < pwh
    return {"pdh": pdh, "uptrend": yday_close > sma10, "sma10": sma10,
            "yday_close": yday_close, "today_high": today_high,
            "pwh": pwh, "week_high": week_high, "pwh_gate": pwh_gate}


def cancel_stale_orders(mt5, offset_sec: int, cur_day: str, cur_week, force=False) -> None:
    """UTC mode: remove our pending orders that outlived their window — a PDH
    order from a previous UTC day, or a PWH order from a previous ISO week
    (the weekly order deliberately RESTS across day rolls). force=True removes
    all our pendings regardless of age — used when the measured offset changes,
    because orders placed under the old offset sit at suspect levels.
    Covers the GTC-expiry fallback and restarts; harmless when SPECIFIED
    expiry already culled them. A failed REMOVE is logged loudly."""
    for o in (mt5.orders_get(symbol=SYMBOL) or []):
        if o.magic not in MAGICS:
            continue
        od = datetime.fromtimestamp(int(o.time_setup) - offset_sec, timezone.utc)
        stale = force or (o.magic == MAGIC and od.strftime("%Y-%m-%d") != cur_day) \
                      or (o.magic == MAGIC_W and od.isocalendar()[:2] != cur_week)
        if stale:
            r = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
            rc = getattr(r, "retcode", None)
            ok = rc == mt5.TRADE_RETCODE_DONE
            log({"ts_utc": _now(), "action": "CANCEL-STALE" if ok else "CANCEL-FAIL",
                 "server_day": cur_day,
                 "detail": f"[{MAGICS[o.magic]}] {'removed' if ok else 'FAILED to remove'} "
                           f"order {o.ticket} from {od.date()} (retcode {rc})"
                           + ("" if ok else " — will retry next poll")})


def has_activity(mt5, magic: int) -> bool:
    for o in (mt5.orders_get(symbol=SYMBOL) or []):
        if o.magic == magic:
            return True
    for p in (mt5.positions_get(symbol=SYMBOL) or []):
        if p.magic == magic:
            return True
    return False


def place_buy_stop(mt5, si, pdh, price, sl_usd, dry,
                   magic=MAGIC, comment="PDH_break_uptrend", expiry=None):
    digits = si.digits
    stops_pts = si.trade_stops_level * si.point
    entry = round(pdh, digits)
    adjusted = False
    # broker requires the stop price to sit at least stops_level above current ask
    if entry <= price + stops_pts:
        entry = round(price + stops_pts + si.point, digits)
        adjusted = True
    if adjusted:
        log({"ts_utc": _now(), "action": "note",
             "detail": f"entry adjusted above PDH ({pdh:.3f} -> {entry:.3f}) "
                       f"to satisfy broker min stop distance"})
    sl = round(entry - sl_usd, digits)
    tp = 0.0                       # no take-profit: the exit is the 60-minute clock
    vol = max(si.volume_min, round(VOLUME / si.volume_step) * si.volume_step)
    # symbol filling_mode is a bitmask: 1=FOK, 2=IOC — RETURN is rejected on
    # market-execution symbols like Exness XAUUSDm, so pick from what's allowed
    filling = mt5.ORDER_FILLING_IOC if (si.filling_mode & 2) else mt5.ORDER_FILLING_FOK
    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": SYMBOL, "volume": vol,
        "type": mt5.ORDER_TYPE_BUY_STOP, "price": entry,
        "sl": sl, "tp": tp,
        "type_time": mt5.ORDER_TIME_DAY,          # auto-cancel unfilled at day end
        "type_filling": filling,
        "magic": magic, "comment": comment,
    }
    if expiry is not None:
        # UTC-boundary mode: the trading day ends at UTC midnight, which on
        # GMT+2/+3 servers is 2-3 h AFTER the broker's day-expiry would have
        # culled the order — right when the strategy's burst window fires.
        if si.expiration_mode & 4:                # SYMBOL_EXPIRATION_SPECIFIED
            req["type_time"] = mt5.ORDER_TIME_SPECIFIED
            req["expiration"] = int(expiry)       # server-clock epoch
        else:                                     # fallback: GTC + day-roll cancel
            req["type_time"] = mt5.ORDER_TIME_GTC
    if dry:
        return None, req
    res = mt5.order_send(req)
    return res, req


def close_position(mt5, si, pos):
    tick = mt5.symbol_info_tick(SYMBOL)
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": pos.volume,
           "type": mt5.ORDER_TYPE_SELL, "position": pos.ticket,
           "price": tick.bid, "deviation": 2000,      # $2 tolerance: the exit MUST fill
           "magic": pos.magic,                        # keep the STREAM's magic so the
           "type_filling": mt5.ORDER_FILLING_IOC,     # CLOSED log row is tagged correctly
           "comment": f"{MAGICS.get(pos.magic, 'PDH')}_time_exit"}
    return mt5.order_send(req)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protect", type=float, default=PROTECT_USD,
                    help="time mode: disaster-SL distance ($/oz). Default 12 is walk-forward "
                         "validated (see docstring); 25 was the earlier arbitrary value.")
    ap.add_argument("--hold", type=int, default=HOLD_MIN)
    ap.add_argument("--volume", type=float, default=VOLUME)
    ap.add_argument("--symbol", default="auto",
                    help="broker symbol name; 'auto' tries XAUUSDm then XAUUSD")
    ap.add_argument("--day-boundary", choices=("utc", "server"), default="utc",
                    help="'utc' (default) = validated spec: levels/day-keys on UTC days, "
                         "offset measured from live ticks. 'server' = old terminal-D1 behavior.")
    ap.add_argument("--allow-real", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true", help="evaluate once and exit (for cron/testing)")
    ap.add_argument("--terminal-path", default=None,
                    help="path to terminal64.exe — lets the trader relaunch MT5 itself on reconnect")
    a = ap.parse_args()
    globals()["VOLUME"] = a.volume

    import MetaTrader5 as mt5
    si = connect(mt5, a.allow_real, require_trading=not a.dry_run,
                 terminal_path=a.terminal_path, symbol_arg=a.symbol)
    print(f"exit: {a.hold}min time-exit (VALIDATED spec) | disaster-SL -${a.protect} "
          f"| vol {a.volume} | day-boundary {a.day_boundary} | dry_run={a.dry_run}\n")

    import datetime as _dt
    armed_day = None
    armed_off = None                                # offset the day was armed under
    offset_sec = 0 if a.day_boundary == "server" else None   # server_clock - UTC
    last_tick_time = 0
    none_ticks = 0                                  # consecutive polls with no tick
    once_polls = 0                                  # --once: polls spent waiting for offset
    arm_tries = 0                                   # bounded retry of a failed arming
    skip_key = None                                 # throttles the setup-failure log
    fill_time: dict = {}                            # position ticket -> fill epoch (time exit)
    known_pos: set = set()                          # tickets already logged as FILLED
    closed_logged: set = set()                      # position ids already logged as CLOSED
    hist_from = _dt.datetime.now(timezone.utc) - _dt.timedelta(days=1)
    last_beat = 0.0
    errors = 0

    def run():
        """The trading loop. Raises on a dead connection so the supervisor can
        reconnect; state (armed_day etc.) is preserved across restarts."""
        nonlocal armed_day, armed_off, offset_sec, last_tick_time, none_ticks, \
                 once_polls, arm_tries, skip_key, last_beat, errors, hist_from
        while True:
            term = mt5.terminal_info()
            if term is None or not term.connected:
                raise ConnectionError("terminal not connected")
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None or tick.time == 0:
                # a dead symbol feed with a "connected" terminal would spin
                # silently forever — escalate so the supervisor reconnects
                none_ticks += 1
                if none_ticks * POLL_SEC >= 1800:
                    none_ticks = 0
                    raise ConnectionError("no tick data for 30min (symbol feed dead?)")
                time.sleep(POLL_SEC); continue
            none_ticks = 0

            # --- server-UTC offset (utc mode): trust only FLOWING ticks ---
            # A fresh tick is within seconds of the machine clock, so rounding
            # to the hour is exact. The FIRST tick a process sees only PRIMES
            # the gate and is never measured: a terminal's cached pre-sync tick
            # whose age is near a whole hour would otherwise latch a wrong
            # offset (~1 in 6 restart timings; reproduced end-to-end in the
            # 2026-08-01 audit). Re-measured every poll => the broker's DST
            # switch (over a closed weekend) self-corrects at the reopen.
            flowing = last_tick_time != 0 and tick.time != last_tick_time
            if a.day_boundary == "utc" and flowing:
                cand = tick.time - time.time()
                hrs = round(cand / 3600.0)
                if -12 <= hrs <= 14 and abs(cand - hrs * 3600) <= 300:
                    if offset_sec != hrs * 3600:
                        expect = 3 if _us_dst(time.time()) else 2
                        warn = "" if hrs in (0, expect) else \
                            f" — UNUSUAL (expected 0 or +{expect}); check the PC clock!"
                        log({"ts_utc": _now(), "action": "note",
                             "detail": f"server-UTC offset measured: {hrs:+d}h{warn}"})
                        offset_sec = hrs * 3600
            last_tick_time = tick.time
            if offset_sec is None:                  # not yet measured
                if a.once:
                    # first tick only primes, so give a live market a few polls
                    # to deliver a measurable second tick before giving up
                    if once_polls < 4:
                        once_polls += 1
                        time.sleep(POLL_SEC); continue
                    log({"ts_utc": _now(), "action": "skip",
                         "detail": "no fresh ticks — cannot measure server-UTC "
                                   "offset (market closed?); nothing to arm"})
                    return
                if tick.time - last_beat >= 21600 or last_beat == 0.0:
                    last_beat = tick.time
                    log({"ts_utc": _now(), "action": "ALIVE",
                         "detail": "waiting to measure server-UTC offset "
                                   f"(ticks {'FLOWING — if this persists, the PC clock is off by minutes; fix it' if flowing else 'stalled — market closed?'})"})
                time.sleep(POLL_SEC); continue

            now = datetime.fromtimestamp(tick.time - offset_sec, timezone.utc)  # UTC clock
            day = now.strftime("%Y-%m-%d")          # the TRADING day (UTC in utc mode)
            price = tick.ask
            nowiso = _now()

            # heartbeat every 6h — proves liveness even on multi-day no-trade stretches
            if tick.time - last_beat >= 21600:
                last_beat = tick.time
                log({"ts_utc": nowiso, "action": "ALIVE", "server_day": day,
                     "price": round(price, 3),
                     "detail": f"running (boundary={a.day_boundary}, "
                               f"offset={offset_sec // 3600:+d}h)"})

            # --- once per new trading day: evaluate setup & place order ---
            # The arming key is (day, offset): a corrected offset re-arms the
            # day at the corrected levels (audit 2026-08-01). armed_day is set
            # only AFTER a successful evaluation — a transient failure (H1
            # history not yet synced, order rejected) retries next poll,
            # bounded by arm_tries so a permanent failure cannot spam forever.
            if day != armed_day or (a.day_boundary == "utc" and offset_sec != armed_off):
                if (day, offset_sec) != skip_key:
                    arm_tries = 0                   # new day (or new offset): fresh retries
                if a.day_boundary == "utc" and not a.dry_run:
                    # offset changed => orders placed under the old offset rest
                    # at suspect levels (incl. a same-ISO-week PWH order that
                    # date-based cleanup would keep): remove them all, once
                    force = armed_off is not None and offset_sec != armed_off \
                            and arm_tries == 0
                    if force:
                        log({"ts_utc": nowiso, "action": "note", "server_day": day,
                             "detail": f"offset changed {armed_off//3600 if armed_off is not None else '?'}h -> "
                                       f"{offset_sec//3600}h — cancelling orders, re-arming"})
                    cancel_stale_orders(mt5, offset_sec, day, now.isocalendar()[:2],
                                        force=force)
                s = daily_setup_utc(mt5, offset_sec, expect_day=now.date()) \
                    if a.day_boundary == "utc" else daily_setup(mt5, expect_day=now.date())
                if s is None:
                    if (day, offset_sec) != skip_key:      # log once, retry silently
                        skip_key = (day, offset_sec)
                        log({"ts_utc": nowiso, "action": "skip", "server_day": day,
                             "detail": "setup not ready (history missing, or the new "
                                       "day's first bar not printed yet) — retrying "
                                       "every poll"})
                else:
                    skip_key = (day, offset_sec)
                    sl_dist = a.protect
                    # PDH order lives until UTC midnight; PWH until the ISO week
                    # ends (it rests across day rolls — the weekly gate was
                    # already fixed at the week's first day). server mode keeps
                    # broker day-expiry.
                    eff = tick.time - offset_sec
                    next_mid = (eff // 86400 + 1) * 86400
                    expiry_d = expiry_w = None
                    if a.day_boundary == "utc":
                        expiry_d = int(next_mid + offset_sec)
                        expiry_w = int((eff // 86400 + (7 - now.weekday())) * 86400
                                       + offset_sec)
                    base = {"ts_utc": nowiso, "server_day": day, "pdh": round(s["pdh"], 3),
                            "sma10": round(s["sma10"], 3), "yday_close": round(s["yday_close"], 3),
                            "uptrend": s["uptrend"], "price": round(price, 3)}

                    def arm(level, magic, tag, high_guard, guard_name,
                            gate_ok, gate_msg, expiry_ts):
                        """Returns False only on a retryable order failure.
                        quiet on retries: the no-trade/skip rows were already
                        logged on the first attempt."""
                        quiet = arm_tries > 0
                        if not gate_ok:
                            if not quiet:
                                log({**base, "action": "no-trade", "detail": f"{tag}: {gate_msg}"})
                        elif price >= level:
                            if not quiet:
                                log({**base, "action": "no-trade",
                                     "detail": f"{tag}: price already >= level {level:.3f}"})
                        elif high_guard is not None and high_guard >= level:
                            if not quiet:
                                log({**base, "action": "no-trade",
                                     "detail": f"{tag}: level {level:.3f} already touched "
                                               f"({guard_name} high {high_guard:.3f})"})
                        elif has_activity(mt5, magic):
                            if not quiet:
                                log({**base, "action": "skip", "detail": f"{tag}: order/position already exists"})
                        else:
                            res, req = place_buy_stop(mt5, si, level, price, sl_dist, a.dry_run,
                                                      magic=magic, comment=f"{tag}_break_uptrend",
                                                      expiry=expiry_ts)
                            if a.dry_run:
                                log({**base, "action": "DRY-would-place", "order_price": req["price"],
                                     "sl": req["sl"], "tp": req["tp"], "volume": req["volume"],
                                     "detail": f"{tag}: buy-stop at {level:.3f}"})
                            elif res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                                log({**base, "action": "PLACED", "order_price": req["price"],
                                     "sl": req["sl"], "tp": req["tp"], "volume": req["volume"],
                                     "detail": f"{tag}: buy-stop ticket {res.order}"})
                            else:
                                log({**base, "action": "ORDER-FAIL", "order_price": req["price"],
                                     "detail": f"{tag}: retcode {getattr(res,'retcode','?')} "
                                               f"{getattr(res,'comment','')} (try {arm_tries + 1})"})
                                return False
                        return True

                    ok = True
                    # short-session guard: the backtest excludes day-blocks with
                    # under 62 M1 bars — live equivalent: skip the PDH stream
                    # when under 62 minutes of UTC day remain at arming time
                    # (the winter Sunday reopen). Weekly stream is unaffected.
                    if a.day_boundary == "utc" and next_mid - eff < 62 * 60:
                        if arm_tries == 0:
                            log({**base, "action": "no-trade",
                                 "detail": f"PDH: only {(next_mid - eff) // 60}min of UTC day "
                                           "left (short session) — backtest excludes these"})
                    else:
                        ok &= arm(s["pdh"], MAGIC, "PDH", s["today_high"], "today",
                                  s["uptrend"], "not an uptrend day", expiry_d)
                    if s["pwh"] is not None:
                        ok &= arm(s["pwh"], MAGIC_W, "PWH", s["week_high"], "week",
                                  s.get("pwh_gate", False),
                                  "week-start gate failed (not uptrend at week open, "
                                  "or week opened above level)", expiry_w)
                    if ok or arm_tries >= 15:
                        if not ok:
                            log({"ts_utc": nowiso, "action": "ERROR", "server_day": day,
                                 "detail": "giving up on today's failed order after "
                                           f"{arm_tries + 1} tries"})
                        armed_day, armed_off = day, offset_sec
                        arm_tries = 0
                    else:
                        arm_tries += 1              # retry next poll

            # --- fill detection: log the moment a buy-stop becomes a position ---
            for p in (mt5.positions_get(symbol=SYMBOL) or []):
                if p.magic in MAGICS and p.ticket not in known_pos:
                    known_pos.add(p.ticket)
                    log({"ts_utc": nowiso, "action": "FILLED", "server_day": day,
                         "price": p.price_open, "volume": p.volume,
                         "detail": f"[{MAGICS[p.magic]}] position {p.ticket} sl={p.sl} tp={p.tp}"})
                    # The pending order's SL (order_price - protect) IS the validated
                    # spec: the backtest's entry is the level, so its stop is
                    # level - protect. Re-anchoring to fill-protect (removed
                    # 2026-08-22) silently TIGHTENED the stop by the entry slippage
                    # on fast-burst fills (FTMO real fills slipped $0.1-$6.4; the
                    # 2026-08-19 weekly trade was stopped by the re-anchored SL in
                    # a flush the spec stop survives: -$147 live vs +$279 on spec —
                    # a $426 swing). Only attach an SL if the position somehow has
                    # none; log big slips for the live-vs-backtest ledger.
                    if not a.dry_run and p.sl == 0.0:
                        want = round(p.price_open - a.protect, si.digits)
                        r = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP,
                                            "symbol": SYMBOL, "position": p.ticket,
                                            "sl": want, "tp": p.tp})
                        log({"ts_utc": nowiso, "action": "SL-ADJUST", "server_day": day,
                             "sl": want,
                             "detail": f"[{MAGICS[p.magic]}] position had NO SL — set to "
                                       f"fill-{a.protect} (retcode {getattr(r,'retcode','?')})"})
                    slip = p.price_open - (p.sl + a.protect) if p.sl > 0 else 0.0
                    if slip > 1.0:
                        log({"ts_utc": nowiso, "action": "note", "server_day": day,
                             "detail": f"[{MAGICS[p.magic]}] entry slipped ${slip:.2f} above "
                                       f"the level; SL stays at level-{a.protect} (spec)"})

            # --- close detection: log final P&L of finished trades (SL/TP/time) ---
            # sliding 2-day window (a frozen start would re-scan months of
            # history every poll); upper bound padded because deal stamps are
            # SERVER time, which runs ahead of UTC on GMT+2/+3 brokers
            hist_from = max(hist_from, _dt.datetime.now(timezone.utc) - _dt.timedelta(days=2))
            deals = mt5.history_deals_get(
                hist_from, datetime.now(timezone.utc) + _dt.timedelta(hours=13)) or []
            for dl in deals:
                if dl.magic in MAGICS and dl.entry == mt5.DEAL_ENTRY_OUT \
                        and dl.position_id not in closed_logged:
                    closed_logged.add(dl.position_id)
                    log({"ts_utc": nowiso, "action": "CLOSED", "server_day": day,
                         "price": dl.price, "volume": dl.volume,
                         "detail": f"[{MAGICS[dl.magic]}] position {dl.position_id} "
                                   f"pnl ${dl.profit:+.2f} (reason {dl.reason})"})

            # --- time-exit management ---
            if not a.dry_run:
                for p in (mt5.positions_get(symbol=SYMBOL) or []):
                    if p.magic not in MAGICS:
                        continue
                    fill_time.setdefault(p.ticket, p.time)
                    if tick.time - fill_time[p.ticket] >= a.hold * 60:
                        r = close_position(mt5, si, p)
                        log({"ts_utc": nowiso, "action": "TIME-EXIT", "server_day": day,
                             "price": round(tick.bid, 3),
                             "detail": f"[{MAGICS[p.magic]}] closed {p.ticket} after "
                                       f"{a.hold}min, retcode {getattr(r,'retcode','?')}"})

            errors = 0                                # a healthy pass resets the backoff
            if a.once:
                return
            time.sleep(POLL_SEC)

    # --- supervisor: never dies except on Ctrl-C; reconnects with backoff ---
    log({"ts_utc": _now(), "action": "START", "server_day": "",
         "detail": f"{SYMBOL} exit={a.hold}min sl=-${a.protect} vol={a.volume} "
                   f"boundary={a.day_boundary} {'DRY-RUN' if a.dry_run else 'LIVE'}"})
    try:
        while True:
            try:
                run()
                break                                 # returns only in --once mode
            except KeyboardInterrupt:
                raise
            except Exception as e:
                errors += 1
                log({"ts_utc": _now(), "action": "ERROR",
                     "detail": f"{type(e).__name__}: {e} (#{errors}) — reconnecting"})
                ok = try_reconnect(mt5, a.terminal_path)
                fresh = mt5.symbol_info(SYMBOL) if ok else None
                if fresh is not None:
                    si = fresh
                    log({"ts_utc": _now(), "action": "RECONNECT", "detail": "connection restored"})
                time.sleep(min(120, 10 * errors))     # exponential-ish backoff, capped 2min
    except KeyboardInterrupt:
        print("\nstopped by user (broker-side orders + SL/TP remain active).")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
