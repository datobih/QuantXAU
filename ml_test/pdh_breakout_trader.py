"""
Prior-Day-High breakout live trader (XAUUSD) — DEMO-FIRST, risk-controlled.

STRATEGY (as researched; see breakout-track memory) — TWO validated level streams:
  Daily setup (evaluated once per server day, from completed daily bars):
    - PDH  = prior day's HIGH                  (validated +$1.58/oz, t~3.4)
    - PWH  = prior COMPLETED ISO-week's HIGH   (validated +$2.55/oz, t=2.65)
    - trend filter = prior day's CLOSE > 10-day SMA of daily closes  (uptrend)
  Entry (long only; one PDH trade per day, one PWH trade per week):
    - only on uptrend days, and only if current price is BELOW the level
    - a BUY-STOP pending order rests at each armed level (fills on the break);
      the weekly order re-arms daily (day-expiry) until the week's level trades.
      When PWH == PDH (~27% of weeks) both streams fire at the same level —
      that's 2x volume on those breaks, matching the backtest's separate streams.
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

SAFETY (things the repo's other traders got wrong):
  - refuses to trade a REAL account unless --allow-real is passed
  - verifies terminal is connected and the symbol is tradable
  - respects broker min stop-distance and volume min/step
  - never double-places: checks existing orders/positions by magic number
  - pending order expires end-of-day (ORDER_TIME_DAY) so nothing lingers
  - --dry-run computes and logs the decision without sending anything

THIS IS NOT VALIDATED OUT-OF-SAMPLE. It is a forward-test harness. Run it on a
DEMO account to collect honest out-of-sample trades before ever risking money.

Usage (MT5 terminal running + logged in, Algo Trading enabled):
  python pdh_breakout_trader.py --dry-run            # see today's decision
  python pdh_breakout_trader.py                      # trade on demo (validated spec)
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

SYMBOL = "XAUUSDm"
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


def connect(mt5, allow_real: bool, require_trading: bool = True, terminal_path=None):
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
    si = mt5.symbol_info(SYMBOL)
    if si is None or not mt5.symbol_select(SYMBOL, True):
        raise SystemExit(f"symbol {SYMBOL} unavailable.")
    return si


def daily_setup(mt5):
    """Daily + weekly levels from D1 bars. Returns dict or None.
    pwh = prior COMPLETED ISO-week's high; week_high = this week's high so far
    (completed days this week + the forming day) — the restart-proof
    one-trade-per-week guard."""
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, max(SMA_PERIOD + 3, 25))
    if rates is None or len(rates) < SMA_PERIOD + 2:
        return None
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
    if prev_weeks:
        last_prev = max(prev_weeks)
        pwh = max(float(r["high"]) for r in completed if iso(r["time"]) == last_prev)
        wk_highs = [float(r["high"]) for r in rates if iso(r["time"]) == cur_week]
        week_high = max(wk_highs) if wk_highs else today_high
    return {"pdh": pdh, "uptrend": yday_close > sma10, "sma10": sma10,
            "yday_close": yday_close, "today_high": today_high,
            "pwh": pwh, "week_high": week_high}


def has_activity(mt5, magic: int) -> bool:
    for o in (mt5.orders_get(symbol=SYMBOL) or []):
        if o.magic == magic:
            return True
    for p in (mt5.positions_get(symbol=SYMBOL) or []):
        if p.magic == magic:
            return True
    return False


def place_buy_stop(mt5, si, pdh, price, sl_usd, dry,
                   magic=MAGIC, comment="PDH_break_uptrend"):
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
    ap.add_argument("--allow-real", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true", help="evaluate once and exit (for cron/testing)")
    ap.add_argument("--terminal-path", default=None,
                    help="path to terminal64.exe — lets the trader relaunch MT5 itself on reconnect")
    a = ap.parse_args()
    globals()["VOLUME"] = a.volume

    import MetaTrader5 as mt5
    si = connect(mt5, a.allow_real, require_trading=not a.dry_run, terminal_path=a.terminal_path)
    print(f"exit: {a.hold}min time-exit (VALIDATED spec) | disaster-SL -${a.protect} "
          f"| vol {a.volume} | dry_run={a.dry_run}\n")

    import datetime as _dt
    armed_day = None
    fill_time: dict = {}                            # position ticket -> fill epoch (time exit)
    known_pos: set = set()                          # tickets already logged as FILLED
    closed_logged: set = set()                      # position ids already logged as CLOSED
    hist_from = _dt.datetime.now(timezone.utc) - _dt.timedelta(days=1)
    last_beat = 0.0
    errors = 0

    def run():
        """The trading loop. Raises on a dead connection so the supervisor can
        reconnect; state (armed_day etc.) is preserved across restarts."""
        nonlocal armed_day, last_beat, errors
        while True:
            term = mt5.terminal_info()
            if term is None or not term.connected:
                raise ConnectionError("terminal not connected")
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None or tick.time == 0:
                time.sleep(POLL_SEC); continue
            now = datetime.fromtimestamp(tick.time, timezone.utc)  # server clock (UTC for Exness)
            day = now.strftime("%Y-%m-%d")
            price = tick.ask
            nowiso = _now()

            # heartbeat every 6h — proves liveness even on multi-day no-trade stretches
            if tick.time - last_beat >= 21600:
                last_beat = tick.time
                log({"ts_utc": nowiso, "action": "ALIVE", "server_day": day,
                     "price": round(price, 3), "detail": "running"})

            # --- once per new server day: evaluate setup & place order ---
            # (armed_day is set only AFTER evaluation completes, so a disconnect
            #  mid-evaluation retries after reconnect instead of skipping the day)
            if day != armed_day:
                s = daily_setup(mt5)
                if s is None:
                    log({"ts_utc": nowiso, "action": "skip", "server_day": day,
                         "detail": "insufficient daily history"})
                else:
                    sl_dist = a.protect
                    base = {"ts_utc": nowiso, "server_day": day, "pdh": round(s["pdh"], 3),
                            "sma10": round(s["sma10"], 3), "yday_close": round(s["yday_close"], 3),
                            "uptrend": s["uptrend"], "price": round(price, 3)}

                    def arm(level, magic, tag, high_guard, guard_name):
                        if not s["uptrend"]:
                            log({**base, "action": "no-trade", "detail": f"{tag}: not an uptrend day"})
                        elif price >= level:
                            log({**base, "action": "no-trade",
                                 "detail": f"{tag}: price already >= level {level:.3f}"})
                        elif high_guard is not None and high_guard >= level:
                            log({**base, "action": "no-trade",
                                 "detail": f"{tag}: level {level:.3f} already touched "
                                           f"({guard_name} high {high_guard:.3f})"})
                        elif has_activity(mt5, magic):
                            log({**base, "action": "skip", "detail": f"{tag}: order/position already exists"})
                        else:
                            res, req = place_buy_stop(mt5, si, level, price, sl_dist, a.dry_run,
                                                      magic=magic, comment=f"{tag}_break_uptrend")
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
                                               f"{getattr(res,'comment','')}"})

                    arm(s["pdh"], MAGIC, "PDH", s["today_high"], "today")
                    if s["pwh"] is not None:
                        arm(s["pwh"], MAGIC_W, "PWH", s["week_high"], "week")
                armed_day = day

            # --- fill detection: log the moment a buy-stop becomes a position ---
            for p in (mt5.positions_get(symbol=SYMBOL) or []):
                if p.magic in MAGICS and p.ticket not in known_pos:
                    known_pos.add(p.ticket)
                    log({"ts_utc": nowiso, "action": "FILLED", "server_day": day,
                         "price": p.price_open, "volume": p.volume,
                         "detail": f"[{MAGICS[p.magic]}] position {p.ticket} sl={p.sl} tp={p.tp}"})
                    # re-anchor the disaster SL to the ACTUAL fill (backtest measures
                    # the stop from the fill, not the order price; matters on gap fills)
                    if not a.dry_run:
                        want = round(p.price_open - a.protect, si.digits)
                        if p.sl < want - si.point:      # only ever tightens
                            r = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP,
                                                "symbol": SYMBOL, "position": p.ticket,
                                                "sl": want, "tp": p.tp})
                            log({"ts_utc": nowiso, "action": "SL-ADJUST", "server_day": day,
                                 "sl": want,
                                 "detail": f"[{MAGICS[p.magic]}] re-anchored SL to fill-"
                                           f"{a.protect} (retcode {getattr(r,'retcode','?')})"})

            # --- close detection: log final P&L of finished trades (SL/TP/time) ---
            deals = mt5.history_deals_get(hist_from, datetime.now(timezone.utc)) or []
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
                   f"{'DRY-RUN' if a.dry_run else 'LIVE'}"})
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
