"""
Release-impulse demo trader — forward-validates the news-spike chase LIVE.

FROZEN SPEC (registered 2026-07-22, from 17-day tick study; n=7 news trades
in-sample, so this is a candidate under test, not a proven edge):
    - Armed only around scheduled high-impact USD releases (ForexFactory feed).
    - Trigger: |mid move| >= $5.00 within a trailing 10s window, between the
      release time and release +120s. First trigger per event only.
    - Enter MARKET in the impulse direction with a broker-side SL $10/oz away.
    - Hard time exit after 30 seconds (SL stays as disaster backstop).
    - Volume 0.01 lot default.

It logs REQUESTED vs FILLED prices on entry and exit — measuring real news
slippage is the primary purpose of the demo phase. Log:
data/processed/impulse_trades.csv

SAFETY: refuses to run on a REAL account unless --allow-real is passed.

Usage (from ml_test/, MT5 terminal running & logged in):
    python -m news_drift.impulse_trader --list          # show upcoming armed events
    python -m news_drift.impulse_trader                 # wait for events and trade them
    python -m news_drift.impulse_trader --test-order    # verify order path now (open+close 0.01)
    python -m news_drift.impulse_trader --arm-now 120   # arm trigger for next N sec (mechanics test)
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

SYMBOL = "XAUUSDm"
TRIGGER_USD = 5.0        # impulse size
TRIGGER_WIN_S = 10.0     # trailing window
ARM_AFTER_S = 120.0      # trigger window after release
HOLD_S = 30.0            # time exit
SL_USD = 10.0            # broker-side disaster stop
VOLUME = 0.01
DEVIATION_POINTS = 3000  # max slippage accepted on order (3000 pts = $3)
MAGIC = 20260722
LOG = Path("../data/processed/impulse_trades.csv")
LOG_FIELDS = ["event", "event_time_utc", "trigger_time", "side", "impulse",
              "req_price", "fill_price", "entry_slip", "exit_req", "exit_fill",
              "exit_slip", "pnl_usd_per_oz", "pnl_account", "volume", "note"]


def log_row(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def get_mt5(allow_real: bool):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")
    ai = mt5.account_info()
    mode = {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(ai.trade_mode, "?")
    print(f"account {ai.login} [{mode}] balance {ai.balance} {ai.currency} ({ai.server})")
    if mode == "REAL" and not allow_real:
        mt5.shutdown()
        raise SystemExit("REAL account detected — refusing. Pass --allow-real only "
                         "after the demo record justifies it.")
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"symbol_select({SYMBOL}) failed")
    return mt5


def market_order(mt5, side: int, volume: float, sl_price: float | None, comment: str):
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if side > 0 else tick.bid
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": DEVIATION_POINTS,
        "magic": MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if sl_price is not None:
        req["sl"] = round(sl_price, 3)
    res = mt5.order_send(req)
    return price, res


def close_position(mt5, position):
    side = -1 if position.type == mt5.POSITION_TYPE_BUY else 1
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if side > 0 else tick.bid
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL,
        "position": position.ticket,
        "price": price,
        "deviation": DEVIATION_POINTS,
        "magic": MAGIC,
        "comment": "impulse_time_exit",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    return price, res


def upcoming_events(hours_ahead: float = 24 * 7):
    from . import ff_calendar
    evs = []
    for wk in ("thisweek", "nextweek"):
        try:
            df = ff_calendar.fetch_week(wk)
        except Exception as e:
            print(f"  feed {wk} failed: {e}")
            continue
        m = (df["currency"] == "USD") & (df["impact"].str.lower() == "high")
        evs.append(df[m])
    import pandas as pd
    if not evs:
        return []
    df = pd.concat(evs).drop_duplicates(subset=["ts_utc", "event"])
    now = pd.Timestamp.now(tz="UTC")
    df = df[(df["ts_utc"] > now) & (df["ts_utc"] < now + pd.Timedelta(hours=hours_ahead))]
    # collapse simultaneous releases into one armed window
    df = df.sort_values("ts_utc")
    out, last = [], None
    for _, r in df.iterrows():
        if last is not None and (r["ts_utc"] - last).total_seconds() < 60:
            out[-1]["event"] += f" + {r['event']}"
            continue
        last = r["ts_utc"]
        out.append({"ts": r["ts_utc"], "event": r["event"]})
    return out


def trade_window(mt5, event_name: str, t_release: float) -> None:
    """Tight loop from release to +ARM_AFTER_S; one trade max."""
    print(f"\nARMED: {event_name} (release {datetime.fromtimestamp(t_release, timezone.utc):%H:%M:%S} UTC)")
    hist: deque[tuple[float, float]] = deque()
    traded = False
    while True:
        now = time.time()
        if now > t_release + ARM_AFTER_S:
            print("  window closed" + ("" if traded else " — no trigger"))
            return
        tick = mt5.symbol_info_tick(SYMBOL)
        mid = (tick.bid + tick.ask) / 2
        hist.append((now, mid))
        while hist and hist[0][0] < now - TRIGGER_WIN_S:
            hist.popleft()
        if not traded and now >= t_release and len(hist) > 2:
            move = mid - hist[0][1]
            if abs(move) >= TRIGGER_USD:
                traded = True
                side = 1 if move > 0 else -1
                sl = mid - side * SL_USD
                req_price, res = market_order(mt5, side, VOLUME, sl,
                                              f"impulse {event_name[:20]}")
                if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"  ORDER FAILED: {getattr(res, 'retcode', '?')} "
                          f"{getattr(res, 'comment', '')}")
                    log_row({"event": event_name, "trigger_time": datetime.now(timezone.utc).isoformat(),
                             "side": side, "impulse": round(move, 2), "req_price": req_price,
                             "note": f"order failed {getattr(res, 'retcode', '?')}"})
                    return
                fill = res.price
                print(f"  TRIGGER {'+' if side>0 else '-'}${abs(move):.2f} -> "
                      f"{'BUY' if side>0 else 'SELL'} {VOLUME} @ {fill} "
                      f"(req {req_price}, slip {side*(fill-req_price):+.3f}) SL {sl:.3f}")
                time.sleep(HOLD_S)
                pos = [p for p in (mt5.positions_get(symbol=SYMBOL) or [])
                       if p.magic == MAGIC]
                if not pos:
                    note, exit_req, exit_fill = "closed by SL before time exit", "", ""
                    pnl_oz = ""
                else:
                    exit_req, eres = close_position(mt5, pos[0])
                    exit_fill = eres.price if eres and eres.retcode == mt5.TRADE_RETCODE_DONE else ""
                    note = "" if exit_fill != "" else f"exit failed {getattr(eres,'retcode','?')}"
                    pnl_oz = round(side * (exit_fill - fill), 2) if exit_fill != "" else ""
                    print(f"  EXIT @ {exit_fill} (req {exit_req}) pnl {pnl_oz} $/oz")
                log_row({"event": event_name,
                         "event_time_utc": datetime.fromtimestamp(t_release, timezone.utc).isoformat(),
                         "trigger_time": datetime.now(timezone.utc).isoformat(),
                         "side": side, "impulse": round(move, 2),
                         "req_price": req_price, "fill_price": fill,
                         "entry_slip": round(side * (fill - req_price), 3),
                         "exit_req": exit_req, "exit_fill": exit_fill,
                         "exit_slip": (round(-side * (exit_fill - exit_req), 3)
                                       if exit_fill != "" else ""),
                         "pnl_usd_per_oz": pnl_oz,
                         "pnl_account": (round(pnl_oz * VOLUME * 100, 2)
                                         if pnl_oz != "" else ""),
                         "volume": VOLUME, "note": note})
                return
        time.sleep(0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test-order", action="store_true")
    ap.add_argument("--arm-now", type=int, metavar="SECONDS")
    ap.add_argument("--allow-real", action="store_true")
    a = ap.parse_args()

    if a.list:
        for e in upcoming_events():
            print(f"  {e['ts']:%a %Y-%m-%d %H:%M} UTC  {e['event']}")
        return

    mt5 = get_mt5(a.allow_real)
    try:
        if a.test_order:
            print("test: opening+closing 0.01 to verify the order path...")
            req_price, res = market_order(mt5, 1, 0.01, None, "impulse selftest")
            assert res is not None and res.retcode == mt5.TRADE_RETCODE_DONE, \
                f"open failed: {res}"
            print(f"  opened @ {res.price} (req {req_price})")
            time.sleep(2)
            pos = [p for p in mt5.positions_get(symbol=SYMBOL) if p.magic == MAGIC][0]
            exit_req, eres = close_position(mt5, pos)
            assert eres.retcode == mt5.TRADE_RETCODE_DONE, f"close failed: {eres}"
            print(f"  closed @ {eres.price} (req {exit_req}) — order path OK")
            return
        if a.arm_now:
            trade_window(mt5, "MECHANICS-TEST", time.time() + 2)
            return
        events = upcoming_events()
        if not events:
            print("no upcoming high-impact USD events in the feed.")
            return
        print(f"{len(events)} armed events ahead:")
        for e in events:
            print(f"  {e['ts']:%a %H:%M} UTC  {e['event']}")
        for e in events:
            wait = e["ts"].timestamp() - time.time() - 15
            if wait > 0:
                print(f"\nsleeping {wait/60:.1f} min until {e['event']} ...")
                time.sleep(wait)
            trade_window(mt5, e["event"], e["ts"].timestamp())
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
