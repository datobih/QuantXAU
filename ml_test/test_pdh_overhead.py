"""Tests for the overhead skip: (1) overhead_skip() cases, (2) ATR14 / pwh_gap from daily_setup_utc on synthetic
H1 bars (Sunday stub included), (3) the real arming path in main() with a fake terminal and synthetic D1 bars:
big gap -> PDH not armed but PWH armed; small gap -> PDH armed; --no-overhead-skip -> PDH armed despite the gap.
Run: python ml_test/test_pdh_overhead.py"""
import sys, time, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdh_breakout_trader as tr

tr.time.sleep = lambda s: None
tr.LOG = Path(tempfile.mkdtemp()) / "log.csv"
DT = np.dtype([("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8")])

# ---------- 1. the pure rule ----------
assert tr.overhead_skip(0.6) is True
assert tr.overhead_skip(0.5) is False, "exactly at the threshold is not skipped (rule is strictly greater)"
assert tr.overhead_skip(0.2) is False
assert tr.overhead_skip(-1.0) is False, "PDH above PWH never skips"
assert tr.overhead_skip(None) is False, "no PWH / no ATR -> stream behaves as before"
assert tr.overhead_skip(3.0, enabled=False) is False, "--no-overhead-skip"

# ---------- 2. ATR14 and gap from H1 bars (UTC mode, offset 0) ----------
today = datetime.now(timezone.utc).date()
def midnight(d): return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
bars = []; ranges = {}
for back in range(40, 0, -1):
    d = today - timedelta(days=back)
    if d.weekday() == 5: continue                       # Saturday: no bars
    hours = [22, 23] if d.weekday() == 6 else range(24)  # Sunday stub: two bars
    hi = 4400.0 - 2.0 * back; lo = hi - (6.0 if d.weekday() == 6 else 20.0)
    ranges[d] = hi - lo
    for h in hours:
        bars.append((midnight(d) + h * 3600, lo + 5, hi if h == max(hours) else hi - 3, lo, hi - 5 if h == max(hours) else hi - 8))
bars.append((midnight(today), 4380.0, 4385.0, 4375.0, 4380.0))   # forming day, high below PDH
rates = np.array(bars, dtype=DT)
class H1MT5:
    TIMEFRAME_H1 = 2
    def copy_rates_from_pos(self, sym, tf, start, n): return rates
s = tr.daily_setup_utc(H1MT5(), 0, expect_day=today)
assert s is not None
completed = sorted(ranges)
exp_atr = sum(ranges[d] for d in completed[-14:]) / 14
assert abs(s["atr14"] - exp_atr) < 1e-9, (s["atr14"], exp_atr)
assert any(d.weekday() == 6 for d in completed[-14:]), "test must include a Sunday stub in the ATR window"
assert s["pwh"] is not None and abs(s["pwh_gap"] - (s["pwh"] - s["pdh"]) / s["atr14"]) < 1e-12

# ---------- 3. the arming path (server mode, synthetic D1 bars) ----------
class FakeMT5:
    TIMEFRAME_D1 = 1; TIMEFRAME_H1 = 2
    TRADE_ACTION_DEAL = 1; TRADE_ACTION_PENDING = 5; TRADE_ACTION_SLTP = 6; TRADE_ACTION_REMOVE = 8
    ORDER_TYPE_BUY_STOP = 4; ORDER_TYPE_SELL = 1
    ORDER_TIME_DAY = 1; ORDER_TIME_SPECIFIED = 3; ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1; ORDER_FILLING_FOK = 0
    TRADE_RETCODE_DONE = 10009; DEAL_ENTRY_OUT = 1
    def __init__(self, d1): self.d1 = d1; self.placed = []; self.t = int(time.time())
    def initialize(self, path=None): return True
    def shutdown(self): pass
    def last_error(self): return (0, "")
    def terminal_info(self): return NS(connected=True, trade_allowed=True)
    def account_info(self): return NS(login=1, trade_mode=0, balance=10000, currency="USD", server="fake")
    def symbol_info(self, name): return NS(digits=2, point=0.01, trade_stops_level=0, volume_min=0.01,
                                           volume_step=0.01, filling_mode=2, expiration_mode=4)
    def symbol_select(self, name, on): return True
    def symbol_info_tick(self, name): return NS(time=self.t, bid=4380.0, ask=4380.2)
    def copy_rates_from_pos(self, sym, tf, start, n): return self.d1
    def orders_get(self, symbol=None): return []
    def positions_get(self, symbol=None): return []
    def history_deals_get(self, *a): return []
    def order_send(self, req):
        if req.get("action") == self.TRADE_ACTION_PENDING: self.placed.append((req["magic"], req["price"]))
        return NS(retcode=self.TRADE_RETCODE_DONE, order=len(self.placed))

def d1_rates(last_week_high):
    """30 completed weekdays with rising closes (uptrend). Highs on trend except the LAST COMPLETED ISO WEEK,
    whose highs are set to last_week_high. Yesterday's high (PDH) is 4400."""
    days = []; d = today - timedelta(days=1)
    while len(days) < 30:
        if d.weekday() < 5: days.append(d)
        d -= timedelta(days=1)
    days = days[::-1]
    cur_wk = today.isocalendar()[:2]
    prev_wk = max(x.isocalendar()[:2] for x in days if x.isocalendar()[:2] != cur_wk)
    out = []
    for k, x in enumerate(days):
        close = 4395.0 - 3.0 * (len(days) - 1 - k)            # rising closes -> uptrend
        hi = last_week_high if x.isocalendar()[:2] == prev_wk else close + 5.0
        lo = close - 15.0
        out.append((midnight(x), lo + 5, hi, lo, close))
    out.append((midnight(today), 4380.0, 4385.0, 4375.0, 4380.0))   # forming today, high < PDH
    return np.array(out, dtype=DT), prev_wk

def run(last_week_high, extra=()):
    d1, _ = d1_rates(last_week_high)
    fake = FakeMT5(d1); sys.modules["MetaTrader5"] = fake
    sys.argv = ["x", "--once", "--day-boundary", "server", *extra]
    tr.main()
    s = tr.daily_setup(fake, expect_day=today)
    return fake, s

# big gap: last week's high 4450 vs PDH 4400
fake, s = run(4450.0)
assert abs(s["pdh"] - 4400.0) < 1e-9 and s["uptrend"], s
assert s["pwh_gap"] > tr.OVERHEAD_ATR, s["pwh_gap"]
magics = [m for m, _ in fake.placed]
assert tr.MAGIC not in magics, "PDH must NOT be armed under a big overhead gap"
assert tr.MAGIC_W in magics, "PWH stream must still be armed"
txt = tr.LOG.read_text(); assert "overhead rule" in txt, "the skip must be logged as a no-trade reason"
# small gap: last week's high 4405 -> gap ~0.2 ATR -> PDH armed
fake, s = run(4405.0)
assert s["pwh_gap"] < tr.OVERHEAD_ATR, s["pwh_gap"]
assert tr.MAGIC in [m for m, _ in fake.placed], "PDH must be armed under a small gap"
# big gap but --no-overhead-skip -> PDH armed
fake, s = run(4450.0, extra=["--no-overhead-skip"])
assert tr.MAGIC in [m for m, _ in fake.placed], "--no-overhead-skip must restore the old behaviour"
print("test_pdh_overhead: all assertions passed")
