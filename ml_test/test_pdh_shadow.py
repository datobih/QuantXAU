"""Tests for the shadow-ledger features: shadow_features() on synthetic lows, both setup paths returning them, and a SHADOW
log row written by main() with the right values. Run: python ml_test/test_pdh_shadow.py"""
import sys, time, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdh_breakout_trader as tr

# ---------- 1. the pure function ----------
sf = tr.shadow_features
assert sf([1.0] * 10, [1.0, 2.0]) == {"days_since_20d_low": None, "prior_week_higher_low": True}, "too few days -> None"
lows = [100.0 - k for k in range(25)]          # falling every day: yesterday (last) is a fresh low
assert sf(lows, [])["days_since_20d_low"] == 0
lows = [100.0 - k for k in range(25)] + [80.0 + k for k in range(5)]   # fresh low 5 days ago, then rising
assert sf(lows, [])["days_since_20d_low"] == 5
lows = [50.0 + k for k in range(40)]          # rising for 40 days: no fresh low in the window -> capped 60? no: earliest checkable day j=20
r = sf(lows, [])["days_since_20d_low"]
assert r == 60, r                              # never a fresh low within the checkable range -> cap
assert sf([1.0] * 30, [5.0, 4.0])["prior_week_higher_low"] is False
assert sf([1.0] * 30, [4.0, 5.0])["prior_week_higher_low"] is True
assert sf([1.0] * 30, [5.0])["prior_week_higher_low"] is None
# a day equal to the prior 20-day min counts as a fresh low
lows = [10.0] * 20 + [10.0] + [12.0, 13.0]
assert sf(lows, [])["days_since_20d_low"] == 2

# ---------- 2. both setup paths return the features ----------
tr.time.sleep = lambda s: None
tr.LOG = Path(tempfile.mkdtemp()) / "log.csv"
DT = np.dtype([("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8")])
today = datetime.now(timezone.utc).date()
def midnight(d): return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
def d1_rates(fresh_low_days_ago, week_hl):
    """30 completed weekdays, rising closes; a fresh 20-day low `fresh_low_days_ago` completed days ago;
    weekly lows arranged so prior week's low is higher (True) or lower (False) than the week before."""
    days = []; d = today - timedelta(days=1)
    while len(days) < 30:
        if d.weekday() < 5: days.append(d)
        d -= timedelta(days=1)
    days = days[::-1]
    out = []
    for k, x in enumerate(days):
        close = 4395.0 - 3.0 * (len(days) - 1 - k); lo = close - 15.0; hi = close + 5.0
        out.append([midnight(x), lo + 5, hi, lo, close])
    j = len(days) - 1 - fresh_low_days_ago
    out[j][3] = 4000.0                                     # a low far below everything before it
    cur = today.isocalendar()[:2]
    wk = sorted({x.isocalendar()[:2] for x in days if x.isocalendar()[:2] != cur})
    prev, before = wk[-1], wk[-2]
    for k, x in enumerate(days):                           # set the two weeks' lows explicitly (keep the fresh-low day)
        if k == j: continue
        if x.isocalendar()[:2] == prev: out[k][3] = min(out[k][3], 4300.0 if week_hl else 4200.0)
        if x.isocalendar()[:2] == before: out[k][3] = min(out[k][3], 4250.0)
    out.append([midnight(today), 4380.0, 4385.0, 4375.0, 4380.0])
    return np.array([tuple(r) for r in out], dtype=DT), days, j, prev, before
rates, days, j, prev, before = d1_rates(5, True)
class D1:
    TIMEFRAME_D1 = 1
    def copy_rates_from_pos(self, *a): return rates
s = tr.daily_setup(D1(), expect_day=today)
assert s["shadow"]["days_since_20d_low"] == 5, s["shadow"]
# weekly flag: the fresh-low day (4000) may sit in one of the two weeks -> compute expectation from the data itself
wl = {}
for k, x in enumerate(days):
    w = x.isocalendar()[:2]
    if w != today.isocalendar()[:2]: wl[w] = min(wl.get(w, 1e9), rates[k]["low"])
exp_hl = bool(wl[prev] > wl[before])
assert s["shadow"]["prior_week_higher_low"] == exp_hl, (s["shadow"], wl)

# ---------- 3. main() writes a SHADOW row (server mode, fake terminal) ----------
class Fake(D1):
    TIMEFRAME_H1 = 2; TRADE_ACTION_DEAL = 1; TRADE_ACTION_PENDING = 5; TRADE_ACTION_SLTP = 6; TRADE_ACTION_REMOVE = 8
    ORDER_TYPE_BUY_STOP = 4; ORDER_TYPE_SELL = 1; ORDER_TIME_DAY = 1; ORDER_TIME_SPECIFIED = 3; ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1; ORDER_FILLING_FOK = 0; TRADE_RETCODE_DONE = 10009; DEAL_ENTRY_OUT = 1
    def initialize(self, path=None): return True
    def shutdown(self): pass
    def last_error(self): return (0, "")
    def terminal_info(self): return NS(connected=True, trade_allowed=True)
    def account_info(self): return NS(login=1, trade_mode=0, balance=1e4, currency="USD", server="f")
    def symbol_info(self, n): return NS(digits=2, point=0.01, trade_stops_level=0, volume_min=0.01, volume_step=0.01, filling_mode=2, expiration_mode=4)
    def symbol_select(self, n, o): return True
    def symbol_info_tick(self, n): return NS(time=int(time.time()), bid=4380.0, ask=4380.2)
    def orders_get(self, symbol=None): return []
    def positions_get(self, symbol=None): return []
    def history_deals_get(self, *a): return []
    def order_send(self, req): return NS(retcode=self.TRADE_RETCODE_DONE, order=1)
sys.modules["MetaTrader5"] = Fake(); sys.argv = ["x", "--once", "--day-boundary", "server"]
tr.main()
rows = [l for l in tr.LOG.read_text().splitlines() if ",SHADOW," in l]
assert len(rows) == 1, rows
assert "days_since_20d_low=5" in rows[0] and f"prior_week_higher_low={exp_hl}" in rows[0], rows[0]
assert "pwh_gap_atr=" in rows[0] and "atr14=" in rows[0]
# and trading decisions are untouched by the shadow row: a PLACED/no-trade row still follows as before
acts = [l.split(",")[1] for l in tr.LOG.read_text().splitlines()[1:]]
assert any(a in ("PLACED", "no-trade", "DRY-would-place") for a in acts), acts
print("test_pdh_shadow: all assertions passed")
