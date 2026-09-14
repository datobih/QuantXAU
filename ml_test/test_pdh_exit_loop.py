"""Loop-level test: drives pdh_breakout_trader.main() with a fake MetaTrader5 module and
checks the conditional exit end to end (red at 60 -> close; green at 60 -> HOLD-EXTEND;
extended -> close at 90; --flat-hold -> close at 60). Run: python ml_test/test_pdh_exit_loop.py"""
import sys, types, time, tempfile
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdh_breakout_trader as tr

NOW = 1_800_000_000          # any epoch
tr.time.sleep = lambda s: None
tr.LOG = Path(tempfile.mkdtemp()) / "log.csv"


class FakeMT5:
    TIMEFRAME_D1 = 1; TIMEFRAME_H1 = 2
    TRADE_ACTION_DEAL = 1; TRADE_ACTION_PENDING = 5; TRADE_ACTION_SLTP = 6; TRADE_ACTION_REMOVE = 8
    ORDER_TYPE_BUY_STOP = 4; ORDER_TYPE_SELL = 1
    ORDER_TIME_DAY = 1; ORDER_TIME_SPECIFIED = 3; ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1; ORDER_FILLING_FOK = 0
    TRADE_RETCODE_DONE = 10009
    DEAL_ENTRY_OUT = 1

    def __init__(self, positions, bid):
        self.positions = positions; self.bid = bid; self.closed = []; self.t = NOW

    def initialize(self, path=None): return True
    def shutdown(self): pass
    def last_error(self): return (0, "")
    def terminal_info(self): return NS(connected=True, trade_allowed=True)
    def account_info(self): return NS(login=1, trade_mode=0, balance=10000, currency="USD", server="fake")
    def symbol_info(self, name): return NS(digits=2, point=0.01, trade_stops_level=0, volume_min=0.01,
                                           volume_step=0.01, filling_mode=2, expiration_mode=4)
    def symbol_select(self, name, on): return True
    def symbol_info_tick(self, name): return NS(time=self.t, bid=self.bid, ask=self.bid + 0.2)
    def copy_rates_from_pos(self, *a): return None          # setup "not ready" -> no arming
    def orders_get(self, symbol=None): return []
    def positions_get(self, symbol=None): return [p for p in self.positions if p.ticket not in self.closed]
    def history_deals_get(self, *a): return []
    def order_send(self, req):
        if req.get("action") == self.TRADE_ACTION_DEAL:
            self.closed.append(req["position"])
        return NS(retcode=self.TRADE_RETCODE_DONE, order=1)


def run(positions, bid, extra=()):
    fake = FakeMT5(positions, bid)
    sys.modules["MetaTrader5"] = fake
    sys.argv = ["x", "--once", "--day-boundary", "server", *extra]
    tr.main()
    return fake


def pos(ticket, minutes_ago, fill, magic=tr.MAGIC):
    return NS(ticket=ticket, magic=magic, time=NOW - minutes_ago * 60, price_open=fill, sl=fill - 12, tp=0.0,
              volume=0.01)


def log_actions():
    return [line.split(",")[1] for line in tr.LOG.read_text().splitlines()[1:]]


# 1. red at 60 -> closed at 60
f = run([pos(1, 61, 4400.0)], bid=4399.0)
assert f.closed == [1], "red trade at 61 min must be closed"
# 2. green at 60 -> not closed, HOLD-EXTEND logged
tr.LOG.unlink()
f = run([pos(2, 61, 4400.0)], bid=4402.0)
assert f.closed == [], "green trade at 61 min must NOT be closed"
assert "HOLD-EXTEND" in log_actions(), "extension must be logged"
# 3. AT the fill price is red
f = run([pos(3, 61, 4400.0)], bid=4400.0)
assert f.closed == [3]
# 4. before 60: nothing
f = run([pos(4, 59, 4400.0)], bid=4390.0)
assert f.closed == []
# 5. green trade past 90 on a fresh process: re-judged green -> extend on this pass (close comes next poll)
f = run([pos(5, 95, 4400.0)], bid=4405.0)
assert f.closed == []
# 6. red trade past 90 on a fresh process: closed
f = run([pos(6, 95, 4400.0)], bid=4395.0)
assert f.closed == [6]
# 7. --flat-hold: green at 60 is closed
f = run([pos(7, 61, 4400.0)], bid=4410.0, extra=["--flat-hold"])
assert f.closed == [7], "flat-hold must close green trades at 60"
# 8. zero bid: no decision this poll
f = run([pos(8, 61, 4400.0)], bid=0.0)
assert f.closed == []
# 9. weekly stream uses the same rule
f = run([pos(9, 61, 4400.0, magic=tr.MAGIC_W)], bid=4399.5)
assert f.closed == [9]
# 10. foreign magic untouched
f = run([pos(10, 200, 4400.0, magic=1)], bid=4300.0)
assert f.closed == []
print("test_pdh_exit_loop: all assertions passed")
