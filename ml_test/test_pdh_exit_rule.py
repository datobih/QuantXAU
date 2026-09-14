"""Unit test for the conditional time exit in pdh_breakout_trader.exit_decision.
Run: python ml_test/test_pdh_exit_rule.py   (no terminal needed)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdh_breakout_trader import exit_decision as ed

FILL = 4400.00
H, G = 60, 90


def close_at(elapsed_min, bid, flat=False, extended=False):
    return ed(elapsed_min * 60, bid, FILL, H, G, flat, extended)


# --- default rule: red at 60 -> close; green at 60 -> extend; close at 90 ---
assert close_at(59.9, FILL - 5) == (None, None), "nothing before the check minute"
assert close_at(60.0, FILL - 5)[0] == "close", "red at 60 closes"
assert close_at(60.0, FILL)[0] == "close", "AT the fill price counts as red (bid <= fill)"
assert close_at(60.3, FILL + 0.01)[0] == "extend", "green at 60 extends"
assert close_at(75, FILL - 20, extended=True) == (None, None), "once extended, a later red does not close before 90"
assert close_at(89.9, FILL + 5, extended=True) == (None, None)
act, why = close_at(90.0, FILL - 3, extended=True)
assert act == "close" and "90min" in why, "extended positions close at 90 regardless of P&L"
assert close_at(95, FILL + 9, extended=True)[0] == "close", "late polls after 90 still close"
# a green trade that was never marked extended (e.g. after a restart) is re-judged on the current bid
assert close_at(70, FILL + 2)[0] == "extend"
assert close_at(70, FILL - 2)[0] == "close"

# --- --flat-hold: old spec, close at 60 regardless ---
assert close_at(59.9, FILL + 5, flat=True) == (None, None)
assert close_at(60.0, FILL + 5, flat=True)[0] == "close", "flat: green at 60 still closes"
assert close_at(60.0, FILL - 5, flat=True)[0] == "close"
assert close_at(95, FILL + 5, flat=True, extended=True)[0] == "close", "flat ignores the extended flag"
assert "60min" in close_at(60.0, FILL + 5, flat=True)[1]

# --- reasons are informative ---
assert "red at 60min" in close_at(60.0, FILL - 1)[1]
print("test_pdh_exit_rule: all assertions passed")
