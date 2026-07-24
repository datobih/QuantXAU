# XAUUSD Quantitative Research

Research repository for systematic trading strategies on gold (XAUUSD), built against
MetaTrader 5 and Dukascopy data. Contains one validated, live-forward-tested strategy and
a large archive of hypotheses that were tested and rejected.

The rejected work is kept deliberately. Knowing *what does not work* — and why each idea
died — is most of the value here.

---

## 1. The validated strategy: Prior-Day-High breakout

The only strategy in this repo that survived full validation. Currently running
forward on a demo account.

### Spec

| | |
|---|---|
| **Instrument** | XAUUSD (gold), M1 bars, server clock = UTC |
| **Setup** | `PDH` = prior day's high. Uptrend = yesterday's daily *close* > 10-day SMA of daily closes |
| **Entry** | Buy-stop resting at PDH. Fills on first touch (fakeouts included). Only if the day opens *below* PDH. One trade per day, long only |
| **Exit** | Close 60 minutes after fill. Broker-side disaster stop at −$12/oz |
| **Second stream** | Same rules on the prior *week's* high (`PWH`), one trade per week |
| **Costs modelled** | Actual per-bar spread + 1/3 median spread slippage (≈ $0.30/oz round trip) |

### Evidence

Backtest 2021-08 → 2026-07, realistic first-touch fills, real spreads:

| metric | daily (PDH) | weekly (PWH) |
|---|---|---|
| trades | 470 | 109 |
| mean | **+$1.62/oz** (+5.48 bp) | **+$2.55/oz** (+9.57 bp) |
| t-stat | 3.76 | 3.19 |
| hit rate | 54% | 61% |
| years positive | **6/6** | **6/6** |
| profit / max-drawdown | 8.98× | 4.23× |

It passed every falsification test applied to it:

- **Drift control** — random long entries on the *same uptrend days and hours* returned
  **+$0.08**; the strategy returned **+$2.96** gross. The edge is entry timing, not gold's
  bull market (p < 0.0001).
- **Placebo levels** — fake levels below PDH, and the stale 2-day-old high, are dead
  (−$0.26, −$0.24). Only genuine new highs work.
- **Independent-feed replication** — the identical spec on **Dukascopy** data (a completely
  separate price feed) gives **+$1.87/oz, weekly-cluster t = 2.85**, positive every year.
  9 of the 10 biggest trades reconcile to within a few dollars across feeds.
- **Cluster-robust stats** — weekly-block bootstrap 95% CI on the mean: **[+$0.72, +$2.49]**,
  excludes zero.
- **Stationarity** — normalised by the daily ATR known *before* entry, the edge is flat
  across all six years (+0.065, +0.028, +0.054, +0.069, +0.042, +0.066 ATR/trade;
  pooled **t = 4.83**). Gold's ATR more than doubled over the period (104 → 236 bp);
  the strategy captures a constant ~5% of a daily range. **The edge is stationary — only
  volatility grew.**
- **Parameter robustness** — the trend filter is not tuned: SMA 5–100, EMA 10–50 and
  MA-slope variants all yield +$1.1–1.8/oz. The hold is a flat plateau: 30–360 min are
  statistically indistinguishable (paired t of 120 vs 60 min = 1.72).
- **News** — no measurable influence on edge or risk. Only 2.5% of trades hold through a
  high-impact release; max drawdown contains zero news-touched trades.

### Known limits (stated, not hidden)

- Long only. Breaking prior-day *lows* carries no momentum (−$0.08, t = −0.16) on any
  symbol tested.
- All in-sample 2021–2026. Live forward testing is the only remaining validation.
- Dollar returns scale with volatility. If gold's ATR reverts toward 110 bp, per-trade
  dollars roughly halve — the *edge* is stationary, the *payout* is not.
- Profit is lumpy: the best 6 months of 60 delivered 62% of all profit.

### Running it

```bash
cd ml_test
python pdh_breakout_trader.py --dry-run      # show today's decision, place nothing
python pdh_breakout_trader.py                # trade (demo-only by default)
```

Requires the MT5 terminal running, logged in, with **Algo Trading enabled**. The trader
refuses real accounts unless `--allow-real` is passed. For unattended operation use
`run_pdh_trader.bat` (auto-restart wrapper) and `install_pdh_task.bat` (start at logon).

Every decision, fill and exit is logged to `data/processed/pdh_trader_log.csv`.

---

## 2. Research tracks that were tested and killed

| track | verdict |
|---|---|
| **Post-news drift** | Dead. No drift on any of 16 US indicators, any horizon, after costs. A CPI candidate (t = 3.19) proved to be a sample-selection artifact and was independently killed by a one-shot locked holdout (t = −0.12). |
| **Tick microstructure at breakouts** | Dead — and instructive. Break-velocity "predicted" continuation at t = 2.96 and *replicated out-of-sample*, but the 30 s measurement window overlapped the outcome window. Entering after the signal was observable: t = −0.19. **Out-of-sample replication cannot catch a leak built into the construction.** |
| **Key levels on 9 other symbols** | Dead. ~4,000 cells across silver, indices, crypto, FX, oil. Zero non-gold cells cleared the bar. The breakout impulse is near-universal (t > 2 on 7 of 10 symbols) — what is gold-specific is the **impulse-to-cost ratio: 3.68 on gold, 0.03–0.81 everywhere else.** Gold's edge is a cost edge, not a behavioural one. |
| **Opening-range breakouts** | Dead, with a mechanism: 0 of 432 cells above t = +2. The OR break has **negative timing value** (−18 to −42 bp) because the range is defined by the very impulse being traded — a late entry by construction. |
| **Session ranges / Asian range** | Retracted. Looked symmetric under close-based crossing; does not survive realistic first-touch entry. |
| **Rolling-range breakouts** | Dead. Post-break price is a random walk (MFE ≈ MAE), 84–94% false-break rate. |
| **Blanket tick-level ML** | Dead. Test correlation with the forward 60 s move is +0.016 — real but ~30× smaller than the spread. |

**Cost-blocked (real phenomenon, wrong venue):** silver's weekly-high break has gross
+18.89 bp against a 19.24 bp spread. At COMEX-grade costs (~2.2 bp) that would be
+16.7 bp/trade. Filed as intelligence for a possible venue change, not as a strategy.

---

## 3. Repo layout

```
ml_test/
  pdh_breakout_trader.py     the live trader (validated strategy)
  run_pdh_trader.bat         auto-restart wrapper
  install_pdh_task.bat       register to start at logon
  news_drift/                reusable module: FF calendar, tick/bar fetchers, alignment
    duka_fetch.py            Dukascopy tick fetcher (free, bid/ask, back to 2003)
    duka_m1.py               Dukascopy M1 candle fetcher
    fetch_bars.py            MT5 bar fetcher (needs "Max bars in chart" = Unlimited)
    tick_archive.py          rolling tick archiver (broker keeps only ~3 weeks)
    ff_calendar.py           ForexFactory calendar parser, real consensus surprises
    align.py                 event↔bar alignment, broker clock verification
  <many>.py                  archived research scripts (see §2)
src/, smc/, configs/, models/   older experiments, largely superseded
```

## 4. Setup

```bash
pip install numpy pandas pyarrow MetaTrader5 scikit-learn
```

**Data is not committed** (`data/` is gitignored — it runs to ~760 MB). Regenerate it:

```bash
cd ml_test
python -m news_drift.fetch_bars  --symbol XAUUSDm --out ../data/raw/XAUUSD1.csv
python -m news_drift.duka_m1     --start 2021-07-01 --end 2026-07-22
python -m news_drift.tick_archive --symbol XAUUSDm      # run weekly; broker expires ticks
```

`FRED_API_KEY` must be set in the environment for `news_reaction_backtest.py`
(free key from https://fredaccount.stlouisfed.org/apikeys). Never hard-code it.

---

## 5. Methodology notes

Traps that produced false positives in this repo, recorded so they are not repeated:

1. **Measurement overlap** — if the signal window overlaps the outcome window, the result
   is mechanical and *will* replicate out-of-sample. Always test entering *after* the
   signal is observable.
2. **Sample-definition bias** — the CPI "edge" came from silently filtering which days
   counted. Attack the sample definition before believing a result.
3. **Session confounds** — "news vs clean" was really "US session vs Asia session".
   Stratify before concluding.
4. **Nested holds are not independent confirmations** — 30/60/120/240 min are the same
   trades with nested exit windows. Use paired tests.
5. **Multiple comparisons** — ~4,000 cells produce ~200 results at |t| > 2 by chance.
   Judge by year-stability, placebo tests and random-entry benchmarks, not headline t.
6. **Filters must beat random deletion** — a filter that removes N trades is only useful if
   it beats removing N trades at random. Most proposed filters do not.

## Disclaimer

Research code. Nothing here is financial advice. The one validated strategy is under demo
forward testing and has not been traded with real capital.
