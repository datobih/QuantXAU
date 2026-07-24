import numpy as np, pandas as pd
from repl_levels import *

pd.set_option('display.width', 300, 'display.max_columns', 60)
HOLD = 60
allt = {}
for sym in SYMS:
    for lname, fn in LEVELS.items():
        lvl, blk, tf, win = fn(sym)
        tr, r1, r2 = run(sym, lvl, blk, tf, win, direction=+1, hold=HOLD, n_rand=100)
        tr['symbol'] = sym; tr['level'] = lname
        allt[(sym, lname)] = tr
T = pd.concat(allt.values(), ignore_index=True)
T.to_parquet('repl_trades_h60.parquet')

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 3 else np.nan

print('=== A. GROSS t-stats and breakeven cost (is the PHENOMENON there, cost aside?) ===')
rows = []
for (sym, lv), tr in allt.items():
    if len(tr) < 30: continue
    g = tr.gross_bp.values; c = tr.cost_bp.mean()
    se = g.std(ddof=1)/np.sqrt(len(g))
    # cost level at which net t-stat would equal 2.0
    cost_for_t2 = g.mean() - 2*se
    rows.append(dict(symbol=sym, level=lv, n=len(g), gross=g.mean(), gross_t=g.mean()/se,
                     cost=c, net=g.mean()-c,
                     cost_for_breakeven=g.mean(), cost_for_t2=cost_for_t2,
                     cost_cut_needed_bp=c - cost_for_t2,
                     cost_cut_pct=(c - cost_for_t2)/c*100))
R = pd.DataFrame(rows).sort_values('gross_t', ascending=False)
print(R.round(2).to_string(index=False))

print('\n=== B. POOLED by level family, gold excluded (equal-weight per trade) ===')
for lv in LEVELS:
    sub = T[(T.level == lv) & (T.symbol != 'XAUUSDm')]
    subg = T[(T.level == lv) & (T.symbol == 'XAUUSDm')]
    # also equal-weight per symbol so XAG's huge cost doesn't dominate
    per_sym = sub.groupby('symbol').net_bp.mean()
    print(f'{lv:9s} nonGold n={len(sub):5d} net={sub.net_bp.mean():+7.2f} t={tstat(sub.net_bp):+5.2f} '
          f'| gross={sub.gross_bp.mean():+6.2f} t={tstat(sub.gross_bp):+5.2f} '
          f'| symEW net={per_sym.mean():+6.2f} ({(per_sym>0).sum()}/9 syms +) '
          f'|| GOLD net={subg.net_bp.mean():+6.2f} t={tstat(subg.net_bp):+5.2f}')

print('\n=== C. Excess over time-of-day-matched random entry (paired, gross) ===')
for (sym, lv), tr in allt.items():
    if len(tr) < 30: continue
    e = (tr.gross_bp - tr.rand_tod_bp).dropna()
    print(f'{sym:9s} {lv:9s} n={len(e):4d} excess={e.mean():+7.2f} t={tstat(e):+5.2f}')

print('\n=== D. Multiple testing: 30 primary cells (10 sym x 3 levels, long, 60m) ===')
S = pd.read_csv('repl_levels_h60.csv')
tt = S.t.dropna().values
print(f'cells={len(tt)}  |t|>2: {(np.abs(tt)>2).sum()}  (expect ~1.5 by chance)')
print(f'  t>+2: {(tt>2).sum()}  t<-2: {(tt<-2).sum()}')
print('  cells with t>+2:', S.loc[S.t > 2, ['symbol','level','net_bp','t']].to_dict('records'))
# Bonferroni / Sidak on the positive side
from scipy import stats
for _, r in S.sort_values('t', ascending=False).head(6).iterrows():
    p1 = 1 - stats.t.cdf(r.t, r.n - 1)
    print(f"  {r.symbol:9s} {r['level']:9s} t={r.t:+5.2f} p1={p1:.4g} "
          f"p_bonf(30)={min(1, p1*30):.4g} sidak={1-(1-p1)**30:.4g}")

print('\n=== E. Asian-range LOW, short side (symmetric level sanity check) ===')
for sym in SYMS:
    lvl, blk, tf, win = build_asian(sym, side='low')
    tr, r1, r2 = run(sym, lvl, blk, None, win, direction=-1, hold=HOLD, n_rand=50)
    if len(tr) < 30: print(f'{sym:9s} n={len(tr)}'); continue
    print(f'{sym:9s} n={len(tr):4d} gross={tr.gross_bp.mean():+6.2f} cost={tr.cost_bp.mean():5.2f} '
          f'net={tr.net_bp.mean():+7.2f} t={tstat(tr.net_bp):+5.2f}')
