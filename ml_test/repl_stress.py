import numpy as np, pandas as pd
from repl_levels import *
pd.set_option('display.width', 300, 'display.max_columns', 60)

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 3 else np.nan

def build_pwh_anchor(sym, anchor):
    """prior-week high with a configurable week-start anchor."""
    d = load(sym); idx = d.index
    wkey = pd.Index(idx.to_period(anchor).start_time)
    whi = d.groupby(wkey)['high'].max()
    lvl = whi.shift(1).reindex(wkey).values
    dl = daily_frame(sym); key = pd.Index(idx.normalize())
    tf = (dl['cl'].shift(1) > dl['sma10'].shift(1)).reindex(key).values
    return lvl, wkey.values, tf, np.ones(len(idx), bool)

CAND = ['XAUUSDm','EURUSDm','USDJPYm','GBPUSDm','USTECm','USOILm']
print('=== F. PWH week-anchor sensitivity (net_bp / t), 60m hold ===')
print(f"{'symbol':9s} " + ' '.join(f'{a:>18s}' for a in ['W-SUN(Mon)','W-SAT(Sun)','W-FRI(Sat)','W-MON(Tue)']))
for sym in CAND:
    cells = []
    for a in ['W-SUN','W-SAT','W-FRI','W-MON']:
        lvl, blk, tf, win = build_pwh_anchor(sym, a)
        tr, _, _ = run(sym, lvl, blk, tf, win, hold=60, n_rand=1)
        cells.append(f'n{len(tr):3d} {tr.net_bp.mean():+6.2f} t{tstat(tr.net_bp):+5.2f}'
                     if len(tr) > 10 else '   --')
    print(f'{sym:9s} ' + ' '.join(f'{c:>18s}' for c in cells))

print('\n=== G. FX PWH: with vs without the 10d-SMA uptrend filter ===')
for sym in ['XAUUSDm','EURUSDm','USDJPYm','GBPUSDm']:
    for lab, use in [('trendfilt', True), ('no-filter', False)]:
        lvl, blk, tf, win = build_pwh_anchor(sym, 'W-SUN')
        tr, _, _ = run(sym, lvl, blk, tf if use else None, win, hold=60, n_rand=1)
        print(f'{sym:9s} {lab} n={len(tr):4d} gross={tr.gross_bp.mean():+6.2f} '
              f'net={tr.net_bp.mean():+6.2f} t={tstat(tr.net_bp):+5.2f}')

print('\n=== H. Outlier robustness of the survivors (net_bp) ===')
for sym, lv, fn in [('XAUUSDm','PDH',build_pdh), ('XAUUSDm','PWH',build_pwh),
                    ('EURUSDm','PWH',build_pwh), ('USDJPYm','PWH',build_pwh)]:
    lvl, blk, tf, win = fn(sym)
    tr, _, _ = run(sym, lvl, blk, tf, win, hold=60, n_rand=1)
    x = tr.net_bp.values
    w = np.clip(x, np.percentile(x, 5), np.percentile(x, 95))
    print(f'{sym:9s} {lv:4s} n={len(x):4d} mean={x.mean():+6.2f} t={tstat(x):+5.2f} | '
          f'winsor5-95 mean={w.mean():+6.2f} t={tstat(w):+5.2f} | median={np.median(x):+6.2f} | '
          f'drop-top3 mean={np.sort(x)[:-3].mean():+6.2f}')

print('\n=== I. Yearly net_bp for survivors ===')
for sym, lv, fn in [('XAUUSDm','PDH',build_pdh), ('XAUUSDm','PWH',build_pwh),
                    ('EURUSDm','PWH',build_pwh), ('USDJPYm','PWH',build_pwh)]:
    lvl, blk, tf, win = fn(sym)
    tr, _, _ = run(sym, lvl, blk, tf, win, hold=60, n_rand=1)
    g = tr.groupby('year').net_bp.agg(['count','mean'])
    print(f'{sym} {lv}: ' + '  '.join(f'{y}:n{int(r["count"])}/{r["mean"]:+.1f}' for y, r in g.iterrows()))

print('\n=== J. What cost would XAG / BTC / USOIL need? (bp, 60m hold, PDH & PWH) ===')
for sym in ['XAGUSDm','BTCUSDm','USOILm']:
    for lv, fn in [('PDH',build_pdh), ('PWH',build_pwh), ('ASIAN_HI',build_asian)]:
        lvl, blk, tf, win = fn(sym)
        tr, _, _ = run(sym, lvl, blk, tf, win, hold=60, n_rand=1)
        g = tr.gross_bp.values; c = tr.cost_bp.mean()
        se = g.std(ddof=1)/np.sqrt(len(g))
        print(f'{sym:9s} {lv:9s} n={len(g):4d} gross={g.mean():+6.2f}(t{g.mean()/se:+.2f}) '
              f'cost_now={c:6.2f}  breakeven_cost={g.mean():6.2f}  cost_for_t2={g.mean()-2*se:6.2f}  '
              f'=> need cost cut of {c-(g.mean()-2*se):6.2f}bp ({(c-(g.mean()-2*se))/c*100:5.1f}%)')
