import sys, numpy as np, pandas as pd
sys.path.insert(0,'psych_levels')
from harness import *
from scipy import stats

print('='*70); print('A. PER-YEAR for finalists')
for sym,key,dirn,hold,tfk in [('XAUUSDm','h10',+1,240,'up'),('XAUUSDm','h10',+1,120,'up'),
                              ('XAUUSDm','pdh',+1,240,'up'),
                              ('USDJPYm','h5',+1,240,None),('USDJPYm','h3',+1,240,None),
                              ('USDJPYm','h5',+1,480,None),('BTCUSDm','h3',+1,120,None)]:
    D=daily(sym); tf=D[tfk] if tfk else None
    tr=backtest(sym,D[key],D['dayid'],dirn,hold,tf); s=stat(tr)
    print(f"{sym:8s} {key:4s} hold{hold:4d} tf={tfk} n={s['n']:4d} net={s['net']:6.2f} "
          f"t={s['t']:5.2f} hit={s['hit']:.2f} {s['yrs']}  {s['yrd']}")

print('='*70); print('B. GOLD: is 10d-high an INDEPENDENT level or a subset of PDH?')
D=daily('XAUUSDm')
a=backtest('XAUUSDm',D['pdh'],D['dayid'],+1,240,D['up'])
b=backtest('XAUUSDm',D['h10'],D['dayid'],+1,240,D['up'])
sa,sb=set(a.ts.dt.normalize()),set(b.ts.dt.normalize())
print(f"pdh trade-days={len(sa)}  h10 trade-days={len(sb)}  overlap={len(sa&sb)} "
      f"({len(sa&sb)/len(sb)*100:.0f}% of h10 days are also pdh days)")
# does h10 add value ON TOP of pdh? split pdh trades by whether level was also a 10d high
lev=D['pdh']; is10=(D['pdh']>=D['h10']-1e-9)
for lab,f in (('pdh IS also the 10d high',is10),('pdh is NOT the 10d high',~is10)):
    tr=backtest('XAUUSDm',lev,D['dayid'],+1,240,(D['up']&f))
    s=stat(tr); print(f"  {lab:28s} n={s['n']:4d} net={s['net']:6.2f} t={s['t']:5.2f} {s['yrs']}")

print('='*70); print('C. ROUND vs PLACEBO paired test (gross bp), the key control')
r=pd.read_pickle('psych_levels/round_results.pkl'); r=r[r.n>=30]
k=['sym','tag','dirn','tf','hold']
m=r[r.kind=='round'].set_index(k)['gross'].align(r[r.kind=='placebo'].set_index(k)['gross'],join='inner')
d=(m[0]-m[1]).dropna()
print(f"  paired cells={len(d)}  mean(round-placebo)={d.mean():.3f}bp  median={d.median():.3f}")
print(f"  round>placebo in {(d>0).sum()}/{len(d)}  binom p={stats.binomtest((d>0).sum(),len(d)).pvalue:.2e}")
print("  (cells are NOT independent - same trades reused across holds/tf; treat as directional only)")
# independent version: one number per (sym,tag,dirn), hold=60 tf=none only
d2=d.rename('dif').reset_index(); d2=d2[(d2.hold==60)&(d2.tf=='none')]
print(f"  INDEPENDENT subset (hold60,tf=none): n={len(d2)} mean={d2.dif.mean():.2f} "
      f"round>placebo {(d2.dif>0).sum()}/{len(d2)} binom p={stats.binomtest(int((d2.dif>0).sum()),len(d2)).pvalue:.3f}")

print('='*70); print('D. PRIOR-DAY CLOSE: universality of the NEGATIVE result')
r2=pd.read_pickle('psych_levels/close_consol_results.pkl'); r2=r2[r2.n>=50]
p=r2[r2.level=='pdc']
print(f"  pdc cells: {len(p)}  gross<0 in {(p.gross<0).sum()}  mean gross={p.gross.mean():.2f}bp "
      f"mean net={p.net.mean():.2f}bp")
print(f"  by symbol, mean gross: all 10 negative = {(p.groupby('sym').gross.mean()<0).all()}")

print('='*70); print('E. USDJPY robustness: drop the 2022-24 BOJ carry trend')
D=daily('USDJPYm')
for hold in (120,240,480):
    tr=backtest('USDJPYm',D['h5'],D['dayid'],+1,hold,None)
    sub=tr[~tr.year.isin([2022,2023,2024])]
    b=sub['bp'].values
    print(f"  h5 hold{hold}: full n={len(tr)} net={tr.bp.mean():5.2f} | ex-22/23/24 n={len(b)} "
          f"net={b.mean():5.2f} t={b.mean()/(b.std(ddof=1)/np.sqrt(len(b))):5.2f}")
