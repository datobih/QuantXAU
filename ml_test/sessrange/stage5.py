"""Stage 5: FIRST-BREAK-ONLY.
Diagnosis showed days where only one side of the session range breaks are strongly
trending, days where both sides break are whipsaws. 'Both sides broke' is not known
ex ante, but 'this is the first break of the day' IS. So: trade whichever boundary
breaks first, one trade per day, then stand down."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'c:/Users/David/Downloads/prototype_3-main/prototype_3-main/ml_test')
from sessrange.core import *
pd.set_option('display.width',300,'display.max_rows',600)
def tstat(x):
    x=np.asarray(x,float); return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>2 else np.nan

def firstbreak(sym,rng,hold):
    a,b,c,e=rng
    hi,lo,blk,w=range_level(sym,a,b,c,e)
    up=backtest(sym,hi,blk,+1,hold).assign(leg='UP')
    dn=backtest(sym,lo,blk,-1,hold).assign(leg='DN')
    both=pd.concat([up,dn],ignore_index=True)
    both['day']=both['ts'].dt.normalize()
    both=both.sort_values('ts')
    first=both.groupby('day',as_index=False).first()      # earliest entry that day
    second=both[~both.index.isin(both.sort_values('ts').groupby('day').head(1).index)]
    return first, both

RANGES={'ASIA':(0,7,7,20),'LONDON':(7,12,12,21)}
print('='*110)
print('FIRST-BREAK-ONLY (one trade/day, whichever side of the session range breaks first)')
for rname,rng in RANGES.items():
    print(f'\n--- {rname} range ---')
    print(f'{"sym":9s} {"hold":>5s} {"n":>5s} {"gross":>7s} {"cost":>6s} {"net":>7s} {"t":>6s} {"hit":>6s}  years')
    for sym in SYMS:
        for hold in [60,120,240]:
            f,both=firstbreak(sym,rng,hold)
            g=f.groupby('year')['bp'].agg(['mean','size']); g=g[g['size']>=25]
            yp=f'{int((g["mean"]>0).sum())}/{len(g)}'
            ys=' '.join(f'{int(y)}:{m:+.1f}' for y,m in g['mean'].items())
            print(f'{sym:9s} {hold:5d} {len(f):5d} {f["gross_bp"].mean():+7.2f} {f["cost_bp"].mean():6.2f} '
                  f'{f["bp"].mean():+7.2f} {tstat(f["bp"]):+6.2f} {(f["bp"]>0).mean():6.3f}  {yp}  {ys}')

print()
print('='*110)
print('SANITY: first-break vs second-break (LONDON h120) - is the whipsaw really in the 2nd leg?')
for sym in ['USDJPYm','XAUUSDm','USTECm','USOILm','US500m']:
    f,both=firstbreak(sym,RANGES['LONDON'],120)
    ids=both.sort_values('ts').groupby(both.sort_values('ts')['ts'].dt.normalize()).head(1).index
    sec=both[~both.index.isin(ids)]
    print(f'  {sym:9s} first n={len(f):4d} net={f["bp"].mean():+6.2f} gross={f["gross_bp"].mean():+6.2f} | '
          f'second n={len(sec):4d} net={sec["bp"].mean():+7.2f} gross={sec["gross_bp"].mean():+7.2f}')
