"""Stage 2: pooled symmetric test + A/C gross-vs-cost decomposition for session ranges."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'c:/Users/David/Downloads/prototype_3-main/prototype_3-main/ml_test')
from sessrange.core import *

HOLDS=[30,60,120,240]
RANGES={'ASIA':(0,7,7,20),'LONDON':(7,12,12,21)}
rows=[];pool=[]
for sym in SYMS:
    d=load(sym); idx=d.index; dkey=idx.normalize()
    for rname,(a,b,c,e) in RANGES.items():
        hi,lo,blk,w=range_level(sym,a,b,c,e)
        for hold in HOLDS:
            up=backtest(sym,hi,blk,+1,hold)      # A: long the up-break
            upS=backtest(sym,hi,blk,-1,hold)     # (not meaningful) skip
            dn=backtest(sym,lo,blk,-1,hold)      # short the down-break
            both=pd.concat([up,dn],ignore_index=True)
            if len(both)<20: continue
            x=both['bp'].values
            t=x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
            g=both.groupby('year')['bp'].agg(['mean','size']); g=g[g['size']>=25]
            rows.append(dict(sym=sym,rng=rname,hold=hold,n=len(both),
                gross=round(both['gross_bp'].mean(),2),cost=round(both['cost_bp'].mean(),2),
                net=round(x.mean(),2),t=round(float(t),2),hit=round(float((x>0).mean()),3),
                yrpos=f"{int((g['mean']>0).sum())}/{len(g)}",
                yrs=' '.join(f"{int(y)}:{m:+.1f}" for y,m in g['mean'].items()),
                nup=len(up),netup=round(up['bp'].mean(),2) if len(up) else np.nan,
                ndn=len(dn),netdn=round(dn['bp'].mean(),2) if len(dn) else np.nan))
P=pd.DataFrame(rows)
P.to_csv('c:/Users/David/Downloads/prototype_3-main/prototype_3-main/ml_test/sessrange/pooled.csv',index=False)
pd.set_option('display.width',260,'display.max_rows',400)
print(P[['sym','rng','hold','n','gross','cost','net','t','hit','yrpos','nup','netup','ndn','netdn']].to_string())
