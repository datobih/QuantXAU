"""Stage 3: robustness of the one surviving candidate (USDJPY London-range break)
   + cost-blocked gross family (USOIL / US indices London)."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'c:/Users/David/Downloads/prototype_3-main/prototype_3-main/ml_test')
from sessrange.core import *
pd.set_option('display.width',300,'display.max_rows',600)

def tstat(x):
    x=np.asarray(x); return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>2 else np.nan

def clustered_t(tr):
    """day-clustered t (one obs per calendar day, averaging same-day trades)"""
    g=tr.groupby(tr['ts'].dt.normalize())['bp'].mean()
    return tstat(g.values), len(g)

print('='*95); print('(1) USDJPY LONDON: finer hold sweep, pooled both directions')
sym='USDJPYm'; d=load(sym); dkey=d.index.normalize()
hi,lo,blk,w=range_level(sym,7,12,12,21)
for hold in [15,30,45,60,90,120,180,240,360]:
    up=backtest(sym,hi,blk,+1,hold); dn=backtest(sym,lo,blk,-1,hold)
    both=pd.concat([up,dn],ignore_index=True)
    ct,nd=clustered_t(both)
    print(f'  h{hold:<4d} n={len(both):4d} gross={both["gross_bp"].mean():+5.2f} cost={both["cost_bp"].mean():4.2f} '
          f'net={both["bp"].mean():+5.2f} t={tstat(both["bp"]):+5.2f} clust_t={ct:+5.2f} '
          f'| UP {up["bp"].mean():+5.2f} DN {dn["bp"].mean():+5.2f}')

print(); print('='*95); print('(2) USDJPY LONDON h120: leave-one-year-out (both dirs pooled)')
up=backtest(sym,hi,blk,+1,120); dn=backtest(sym,lo,blk,-1,120)
both=pd.concat([up,dn],ignore_index=True)
for y in sorted(both['year'].unique()):
    s=both[both['year']!=y]
    print(f'  drop {y}: n={len(s):4d} net={s["bp"].mean():+5.2f} t={tstat(s["bp"]):+5.2f}')

print(); print('='*95); print('(3) USDJPY: window sensitivity (range end / break start moved), h120 pooled')
for (a,b,c,e) in [(6,11,11,21),(7,11,11,21),(7,12,12,21),(7,13,13,21),(8,12,12,21),(8,13,13,21),(6,12,12,21)]:
    h2,l2,bk2,_=range_level(sym,a,b,c,e)
    u=backtest(sym,h2,bk2,+1,120); dd=backtest(sym,l2,bk2,-1,120)
    bb=pd.concat([u,dd],ignore_index=True)
    print(f'  range {a:02d}-{b:02d} break {c:02d}-{e:02d}: n={len(bb):4d} gross={bb["gross_bp"].mean():+5.2f} '
          f'net={bb["bp"].mean():+5.2f} t={tstat(bb["bp"]):+5.2f}')

print(); print('='*95); print('(4) SIBLING CHECK: same London-range rule on the other FX majors (h120, pooled)')
for s2 in ['EURUSDm','GBPUSDm','XAUUSDm']:
    h2,l2,bk2,_=range_level(s2,7,12,12,21)
    u=backtest(s2,h2,bk2,+1,120); dd=backtest(s2,l2,bk2,-1,120)
    bb=pd.concat([u,dd],ignore_index=True)
    print(f'  {s2:9s} n={len(bb):4d} gross={bb["gross_bp"].mean():+5.2f} cost={bb["cost_bp"].mean():4.2f} '
          f'net={bb["bp"].mean():+5.2f} t={tstat(bb["bp"]):+5.2f}')

print(); print('='*95); print('(5) COST-BLOCKED FAMILY: London-range break, GROSS only, hold sweep')
print('     (net = gross - cost at the Exness "m" spread; a 1-2bp venue would keep most of gross)')
for s2 in ['USOILm','USTECm','US500m','US30m','XAUUSDm','XAGUSDm','BTCUSDm']:
    h2,l2,bk2,_=range_level(s2,7,12,12,21)
    line=[]
    for hold in [30,60,120,240]:
        u=backtest(s2,h2,bk2,+1,hold); dd=backtest(s2,l2,bk2,-1,hold)
        bb=pd.concat([u,dd],ignore_index=True)
        line.append(f'h{hold}:{bb["gross_bp"].mean():+5.2f}(t{tstat(bb["gross_bp"]):+4.1f})')
    cost=bb['cost_bp'].mean()
    print(f'  {s2:9s} cost={cost:5.2f}  gross-> '+'  '.join(line))

print(); print('='*95); print('(6) USOIL LONDON h120 gross per-year (the biggest cost-blocked effect)')
h2,l2,bk2,_=range_level('USOILm',7,12,12,21)
u=backtest('USOILm',h2,bk2,+1,120); dd=backtest('USOILm',l2,bk2,-1,120)
bb=pd.concat([u,dd],ignore_index=True)
g=bb.groupby('year').agg(n=('bp','size'),gross=('gross_bp','mean'),net=('bp','mean')).round(2)
print(g.to_string()); print('  clustered t on gross:',round(tstat(bb.groupby(bb["ts"].dt.normalize())["gross_bp"].mean().values),2))
