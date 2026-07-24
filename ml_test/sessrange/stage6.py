"""Stage 6: does the LEVEL do any work on USDJPY, or is it just London-session momentum?
Control A: no level at all - at 12:00 UTC take the sign of the London session move, hold N.
Control B: stale level - use YESTERDAY's London range as today's level.
Control C: placebo level - range mid +/- same half-width (i.e. a level at the same distance
           but not the actual session extreme)."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'c:/Users/David/Downloads/prototype_3-main/prototype_3-main/ml_test')
from sessrange.core import *
pd.set_option('display.width',300,'display.max_rows',400)
def tstat(x):
    x=np.asarray(x,float); return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>2 else np.nan
def yrs(tr,minn=25):
    g=tr.groupby('year')['bp'].agg(['mean','size']); g=g[g['size']>=minn]
    return f'{int((g["mean"]>0).sum())}/{len(g)}', ' '.join(f'{int(y)}:{m:+.1f}' for y,m in g['mean'].items())

def firstbreak(sym,hi,lo,blk,hold):
    up=backtest(sym,hi,blk,+1,hold); dn=backtest(sym,lo,blk,-1,hold)
    both=pd.concat([up,dn],ignore_index=True).sort_values('ts')
    return both.groupby(both['ts'].dt.normalize(),as_index=False).first()

SYM='USDJPYm'
d=load(SYM); idx=d.index; pt=POINT[SYM]
O,H,L,C=(d[c].values for c in ('open','high','low','close'))
SPRD=d['spread'].values*pt; med=np.median(SPRD[SPRD>0]); slip=med/3
hr=idx.hour+idx.minute/60.0; dkey=idx.normalize()

print('='*105)
print('BASELINE  USDJPY LONDON-range first-break')
hi,lo,blk,w=range_level(SYM,7,12,12,21)
for hold in [60,120,240]:
    f=firstbreak(SYM,hi,lo,blk,hold); yp,ys=yrs(f)
    print(f'  h{hold:<4d} n={len(f):4d} gross={f["gross_bp"].mean():+5.2f} cost={f["cost_bp"].mean():4.2f} '
          f'net={f["bp"].mean():+5.2f} t={tstat(f["bp"]):+5.2f} yr={yp}  {ys}')

print()
print('='*105)
print('CONTROL A: NO LEVEL. Enter at 12:00 UTC in the direction of the 07:00-12:00 move, hold N.')
o7=pd.Series(np.where(np.isclose(hr,7.0),O,np.nan),index=idx).groupby(dkey).first()
c12i=pd.Series(np.where(np.isclose(hr,12.0),np.arange(len(idx)),np.nan),index=idx).groupby(dkey).first()
ex_all=np.clip(idx.searchsorted(idx+pd.Timedelta(minutes=0)),0,len(d)-1)
for hold in [60,120,240]:
    ex=np.clip(idx.searchsorted(idx+pd.Timedelta(minutes=hold)),0,len(d)-1)
    out=[]
    for day,ii in c12i.dropna().items():
        i=int(ii); o=o7.get(day,np.nan)
        if not np.isfinite(o) or i>=len(d)-1: continue
        sgn=1 if C[i]>o else -1
        px=C[i]; cost=(SPRD[i] if SPRD[i]>0 else med)+slip
        raw=(C[ex[i]]-px)*sgn
        out.append({'ts':idx[i],'year':idx[i].year,'bp':(raw-cost)/px*1e4,'gross_bp':raw/px*1e4,
                    'cost_bp':cost/px*1e4})
    t=pd.DataFrame(out); yp,ys=yrs(t)
    print(f'  h{hold:<4d} n={len(t):4d} gross={t["gross_bp"].mean():+5.2f} cost={t["cost_bp"].mean():4.2f} '
          f'net={t["bp"].mean():+5.2f} t={tstat(t["bp"]):+5.2f} yr={yp}  {ys}')

print()
print('='*105)
print("CONTROL B: STALE LEVEL. Use YESTERDAY's London range as today's break level.")
sh=pd.Series(hi,index=idx).groupby(dkey).first().shift(1)
sl=pd.Series(lo,index=idx).groupby(dkey).first().shift(1)
hiS=sh.reindex(dkey).values; loS=sl.reindex(dkey).values
for hold in [60,120,240]:
    f=firstbreak(SYM,hiS,loS,blk,hold); yp,ys=yrs(f)
    print(f'  h{hold:<4d} n={len(f):4d} gross={f["gross_bp"].mean():+5.2f} cost={f["cost_bp"].mean():4.2f} '
          f'net={f["bp"].mean():+5.2f} t={tstat(f["bp"]):+5.2f} yr={yp}  {ys}')

print()
print('='*105)
print('CONTROL C: PLACEBO LEVEL at mid +/- k*halfwidth (k=1 is the true extreme).')
mid=(hi+lo)/2; half=(hi-lo)/2
for k in [0.5,0.75,1.0,1.25,1.5]:
    hk=mid+k*half; lk=mid-k*half
    f=firstbreak(SYM,hk,lk,blk,120)
    print(f'  k={k:<5.2f} n={len(f):4d} gross={f["gross_bp"].mean():+5.2f} net={f["bp"].mean():+5.2f} '
          f't={tstat(f["bp"]):+5.2f}')

print()
print('='*105)
print('CONTROL D: same first-break rule on a level built from a NON-session window (13:00-18:00),')
print('           broken 18:00-21:00+ - an "arbitrary fractal" analogue.')
h2,l2,b2,_=range_level(SYM,13,18,18,21)
for hold in [60,120]:
    f=firstbreak(SYM,h2,l2,b2,hold); yp,ys=yrs(f)
    print(f'  h{hold:<4d} n={len(f):4d} gross={f["gross_bp"].mean():+5.2f} net={f["bp"].mean():+5.2f} '
          f't={tstat(f["bp"]):+5.2f} yr={yp}')
