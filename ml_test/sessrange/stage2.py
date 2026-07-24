"""Stage 2: (a) pooled symmetric test, (b) A/C decomposition control,
(c) width-sort formalisation, (d) USDJPY drill-down."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'c:/Users/David/Downloads/prototype_3-main/prototype_3-main/ml_test')
from sessrange.core import *
pd.set_option('display.width',300,'display.max_rows',600,'display.max_colwidth',110)

HOLDS=[30,60,120,240]
RANGES={'ASIA':(0,7,7,20),'LONDON':(7,12,12,21)}
CACHE={}
for sym in SYMS:
    d=load(sym); dkey=d.index.normalize()
    for rname,(a,b,c,e) in RANGES.items():
        hi,lo,blk,w=range_level(sym,a,b,c,e)
        CACHE[(sym,rname)]=(hi,lo,blk,w,width_tercile(w,dkey))

def yr(tr,minn=25):
    g=tr.groupby('year')['bp'].agg(['mean','size']); g=g[g['size']>=minn]
    return f"{int((g['mean']>0).sum())}/{len(g)}", ' '.join(f"{int(y)}:{m:+.1f}" for y,m in g['mean'].items())

print('='*100); print('(A) POOLED SYMMETRIC: long the high-break + short the low-break, one book')
rows=[]
for sym in SYMS:
    for rname in RANGES:
        hi,lo,blk,w,terc=CACHE[(sym,rname)]
        for hold in HOLDS:
            up=backtest(sym,hi,blk,+1,hold); dn=backtest(sym,lo,blk,-1,hold)
            both=pd.concat([up,dn],ignore_index=True)
            if len(both)<50: continue
            x=both['bp'].values; t=x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
            yp,ys=yr(both,40)
            rows.append(dict(sym=sym,rng=rname,hold=hold,n=len(both),
                gross=round(both['gross_bp'].mean(),2),cost=round(both['cost_bp'].mean(),2),
                net=round(x.mean(),2),t=round(float(t),2),hit=round(float((x>0).mean()),3),
                yrpos=yp,upnet=round(up['bp'].mean(),2),dnnet=round(dn['bp'].mean(),2)))
P=pd.DataFrame(rows); P.to_csv('sessrange/pooled.csv',index=False)
print(P.to_string())

print(); print('='*100); print('(B) A/C CONTROL: A=long up-break, C=short same up-break. A+C should = -2*cost')
for sym in ['XAUUSDm','USDJPYm','USOILm','USTECm']:
    hi,lo,blk,w,terc=CACHE[(sym,'ASIA')]
    A=backtest(sym,hi,blk,+1,60)
    print(f'  {sym:9s} ASIA-UP gross={A["gross_bp"].mean():+6.2f}  cost={A["cost_bp"].mean():5.2f}  '
          f'net={A["bp"].mean():+6.2f}   (short-side net would be {-A["gross_bp"].mean()-A["cost_bp"].mean():+6.2f})')

print(); print('='*100); print('(C) WIDTH SORT, pooled over symbols (hold=120, both directions, gross bp)')
wr=[]
for sym in SYMS:
    for rname in RANGES:
        hi,lo,blk,w,terc=CACHE[(sym,rname)]
        for dirn,lev in ((+1,hi),(-1,lo)):
            tr=backtest(sym,lev,blk,dirn,120,tag_series=terc)
            for tv in (0,1,2):
                s=tr[tr['tag']==tv]
                if len(s)<40: continue
                wr.append(dict(sym=sym,rng=rname,dir='UP' if dirn>0 else 'DN',terc=tv,
                               n=len(s),gross=s['gross_bp'].mean(),cost=s['cost_bp'].mean(),
                               net=s['bp'].mean()))
W=pd.DataFrame(wr); W.to_csv('sessrange/width.csv',index=False)
piv=W.pivot_table(index=['sym'],columns='terc',values='gross',aggfunc='mean').round(2)
piv.columns=['gross_narrow','gross_mid','gross_wide']
pivn=W.pivot_table(index=['sym'],columns='terc',values='net',aggfunc='mean').round(2)
piv['net_narrow'],piv['net_mid'],piv['net_wide']=pivn[0],pivn[1],pivn[2]
print(piv.to_string())
print('\n  mean over all 10 symbols:', piv.mean().round(2).to_dict())
print('  symbols where gross_wide > gross_narrow:',int((piv.gross_wide>piv.gross_narrow).sum()),'/10')
# trade-level pooled regression of gross on tercile
Wt=[]
for sym in SYMS:
    for rname in RANGES:
        hi,lo,blk,w,terc=CACHE[(sym,rname)]
        for dirn,lev in ((+1,hi),(-1,lo)):
            tr=backtest(sym,lev,blk,dirn,120,tag_series=terc)
            tr=tr[tr['tag']>=0]
            if len(tr): Wt.append(tr.assign(sym=sym))
Wt=pd.concat(Wt)
zs=Wt.groupby('sym')['gross_bp'].transform(lambda s:(s-s.mean())/s.std())
g=pd.DataFrame({'terc':Wt['tag'].values,'z':zs.values}).groupby('terc')['z'].agg(['mean','size','std'])
g['t']=g['mean']/(g['std']/np.sqrt(g['size']))
print('\n  within-symbol z-scored gross by width tercile (all syms/sessions/dirs pooled):')
print(g.round(3).to_string())

print(); print('='*100); print('(D) USDJPY DRILL-DOWN')
sym='USDJPYm'
for rname in RANGES:
    hi,lo,blk,w,terc=CACHE[(sym,rname)]
    for hold in HOLDS:
        for dirn,lev,nm in ((+1,hi,'UP'),(-1,lo,'DN')):
            tr=backtest(sym,lev,blk,dirn,hold)
            yp,ys=yr(tr,20)
            print(f'  {rname:6s} {nm} h{hold:<4d} n={len(tr):4d} gross={tr["gross_bp"].mean():+5.2f} '
                  f'cost={tr["cost_bp"].mean():4.2f} net={tr["bp"].mean():+5.2f} '
                  f't={tr["bp"].mean()/(tr["bp"].std(ddof=1)/np.sqrt(len(tr))):+5.2f} yr={yp}  {ys}')
    print()
