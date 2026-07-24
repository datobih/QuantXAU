import numpy as np, pandas as pd, sys
sys.path.insert(0,'c:/Users/David/Downloads/prototype_3-main/prototype_3-main/ml_test')
from sessrange.core import *

HOLDS=[30,60,120,240]
rows=[]
CELLS=0

RANGES={'ASIA':(0,7,7,20), 'LONDON':(7,12,12,21)}

for sym in SYMS:
    d=load(sym); idx=d.index; dkey=idx.normalize()
    tfu,tfd,_=daily_trend(sym)
    for rname,(a,b,c,e) in RANGES.items():
        hi,lo,blk,w = range_level(sym,a,b,c,e)
        terc = width_tercile(w, dkey)
        for dirn,lev,tf in ((+1,hi,tfu),(-1,lo,tfd)):
            for hold in HOLDS:
                tr=backtest(sym,lev,blk,dirn,hold,tag_series=terc)
                st=stats(tr); CELLS+=1
                rows.append(dict(sym=sym,rng=rname,dir='UP' if dirn>0 else 'DN',
                                 hold=hold,filt='none',**st))
                trf=backtest(sym,lev,blk,dirn,hold,trend_filter=tf,tag_series=terc)
                stf=stats(trf); CELLS+=1
                rows.append(dict(sym=sym,rng=rname,dir='UP' if dirn>0 else 'DN',
                                 hold=hold,filt='trend',**stf))
                if hold==60 and len(tr)>30:
                    for tv in (0,1,2):
                        sub=tr[tr['tag']==tv]
                        s2=stats(sub); CELLS+=1
                        rows.append(dict(sym=sym,rng=rname,dir='UP' if dirn>0 else 'DN',
                                         hold=hold,filt=f'wq{tv}',**s2))

# US overnight for indices
for sym in ['US30m','US500m','USTECm','BTCUSDm']:
    d=load(sym); idx=d.index
    tfu,tfd,_=daily_trend(sym)
    hi,lo,blk,w = us_overnight_level(sym)
    for dirn,lev,tf in ((+1,hi,tfu),(-1,lo,tfd)):
        for hold in HOLDS:
            tr=backtest(sym,lev,blk,dirn,hold); CELLS+=1
            rows.append(dict(sym=sym,rng='ONIGHT',dir='UP' if dirn>0 else 'DN',
                             hold=hold,filt='none',**stats(tr)))
            trf=backtest(sym,lev,blk,dirn,hold,trend_filter=tf); CELLS+=1
            rows.append(dict(sym=sym,rng='ONIGHT',dir='UP' if dirn>0 else 'DN',
                             hold=hold,filt='trend',**stats(trf)))

R=pd.DataFrame(rows)
R.to_csv('c:/Users/David/Downloads/prototype_3-main/prototype_3-main/ml_test/sessrange/grid.csv',index=False)
print('CELLS',CELLS)
pd.set_option('display.width',250,'display.max_rows',700)
print(R[R.filt.isin(['none','trend'])][['sym','rng','dir','hold','filt','n','gross','cost','net','t','hit','yrpos']].to_string())
