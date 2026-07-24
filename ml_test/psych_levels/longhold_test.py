import sys, numpy as np, pandas as pd
sys.path.insert(0,'psych_levels')
from harness import *

# families that showed COHERENT gross momentum; ask whether a longer hold clears cost
CAND=[('USOILm','l3',-1),('USOILm','l5',-1),('USOILm','l10',-1),('USOILm','pdl',-1),
      ('BTCUSDm','h3',+1),('BTCUSDm','h5',+1),('BTCUSDm','h10',+1),
      ('USDJPYm','h5',+1),('USDJPYm','h3',+1),('USDJPYm','pdl',-1),
      ('XAUUSDm','h10',+1),('XAUUSDm','h5',+1),('XAUUSDm','pdh',+1),
      ('XAGUSDm','h10',+1),('XAGUSDm','l10',-1),
      ('USTECm','l5',-1),('US30m','pdl',-1),('GBPUSDm','l5',-1),('EURUSDm','h10',+1)]
rows=[]
for sym,key,dirn in CAND:
    D=daily(sym)
    for tfn,tf in (('none',None),('trend',D['up'] if dirn>0 else D['down'])):
        for hold in (60,120,240,480,720):
            tr=backtest(sym,D[key],D['dayid'],dirn,hold,tf)
            s=stat(tr)
            rows.append(dict(sym=sym,level=key,dirn='up' if dirn>0 else 'dn',tf=tfn,hold=hold,
                             **{k:v for k,v in s.items() if k!='yrd'},yrd=s.get('yrd')))
    print('done',sym,key,flush=True)
r=pd.DataFrame(rows); r.to_pickle('psych_levels/longhold_results.pkl')
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
print(r[['sym','level','dirn','tf','hold','n','gross','cost','net','t','hit','yrs']].round(2).to_string(index=False))
