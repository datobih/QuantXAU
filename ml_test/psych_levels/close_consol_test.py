import sys, numpy as np, pandas as pd
sys.path.insert(0,'psych_levels')
from harness import *

rows=[]
for sym in SYMS:
    D=daily(sym)
    specs=[('pdc','pdc',+1),('pdc','pdc',-1)]
    for k in (3,5,10):
        specs.append((f'{k}dhigh',f'h{k}',+1))
        specs.append((f'{k}dlow', f'l{k}',-1))
    specs.append(('pdh','pdh',+1))   # reference control
    specs.append(('pdl','pdl',-1))
    for name,key,dirn in specs:
        lev=D[key]
        for tfn,tf in (('none',None),('trend',D['up'] if dirn>0 else D['down'])):
            for hold in (30,60,120):
                tr=backtest(sym,lev,D['dayid'],dirn,hold,tf)
                s=stat(tr)
                rows.append(dict(sym=sym,level=name,dirn='up' if dirn>0 else 'dn',
                                 tf=tfn,hold=hold,**{k2:v for k2,v in s.items() if k2!='yrd'},
                                 yrd=s.get('yrd')))
    print('done',sym,flush=True)
r=pd.DataFrame(rows); r.to_pickle('psych_levels/close_consol_results.pkl')
print('cells',len(r))
