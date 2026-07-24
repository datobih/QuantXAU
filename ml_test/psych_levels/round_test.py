import sys, numpy as np, pandas as pd
sys.path.insert(0,'psych_levels')
from harness import *

# (primary, secondary) round increments per instruction
INC={'XAUUSDm':(50,100),'XAGUSDm':(1.0,0.5),'US30m':(500,1000),'US500m':(100,50),
     'USTECm':(500,250),'BTCUSDm':(1000,5000),'EURUSDm':(0.01,0.005),
     'GBPUSDm':(0.01,0.005),'USDJPYm':(1.0,0.5),'USOILm':(5.0,1.0)}
OFF=0.37   # placebo shift as fraction of increment

def round_levels(sym, inc, off_frac):
    """per-bar arrays of the nearest grid level above / below the day's opening price"""
    d=load(sym); idx=d.index; n=idx.normalize()
    ref=d['close'].groupby(n).first()          # first M1 close of each day
    off=off_frac*inc
    up  =(np.floor((ref-off)/inc)+1)*inc+off
    dn  = np.ceil((ref-off)/inc)*inc+off-inc
    return up.reindex(n).values, dn.reindex(n).values

rows=[]
for sym in SYMS:
    D=daily(sym)
    for tag,inc in zip(('P','S'),INC[sym]):
        for kind,offf in (('round',0.0),('placebo',OFF)):
            up,dn=round_levels(sym,inc,offf)
            for dirn,lev,tf in ((+1,up,None),(+1,up,D['up']),(-1,dn,None),(-1,dn,D['down'])):
                for hold in (30,60,120):
                    tr=backtest(sym,lev,D['dayid'],dirn,hold,tf)
                    s=stat(tr)
                    rows.append(dict(sym=sym,inc=inc,tag=tag,kind=kind,
                                     dirn='up' if dirn>0 else 'dn',
                                     tf='trend' if tf is not None else 'none',
                                     hold=hold,**{k:v for k,v in s.items() if k!='yrd'},
                                     yrd=s.get('yrd')))
                    print(sym,inc,kind,('up' if dirn>0 else 'dn'),
                          'trend' if tf is not None else 'none',hold,
                          s['n'],round(s['net'],2) if s['n']>=10 else '-',flush=True)
r=pd.DataFrame(rows)
r.to_pickle('psych_levels/round_results.pkl')
print('cells',len(r))
