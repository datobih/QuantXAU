"""Tight robustness: only the checks that can change a conclusion."""
import numpy as np, pandas as pd, sys, gc
import htf_levels as HL
from htf_levels import backtest, stats

pd.set_option('display.width', 300, 'display.max_rows', 999)
EPOCH_MON = np.datetime64('1970-01-05')

def build(sym, wshift=3):
    d = HL.load(sym); idx = d.index
    t = (idx + pd.Timedelta(hours=wshift)).values.astype('datetime64[m]')
    wid = ((t - EPOCH_MON).astype('timedelta64[m]').astype(np.int64)//(7*1440))
    mid = t.astype('datetime64[M]').astype(np.int64)
    did = idx.normalize().values.astype('datetime64[D]').astype(np.int64)
    df = pd.DataFrame({'hi':d['high'].values,'lo':d['low'].values,'cl':d['close'].values,
                       'w':wid,'m':mid,'d':did})
    wk=df.groupby('w',sort=True).agg(hi=('hi','max'),lo=('lo','min'),cl=('cl','last'))
    mo=df.groupby('m',sort=True).agg(hi=('hi','max'),lo=('lo','min'),cl=('cl','last'))
    dy=df.groupby('d',sort=True).agg(hi=('hi','max'),lo=('lo','min'),cl=('cl','last'))
    R=lambda s,k: s.reindex(k).values
    lv=dict(PWH=R(wk['hi'].shift(1),wid),PWL=R(wk['lo'].shift(1),wid),
            PMH=R(mo['hi'].shift(1),mid),PML=R(mo['lo'].shift(1),mid),
            F5H=R(dy['hi'].rolling(5).max().shift(1),did),
            F10H=R(dy['hi'].rolling(10).max().shift(1),did))
    tf=dict(none=None,
            wup=R(wk['cl'].shift(1)>wk['cl'].rolling(10).mean().shift(1),wid),
            up=R(dy['cl'].shift(1)>dy['cl'].rolling(10).mean().shift(1),did))
    del df; gc.collect(); return lv,dict(W=wid,M=mid,D=did),tf

def line(tag,sym,lv,bl,tf,lvl,bid,d_,h,f,gap=None):
    s=stats(backtest(sym,lv[lvl],bl[bid],d_,h,tf[f],max_gap_min=gap),sym,lvl,h,f,d_)
    print(f"{tag:22s} {sym:8s} {lvl:5s} blk={bid} h={h:4d} f={f:5s} n={s['n']:4d} "
          f"gross={s['gross']:+7.2f} cost={s['cost']:5.2f} net={s['net']:+7.2f} t={s['t']:+5.2f} "
          f"hit={s['hit']:.2f} yrs={s['yrs']:5s} {s.get('yrdetail','')}",flush=True)
    return s

for sym in ['EURUSDm','XAUUSDm','BTCUSDm','GBPUSDm','USDJPYm']:
    lv,bl,tf=build(sym)
    print(f"\n########## {sym} ##########",flush=True)
    print('--- A. PWH long, per-year, all holds (canon) ---')
    for h in [30,60,120,240]:
        for f in ['none','wup']:
            line('A.pwh',sym,lv,bl,tf,'PWH','W',1,h,f)
    print('--- B. weekend-gap guard (240m) ---')
    for lvl,bid,d_,f in [('PWH','W',1,'none'),('PWH','W',1,'wup'),('PMH','M',1,'none'),('PWL','W',-1,'none')]:
        line('B.canon',sym,lv,bl,tf,lvl,bid,d_,240,f)
        line('B.nogap',sym,lv,bl,tf,lvl,bid,d_,240,f,gap=330)
    print('--- C. PLACEBO arbitrary fractal vs real weekly level ---')
    for h in [60,240]:
        line('C.real_PWH',sym,lv,bl,tf,'PWH','W',1,h,'none')
        line('C.placebo_5dH_wk',sym,lv,bl,tf,'F5H','W',1,h,'none')
        line('C.placebo_10dH_wk',sym,lv,bl,tf,'F10H','W',1,h,'none')
        line('C.placebo_5dH_day',sym,lv,bl,tf,'F5H','D',1,h,'none')
    print('--- D. month levels + low side, per-year ---')
    for lvl,bid,d_,h in [('PMH','M',1,120),('PMH','M',1,240),('PML','M',-1,120),('PML','M',-1,240),
                         ('PWL','W',-1,120),('PWL','W',-1,240)]:
        line('D.'+lvl,sym,lv,bl,tf,lvl,bid,d_,h,'none')
    HL._cache.pop(sym,None); del lv,bl,tf; gc.collect()

print('\n########## WEEK-BOUNDARY SENSITIVITY (EURUSDm PWH) ##########',flush=True)
for ws in [0,3,6,12,24,72]:
    lv,bl,tf=build('EURUSDm',wshift=ws)
    for h in [120,240]:
        line(f'E.shift{ws}h','EURUSDm',lv,bl,tf,'PWH','W',1,h,'none')
    del lv,bl,tf; gc.collect()
HL._cache.pop('EURUSDm',None)

print('\n########## BTC LONGER HOLDS ##########',flush=True)
lv,bl,tf=build('BTCUSDm')
for h in [240,480,720,1440]:
    line('F.btc',lv and 'BTCUSDm',lv,bl,tf,'PWH','W',1,h,'none')
    line('F.btc',  'BTCUSDm',lv,bl,tf,'PMH','M',1,h,'none')
