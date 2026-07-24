"""Robustness on HTF-level candidates. Integer block ids, one symbol resident at a time."""
import numpy as np, pandas as pd, sys, gc
import htf_levels as HL
from htf_levels import backtest, stats, POINT

pd.set_option('display.width', 300, 'display.max_rows', 999)
EPOCH_MON = np.datetime64('1970-01-05')   # a Monday

def build(sym, wshift=3):
    d = HL.load(sym); idx = d.index
    t = (idx + pd.Timedelta(hours=wshift)).values.astype('datetime64[m]')
    wid = ((t - EPOCH_MON).astype('timedelta64[m]').astype(np.int64) // (7*1440))
    ym = t.astype('datetime64[M]')
    mid = ym.astype(np.int64)
    did = idx.normalize().values.astype('datetime64[D]').astype(np.int64)

    df = pd.DataFrame({'hi':d['high'].values,'lo':d['low'].values,'cl':d['close'].values,
                       'w':wid,'m':mid,'d':did})
    wk = df.groupby('w',sort=True).agg(hi=('hi','max'),lo=('lo','min'),cl=('cl','last'))
    mo = df.groupby('m',sort=True).agg(hi=('hi','max'),lo=('lo','min'),cl=('cl','last'))
    dy = df.groupby('d',sort=True).agg(hi=('hi','max'),lo=('lo','min'),cl=('cl','last'))
    R = lambda s, k: s.reindex(k).values
    lv = dict(PWH=R(wk['hi'].shift(1),wid), PWL=R(wk['lo'].shift(1),wid),
              PMH=R(mo['hi'].shift(1),mid), PML=R(mo['lo'].shift(1),mid),
              F5H=R(dy['hi'].rolling(5).max().shift(1),did),   # placebo: arbitrary 5d fractal
              F5L=R(dy['lo'].rolling(5).min().shift(1),did))
    tf = dict(none=None,
              wup=R((wk['cl'].shift(1)>wk['cl'].rolling(10).mean().shift(1)),wid),
              wdn=R((wk['cl'].shift(1)<wk['cl'].rolling(10).mean().shift(1)),wid),
              up=R((dy['cl'].shift(1)>dy['cl'].rolling(10).mean().shift(1)),did))
    del df; gc.collect()
    return lv, dict(W=wid,M=mid,D=did), tf

def go(sym,lv,bl,tf,lvl,bid,d_,h,f,tag='',gap=None):
    s = stats(backtest(sym,lv[lvl],bl[bid],d_,h,tf[f],max_gap_min=gap),sym,lvl,h,f,d_)
    s['tag']=tag; return s

R1=[];R2=[];R3=[];R4=[];R5=[]
CAND=['XAUUSDm','EURUSDm','BTCUSDm','USDJPYm','GBPUSDm','US500m','XAGUSDm','USOILm']
for sym in CAND:
    lv,bl,tf = build(sym)
    # 1. weekend-gap guard
    for h in [30,60,120,240]:
        for lvl,bid,d_,f in [('PWH','W',1,'none'),('PWH','W',1,'wup'),('PMH','M',1,'none'),
                             ('PWL','W',-1,'none'),('PML','M',-1,'none')]:
            R1.append(go(sym,lv,bl,tf,lvl,bid,d_,h,f,'canon',None))
            R1.append(go(sym,lv,bl,tf,lvl,bid,d_,h,f,'nogap',h+90))
    # 3. placebo vs real
    for h in [60,240]:
        R3.append(go(sym,lv,bl,tf,'F5H','W',1,h,'none','placebo5d_weekblock'))
        R3.append(go(sym,lv,bl,tf,'PWH','W',1,h,'none','real_PWH'))
        R3.append(go(sym,lv,bl,tf,'F5H','D',1,h,'none','placebo5d_dayblock'))
    # 4. longer holds
    for h in [240,480,720,1440]:
        R4.append(go(sym,lv,bl,tf,'PWH','W',1,h,'none'))
        R4.append(go(sym,lv,bl,tf,'PMH','M',1,h,'none'))
    # 2. week-boundary sensitivity
    for ws in [0,6,24,72]:
        lv2,bl2,tf2 = build(sym,wshift=ws)
        for h in [60,240]:
            for f in ['none','wup']:
                R2.append(go(sym,lv2,bl2,tf2,'PWH','W',1,h,f,f'sh{ws}'))
        del lv2,bl2,tf2; gc.collect()
    for h in [60,240]:
        for f in ['none','wup']:
            R2.append(go(sym,lv,bl,tf,'PWH','W',1,h,f,'sh3'))
    # 5. per-year detail
    for lvl,bid,d_,f,h in [('PWH','W',1,'none',120),('PWH','W',1,'wup',120),('PWH','W',1,'none',240),
                           ('PMH','M',1,'none',120),('PMH','M',1,'none',240),
                           ('PWL','W',-1,'none',240),('PML','M',-1,'none',120)]:
        R5.append(go(sym,lv,bl,tf,lvl,bid,d_,h,f))
    HL._cache.pop(sym,None); del lv,bl,tf; gc.collect()
    print('done',sym,file=sys.stderr,flush=True)

C=['symbol','level','hold','filt','n','gross','cost','net','t','hit','yrs']
print('\n===== 1. WEEKEND-GAP GUARD =====')
r1=pd.DataFrame(R1)
print(r1.pivot_table(index=['symbol','level','filt','hold'],columns='tag',values=['n','net','t']).round(2).to_string())
print('\n===== 2. WEEK-BOUNDARY SENSITIVITY (PWH long) =====')
r2=pd.DataFrame(R2)
print(r2.pivot_table(index=['symbol','filt','hold'],columns='tag',values=['n','net','t']).round(2).to_string())
print('\n===== 3. PLACEBO (arbitrary 5d fractal) vs REAL prior-week high =====')
print(pd.DataFrame(R3)[['symbol','tag','hold','n','gross','cost','net','t','hit','yrs']].round(2).to_string(index=False))
print('\n===== 4. LONGER HOLDS =====')
print(pd.DataFrame(R4)[C].round(2).to_string(index=False))
print('\n===== 5. PER-YEAR DETAIL =====')
for s in R5:
    print(f"{s['symbol']:9s} {s['level']} h={s['hold']:4d} f={s['filt']:5s} n={s['n']:4d} "
          f"gross={s['gross']:+7.2f} cost={s['cost']:5.2f} net={s['net']:+7.2f} t={s['t']:+5.2f} "
          f"hit={s['hit']:.2f} yrs={s['yrs']}  {s.get('yrdetail','')}")
