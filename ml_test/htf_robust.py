"""Robustness on the HTF-level candidates."""
import numpy as np, pandas as pd
from htf_levels import load, backtest, stats, POINT, SYMS

pd.set_option('display.width', 260, 'display.max_rows', 900)

def build(sym, wshift=3):
    d = load(sym); idx = d.index
    sh = idx + pd.Timedelta(hours=wshift)
    wid = pd.Index(sh.strftime('%G-%V')); mid = pd.Index(sh.strftime('%Y-%m'))
    did = pd.Index(idx.normalize())
    df = pd.DataFrame({'hi':d['high'].values,'lo':d['low'].values,'cl':d['close'].values,
                       'w':wid,'m':mid,'d':did})
    wk = df.groupby('w',sort=True).agg(hi=('hi','max'),lo=('lo','min'),cl=('cl','last'))
    mo = df.groupby('m',sort=True).agg(hi=('hi','max'),lo=('lo','min'),cl=('cl','last'))
    dy = df.groupby('d',sort=True).agg(hi=('hi','max'),lo=('lo','min'),cl=('cl','last'))
    lv = dict(PWH=wk['hi'].shift(1).reindex(wid).values, PWL=wk['lo'].shift(1).reindex(wid).values,
              PMH=mo['hi'].shift(1).reindex(mid).values, PML=mo['lo'].shift(1).reindex(mid).values)
    # PLACEBO: rolling 5-day high as of the day before the week starts (arbitrary, not week-aligned).
    # Anchored to the block's first bar so it is a fixed level for the whole week, like PWH.
    r5 = dy['hi'].rolling(5).max().shift(1)
    lv['F5H'] = r5.reindex(did).values     # daily-refreshed 5d fractal
    wsma = wk['cl'].rolling(10).mean()
    tf = dict(none=None,
              wup=(wk['cl'].shift(1)>wsma.shift(1)).reindex(wid).values,
              wdn=(wk['cl'].shift(1)<wsma.shift(1)).reindex(wid).values,
              up=(dy['cl'].shift(1)>dy['cl'].rolling(10).mean().shift(1)).reindex(did).values)
    return lv, dict(W=wid.values,M=mid.values,D=did.values), tf

def row(sym,lvl,bid,d_,h,f,tag,gap=None,lv=None,bl=None,tf=None):
    tr = backtest(sym, lv[lvl], bl[bid], d_, h, tf[f], max_gap_min=gap)
    s = stats(tr,sym,lvl,h,f,d_); s['tag']=tag
    return s

print('='*30,'1. WEEKEND-GAP GUARD (exit must be <= hold+90min after entry)','='*30)
rows=[]
for sym in ['XAUUSDm','EURUSDm','BTCUSDm','USDJPYm','GBPUSDm']:
    lv,bl,tf = build(sym)
    for h in [30,60,120,240]:
        for lvl,bid,d_,f in [('PWH','W',1,'none'),('PWH','W',1,'wup'),('PMH','M',1,'none'),
                             ('PWL','W',-1,'none'),('PML','M',-1,'none')]:
            for gap,tag in [(None,'canon'),(h+90,'nogap')]:
                rows.append(row(sym,lvl,bid,d_,h,f,tag,gap,lv,bl,tf))
r1=pd.DataFrame(rows)
p=r1.pivot_table(index=['symbol','level','filt','hold'],columns='tag',values=['n','net','t'])
print(p.round(2).to_string())

print()
print('='*30,'2. WEEK-BOUNDARY SENSITIVITY (shift 0h / 3h / 6h / 24h=Tue-anchored)','='*30)
rows=[]
for sym in ['XAUUSDm','EURUSDm','BTCUSDm','USDJPYm','US500m']:
    for ws in [0,3,6,24]:
        lv,bl,tf = build(sym,wshift=ws)
        for h in [60,240]:
            for f in ['none','wup']:
                s=row(sym,'PWH','W',1,h,f,f'sh{ws}',None,lv,bl,tf); rows.append(s)
r2=pd.DataFrame(rows)
print(r2.pivot_table(index=['symbol','filt','hold'],columns='tag',values=['n','net','t']).round(2).to_string())

print()
print('='*30,'3. PLACEBO: 5-day rolling fractal high, one trade/week block (not a watched level)','='*30)
rows=[]
for sym in ['XAUUSDm','EURUSDm','BTCUSDm','USDJPYm','GBPUSDm','US500m']:
    lv,bl,tf = build(sym)
    for h in [60,240]:
        rows.append(row(sym,'F5H','W',1,h,'none','placebo5d',None,lv,bl,tf))
        rows.append(row(sym,'PWH','W',1,h,'none','realPWH',None,lv,bl,tf))
        # 5d fractal on DAILY blocks (one trade/day) - direct comparison to PDH
        rows.append(row(sym,'F5H','D',1,h,'none','placebo5d_daily',None,lv,bl,tf))
r3=pd.DataFrame(rows)
print(r3[['symbol','tag','hold','n','gross','cost','net','t','hit','yrs']].round(2).to_string(index=False))

print()
print('='*30,'4. LONGER HOLDS on high-cost symbols (gross must outgrow spread)','='*30)
rows=[]
for sym in ['BTCUSDm','XAGUSDm','USOILm','XAUUSDm','EURUSDm']:
    lv,bl,tf = build(sym)
    for h in [240,480,720,1440]:
        rows.append(row(sym,'PWH','W',1,h,'none','',None,lv,bl,tf))
        rows.append(row(sym,'PMH','M',1,h,'none','',None,lv,bl,tf))
r4=pd.DataFrame(rows)
print(r4[['symbol','level','hold','n','gross','cost','net','t','hit','yrs']].round(2).to_string(index=False))

print()
print('='*30,'5. PER-YEAR DETAIL for candidates','='*30)
for sym,lvl,bid,d_,f,h in [('XAUUSDm','PMH','M',1,'none',240),('XAUUSDm','PWH','W',1,'wup',60),
                           ('EURUSDm','PWH','W',1,'wup',120),('EURUSDm','PWH','W',1,'none',120),
                           ('EURUSDm','PWL','W',-1,'none',240),('EURUSDm','PML','M',-1,'none',120),
                           ('GBPUSDm','PML','M',-1,'none',120),('BTCUSDm','PWH','W',1,'none',240),
                           ('BTCUSDm','PMH','M',1,'none',120),('USDJPYm','PMH','M',1,'none',120),
                           ('USTECm','PMH','M',1,'up',120),('XAGUSDm','PMH','M',1,'none',240)]:
    lv,bl,tf = build(sym)
    s=row(sym,lvl,bid,d_,h,f,'',None,lv,bl,tf)
    print(f"{sym:9s} {lvl} h={h:4d} f={f:5s} n={s['n']:4d} gross={s['gross']:+7.2f} cost={s['cost']:5.2f} "
          f"net={s['net']:+7.2f} t={s['t']:+5.2f} hit={s['hit']:.2f} yrs={s['yrs']}  {s['yrdetail']}")
