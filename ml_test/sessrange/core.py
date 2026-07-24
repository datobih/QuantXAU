import numpy as np, pandas as pd, os

DATA = 'c:/Users/David/Downloads/prototype_3-main/prototype_3-main/data/raw/m1_mt5'
POINT={'XAUUSDm':.001,'XAGUSDm':.001,'US30m':.1,'US500m':.01,'USTECm':.01,
       'BTCUSDm':.01,'EURUSDm':1e-5,'GBPUSDm':1e-5,'USDJPYm':.001,'USOILm':.001}
SYMS=list(POINT)

_cache={}
def load(sym):
    if sym in _cache: return _cache[sym]
    d=pd.read_parquet(f'{DATA}/{sym}.parquet')
    d=d.set_index('dt').sort_index(); d=d[~d.index.duplicated()]
    d=d[d.index>='2021-08-01']
    _cache[sym]=d
    return d


def backtest(sym, level_series, block_ids, direction=+1, hold=60, trend_filter=None,
             extra_filter=None, tag_series=None):
    """Canonical harness (same entry/cost/exit logic as the reference).
    block_ids: -1 (or any negative) marks bars NOT eligible to be part of a block.
    level_series: level active at each bar, aligned to d.index.
    """
    d=load(sym); pt=POINT[sym]
    idx=d.index; O,H,L,C=(d[c].values for c in ('open','high','low','close'))
    SPRD=d['spread'].values*pt
    med=np.median(SPRD[SPRD>0]) if (SPRD>0).any() else pt*10
    slip=med/3
    ex=np.clip(idx.searchsorted(idx+pd.Timedelta(minutes=hold)),0,len(d)-1)
    b=np.asarray(block_ids)
    ch=np.r_[0,np.where(b[1:]!=b[:-1])[0]+1,len(b)]
    out=[]
    for k in range(len(ch)-1):
        s0,s1=ch[k],ch[k+1]
        if b[s0]<0: continue
        sel=np.arange(s0,s1)
        if len(sel)<20: continue          # need a real window
        P=level_series[sel[0]]
        if not np.isfinite(P): continue
        if trend_filter is not None and not bool(trend_filter[sel[0]]): continue
        if extra_filter is not None and not bool(extra_filter[sel[0]]): continue
        if direction>0:
            if C[sel[0]]>=P: continue
            t=sel[H[sel]>=P]
        else:
            if C[sel[0]]<=P: continue
            t=sel[L[sel]<=P]
        if len(t)==0: continue
        i=t[0]
        if i>=len(d)-1: continue
        px=(P if O[i]<=P else O[i]) if direction>0 else (P if O[i]>=P else O[i])
        cost=(SPRD[i] if SPRD[i]>0 else med)+slip
        raw=(C[ex[i]]-px) if direction>0 else (px-C[ex[i]])
        r={'ts':idx[i],'year':idx[i].year,'entry':px,'net':raw-cost,
           'bp':(raw-cost)/px*1e4,'gross_bp':raw/px*1e4,'cost_bp':cost/px*1e4}
        if tag_series is not None: r['tag']=tag_series[sel[0]]
        out.append(r)
    return pd.DataFrame(out)


def stats(tr, min_yr_n=15):
    if len(tr)<10:
        return dict(n=len(tr),net=np.nan,gross=np.nan,cost=np.nan,t=np.nan,hit=np.nan,
                    yrpos='', yrs='')
    x=tr['bp'].values
    t=x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if x.std(ddof=1)>0 else np.nan
    g=tr.groupby('year')['bp'].agg(['mean','size'])
    g=g[g['size']>=min_yr_n]
    npos=int((g['mean']>0).sum()); ntot=len(g)
    return dict(n=len(tr),net=round(x.mean(),2),gross=round(tr['gross_bp'].mean(),2),
                cost=round(tr['cost_bp'].mean(),2),t=round(float(t),2),
                hit=round(float((x>0).mean()),3),
                yrpos=f'{npos}/{ntot}',
                yrs=' '.join(f"{int(y)}:{m:+.1f}" for y,m in g['mean'].items()))


def sessions(sym):
    """Return dict of per-bar arrays describing sessions / levels."""
    d=load(sym); idx=d.index
    day=idx.normalize()
    H,L,C=d['high'].values,d['low'].values,d['close'].values
    hr=idx.hour+idx.minute/60.0
    et=idx.tz_localize('UTC').tz_convert('America/New_York')
    return dict(d=d,idx=idx,day=day,H=H,L=L,C=C,hr=hr,et=et)


def range_level(sym, rng_start, rng_end, brk_start, brk_end):
    """Range built over [rng_start,rng_end) UTC hours, broken over [brk_start,brk_end).
    Returns (hi_arr, lo_arr, block_ids, width_bp_arr) aligned to bars."""
    S=sessions(sym); d=S['d']; idx=S['idx']; hr=S['hr']
    dkey=idx.normalize()
    inr=(hr>=rng_start)&(hr<rng_end)
    inb=(hr>=brk_start)&(hr<brk_end)
    g=d[inr].groupby(dkey[inr])
    hi=g['high'].max(); lo=g['low'].min()
    ndays=g.size()
    ok=ndays>=int((rng_end-rng_start)*60*0.5)     # need >=50% of the range session present
    hi=hi.where(ok); lo=lo.where(ok)
    hia=hi.reindex(dkey).values; loa=lo.reindex(dkey).values
    mid=(hia+loa)/2
    width=(hia-loa)/mid*1e4
    # block id: one block per day over break window; -1 elsewhere
    dnum=pd.factorize(dkey)[0].astype(np.int64)
    blk=np.where(inb, dnum, -1)
    return hia,loa,blk,width


def us_overnight_level(sym):
    """Overnight/pre-open range 22:00 UTC prev day -> US cash open (09:30 ET),
    broken 09:30 ET -> 16:00 ET."""
    S=sessions(sym); d=S['d']; idx=S['idx']; et=S['et']
    ehr=et.hour+et.minute/60.0
    eday=pd.DatetimeIndex(et.normalize().tz_localize(None))
    inr=(ehr>=17.0)|(ehr<9.5)          # 17:00 ET prev -> 09:30 ET
    inb=(ehr>=9.5)&(ehr<16.0)
    # session-day key: bars at/after 17:00 ET belong to the NEXT ET calendar day
    skey=eday+pd.to_timedelta(np.where(ehr>=17.0,1,0),unit='D')
    g=d[inr].groupby(skey[inr])
    hi=g['high'].max(); lo=g['low'].min(); n=g.size()
    ok=n>=200
    hi=hi.where(ok); lo=lo.where(ok)
    hia=hi.reindex(skey).values; loa=lo.reindex(skey).values
    mid=(hia+loa)/2; width=(hia-loa)/mid*1e4
    dnum=pd.factorize(skey)[0].astype(np.int64)
    blk=np.where(inb,dnum,-1)
    return hia,loa,blk,width


def daily_trend(sym):
    """(tf_up, tf_dn, pdh) per-bar arrays. tf_up: yday close > 10d SMA."""
    d=load(sym); idx=d.index
    daily=d.resample('1D').agg(hi=('high','max'),lo=('low','min'),cl=('close','last')).dropna()
    sma=daily['cl'].rolling(10).mean()
    up=(daily['cl'].shift(1)>sma.shift(1))
    tfu=up.reindex(idx.normalize()).values
    tfd=(~up).where(up.notna()).reindex(idx.normalize()).values
    pdh=daily['hi'].shift(1).reindex(idx.normalize()).values
    return np.nan_to_num(tfu,nan=0).astype(bool), np.nan_to_num(tfd,nan=0).astype(bool), pdh


def width_tercile(width, day_key):
    """Trailing-window tercile rank of range width (no lookahead).
    Returns per-bar int 0/1/2 (0=narrowest), -1 unknown."""
    s=pd.Series(width,index=day_key)
    dw=s.groupby(level=0).first()
    q1=dw.shift(1).rolling(250,min_periods=60).quantile(1/3)
    q2=dw.shift(1).rolling(250,min_periods=60).quantile(2/3)
    t=pd.Series(-1,index=dw.index,dtype=int)
    t[(dw<=q1)]=0; t[(dw>q1)&(dw<=q2)]=1; t[(dw>q2)]=2
    t[q1.isna()]=-1
    return t.reindex(day_key).values
