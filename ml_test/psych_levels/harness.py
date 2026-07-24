import numpy as np, pandas as pd, os

DATA='c:/Users/David/Downloads/prototype_3-main/prototype_3-main/data/raw/m1_mt5/{}.parquet'
POINT={'XAUUSDm':.001,'XAGUSDm':.001,'US30m':.1,'US500m':.01,'USTECm':.01,
       'BTCUSDm':.01,'EURUSDm':1e-5,'GBPUSDm':1e-5,'USDJPYm':.001,'USOILm':.001}
SYMS=list(POINT)
_C={}
def load(sym):
    if sym in _C: return _C[sym]
    d=pd.read_parquet(DATA.format(sym))
    d=d.set_index('dt').sort_index(); d=d[~d.index.duplicated()]
    d=d[d.index>='2021-08-01']
    _C[sym]=d
    return d

_D={}
def daily(sym):
    """daily OHLC + derived series reindexed onto the M1 index."""
    if sym in _D: return _D[sym]
    d=load(sym); idx=d.index
    dl=d.resample('1D').agg(op=('open','first'),hi=('high','max'),
                            lo=('low','min'),cl=('close','last')).dropna()
    n=idx.normalize()
    sma=dl['cl'].rolling(10).mean()
    out={}
    out['up']  =(dl['cl'].shift(1)>sma.shift(1)).reindex(n).values
    out['down']=(dl['cl'].shift(1)<sma.shift(1)).reindex(n).values
    out['pdh'] =dl['hi'].shift(1).reindex(n).values
    out['pdl'] =dl['lo'].shift(1).reindex(n).values
    out['pdc'] =dl['cl'].shift(1).reindex(n).values
    for k in (3,5,10):
        out[f'h{k}']=dl['hi'].rolling(k).max().shift(1).reindex(n).values
        out[f'l{k}']=dl['lo'].rolling(k).min().shift(1).reindex(n).values
    out['dayid']=pd.factorize(n)[0]
    _D[sym]=out
    return out

def backtest(sym, level_series, block_ids, direction=+1, hold=60, trend_filter=None):
    d=load(sym); pt=POINT[sym]
    idx=d.index; O,H,L,C=(d[c].values for c in ('open','high','low','close'))
    SPRD=d['spread'].values*pt
    med=np.median(SPRD[SPRD>0]) if (SPRD>0).any() else pt*10
    slip=med/3
    ex=np.clip(idx.searchsorted(idx+pd.Timedelta(minutes=hold)),0,len(d)-1)
    b=np.asarray(block_ids); ch=np.r_[0,np.where(b[1:]!=b[:-1])[0]+1,len(b)]
    out=[]
    for k in range(len(ch)-1):
        sel=np.arange(ch[k],ch[k+1])
        if len(sel)<hold+2: continue
        P=level_series[sel[0]]
        if np.isnan(P): continue
        if trend_filter is not None and not bool(trend_filter[sel[0]]): continue
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
        out.append({'ts':idx[i],'year':idx[i].year,'entry':px,'net':raw-cost,
                    'bp':(raw-cost)/px*1e4,'gross_bp':raw/px*1e4,'cost_bp':cost/px*1e4})
    return pd.DataFrame(out)

def stat(tr):
    if len(tr)<10: return dict(n=len(tr),net=np.nan,gross=np.nan,cost=np.nan,t=np.nan,hit=np.nan,yrs='')
    b=tr['bp'].values
    t=b.mean()/(b.std(ddof=1)/np.sqrt(len(b)))
    yr=tr.groupby('year')['bp'].mean()
    return dict(n=len(b),net=b.mean(),gross=tr['gross_bp'].mean(),cost=tr['cost_bp'].mean(),
                t=t,hit=(b>0).mean(),yrs=f"{(yr>0).sum()}/{len(yr)}",
                yrd={int(k):round(v,2) for k,v in yr.items()})
