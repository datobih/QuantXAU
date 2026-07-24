"""Higher-timeframe watched levels: prior-week / prior-month high & low breakouts, 10 symbols."""
import numpy as np, pandas as pd, sys

BASE = 'c:/Users/David/Downloads/prototype_3-main/prototype_3-main/data/raw/m1_mt5'
POINT = {'XAUUSDm':.001,'XAGUSDm':.001,'US30m':.1,'US500m':.01,'USTECm':.01,
         'BTCUSDm':.01,'EURUSDm':1e-5,'GBPUSDm':1e-5,'USDJPYm':.001,'USOILm':.001}
SYMS = list(POINT)

_cache = {}
def load(sym):
    if sym in _cache: return _cache[sym]
    d = pd.read_parquet(f'{BASE}/{sym}.parquet')
    d = d.set_index('dt').sort_index(); d = d[~d.index.duplicated()]
    d = d[d.index >= '2021-08-01']
    _cache[sym] = d
    return d


def backtest(sym, level_series, block_ids, direction=+1, hold=60, trend_filter=None,
             max_gap_min=None):
    """Canonical harness. max_gap_min: if set, skip trades whose exit bar is more than
    that many minutes after entry (weekend/holiday gap guard). None = canonical behaviour."""
    d = load(sym); pt = POINT[sym]
    idx = d.index
    O,H,L,C = (d[c].values for c in ('open','high','low','close'))
    SPRD = d['spread'].values * pt
    med = np.median(SPRD[SPRD>0]) if (SPRD>0).any() else pt*10
    slip = med/3
    ex = np.clip(idx.searchsorted(idx + pd.Timedelta(minutes=hold)), 0, len(d)-1)
    tsv = idx.values.astype('datetime64[m]').astype(np.int64)
    b = np.asarray(block_ids)
    ch = np.r_[0, np.where(b[1:] != b[:-1])[0]+1, len(b)]
    out = []
    for k in range(len(ch)-1):
        sel = np.arange(ch[k], ch[k+1])
        if len(sel) < hold+2: continue
        P = level_series[sel[0]]
        if np.isnan(P): continue
        if trend_filter is not None and not bool(trend_filter[sel[0]]): continue
        if direction > 0:
            if C[sel[0]] >= P: continue
            t = sel[H[sel] >= P]
        else:
            if C[sel[0]] <= P: continue
            t = sel[L[sel] <= P]
        if len(t) == 0: continue
        i = t[0]
        if i >= len(d)-1: continue
        if max_gap_min is not None and (tsv[ex[i]] - tsv[i]) > max_gap_min: continue
        px = (P if O[i] <= P else O[i]) if direction > 0 else (P if O[i] >= P else O[i])
        cost = (SPRD[i] if SPRD[i] > 0 else med) + slip
        raw = (C[ex[i]] - px) if direction > 0 else (px - C[ex[i]])
        out.append({'ts':idx[i], 'year':idx[i].year,
                    'entry':px, 'net':raw-cost, 'bp':(raw-cost)/px*1e4,
                    'gross_bp':raw/px*1e4, 'cost_bp':cost/px*1e4})
    return pd.DataFrame(out)


# ---------- level builders ----------
def blocks_and_levels(sym):
    """Returns dict of named (level_array, block_id_array) plus trend filter."""
    d = load(sym); idx = d.index
    sh = idx + pd.Timedelta(hours=3)          # trading week/month starts Sun ~21:00 UTC
    wid = pd.Index(sh.strftime('%G-%V'))
    mid = pd.Index(sh.strftime('%Y-%m'))
    did = pd.Index(idx.normalize())

    hi = d['high'].values; lo = d['low'].values; cl = d['close'].values
    df = pd.DataFrame({'hi':hi,'lo':lo,'cl':cl,'w':wid,'m':mid,'d':did})

    wk = df.groupby('w', sort=True).agg(hi=('hi','max'), lo=('lo','min'), cl=('cl','last'))
    mo = df.groupby('m', sort=True).agg(hi=('hi','max'), lo=('lo','min'), cl=('cl','last'))
    dy = df.groupby('d', sort=True).agg(hi=('hi','max'), lo=('lo','min'), cl=('cl','last'))

    lv = {}
    lv['PWH'] = wk['hi'].shift(1).reindex(wid).values
    lv['PWL'] = wk['lo'].shift(1).reindex(wid).values
    lv['PMH'] = mo['hi'].shift(1).reindex(mid).values
    lv['PML'] = mo['lo'].shift(1).reindex(mid).values
    lv['PDH'] = dy['hi'].shift(1).reindex(did).values
    lv['PDL'] = dy['lo'].shift(1).reindex(did).values

    bl = {'W': wid.values, 'M': mid.values, 'D': did.values}

    # daily 10-SMA uptrend / downtrend filter
    sma = dy['cl'].rolling(10).mean()
    tf_up = (dy['cl'].shift(1) > sma.shift(1)).reindex(did).values
    tf_dn = (dy['cl'].shift(1) < sma.shift(1)).reindex(did).values
    # weekly 10-week SMA trend filter
    wsma = wk['cl'].rolling(10).mean()
    tfw_up = (wk['cl'].shift(1) > wsma.shift(1)).reindex(wid).values
    tfw_dn = (wk['cl'].shift(1) < wsma.shift(1)).reindex(wid).values
    return lv, bl, {'up':tf_up, 'dn':tf_dn, 'wup':tfw_up, 'wdn':tfw_dn, 'none':None}


def stats(tr, sym, level, hold, filt, direction):
    if len(tr) < 5:
        return dict(symbol=sym, level=level, hold=hold, filt=filt, dir=direction,
                    n=len(tr), gross=np.nan, cost=np.nan, net=np.nan, t=np.nan,
                    hit=np.nan, yrs='', net_usd=np.nan)
    bp = tr['bp'].values
    t = bp.mean()/ (bp.std(ddof=1)/np.sqrt(len(bp)))
    yr = tr.groupby('year')['bp'].mean()
    npos = int((yr > 0).sum())
    return dict(symbol=sym, level=level, hold=hold, filt=filt, dir=direction,
                n=len(tr), gross=tr['gross_bp'].mean(), cost=tr['cost_bp'].mean(),
                net=bp.mean(), t=t, hit=float((bp > 0).mean()),
                yrs=f'{npos}/{len(yr)}', net_usd=tr['net'].mean(),
                yrdetail=';'.join(f'{y}:{v:+.1f}' for y,v in yr.items()))


if __name__ == '__main__':
    rows = []
    HOLDS = [30, 60, 120, 240]
    for sym in SYMS:
        lv, bl, tfs = blocks_and_levels(sym)
        for h in HOLDS:
            # control: PDH long with uptrend filter
            rows.append(stats(backtest(sym, lv['PDH'], bl['D'], +1, h, tfs['up']),
                              sym, 'PDH', h, 'up', +1))
            # prior-week high long, no filter / daily uptrend / weekly uptrend
            for fn in ['none', 'up', 'wup']:
                rows.append(stats(backtest(sym, lv['PWH'], bl['W'], +1, h, tfs[fn]),
                                  sym, 'PWH', h, fn, +1))
            # prior-month high long
            for fn in ['none', 'up']:
                rows.append(stats(backtest(sym, lv['PMH'], bl['M'], +1, h, tfs[fn]),
                                  sym, 'PMH', h, fn, +1))
            # prior-week low short
            for fn in ['none', 'dn']:
                rows.append(stats(backtest(sym, lv['PWL'], bl['W'], -1, h, tfs[fn]),
                                  sym, 'PWL', h, fn, -1))
            # prior-month low short
            rows.append(stats(backtest(sym, lv['PML'], bl['M'], -1, h, tfs['none']),
                              sym, 'PML', h, 'none', -1))
        print(sym, 'done', file=sys.stderr)
    r = pd.DataFrame(rows)
    r.to_csv('C:/Users/David/AppData/Local/Temp/claude/c--Users-David-Downloads-prototype-3-main-prototype-3-main/ae1e8ac7-ab07-4cf6-a3a3-2fa1058d9b75/scratchpad/htf_main.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_rows', 900)
    print(r[['symbol','level','hold','filt','n','gross','cost','net','t','hit','yrs']].round(2).to_string())
