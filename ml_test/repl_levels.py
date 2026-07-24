"""
Clean-room replication: do human-charted key levels produce breakout edges outside gold?
Independent implementation. Levels: prior-day high, prior-week high, Asian-session-range high.
Long side, 60m hold, 10d-SMA uptrend filter, + random-entry controls matched on day and hold.
"""
import numpy as np, pandas as pd, sys, json

DATA = 'c:/Users/David/Downloads/prototype_3-main/prototype_3-main/data/raw/m1_mt5/{}.parquet'
POINT = {'XAUUSDm':.001,'XAGUSDm':.001,'US30m':.1,'US500m':.01,'USTECm':.01,
         'BTCUSDm':.01,'EURUSDm':1e-5,'GBPUSDm':1e-5,'USDJPYm':.001,'USOILm':.001}
SYMS = list(POINT)
START = '2021-08-01'

_cache = {}
def load(sym):
    if sym in _cache: return _cache[sym]
    d = pd.read_parquet(DATA.format(sym)).set_index('dt').sort_index()
    d = d[~d.index.duplicated()]
    d = d[d.index >= START]
    _cache[sym] = d
    return d


def prep(sym):
    """Returns dict of arrays + daily frame."""
    d = load(sym); pt = POINT[sym]
    idx = d.index
    O,H,L,C = (d[c].values.astype(float) for c in ('open','high','low','close'))
    SPRD = d['spread'].values.astype(float)*pt
    pos = SPRD[SPRD>0]
    med = np.median(pos) if len(pos) else pt*10
    SPRD = np.where(SPRD>0, SPRD, med)
    slip = med/3.0
    # exit index = first bar at/after entry+hold; keep the realised gap so we can veto weekend jumps
    return dict(d=d, idx=idx, O=O,H=H,L=L,C=C, SPRD=SPRD, med=med, slip=slip, pt=pt)


def exit_idx(idx, hold):
    e = idx.searchsorted(idx + pd.Timedelta(minutes=hold))
    e = np.clip(e, 0, len(idx)-1)
    gap = (idx[e] - idx).total_seconds()/60.0
    return e, gap.values


def daily_frame(sym):
    d = load(sym)
    dl = d.resample('1D').agg(hi=('high','max'), lo=('low','min'),
                              cl=('close','last'), op=('open','first')).dropna()
    dl['sma10'] = dl['cl'].rolling(10).mean()
    return dl


# ---------------- level constructors ------------------------------------
def build_pdh(sym):
    d = load(sym); idx = d.index
    dl = daily_frame(sym)
    key = pd.Index(idx.normalize())
    lvl = dl['hi'].shift(1).reindex(key).values
    tf  = (dl['cl'].shift(1) > dl['sma10'].shift(1)).reindex(key).values
    blocks = key.values
    return lvl, blocks, tf, np.ones(len(idx), bool)

def build_pwh(sym):
    d = load(sym); idx = d.index
    wk = d.resample('W-MON', label='left', closed='left').agg(hi=('high','max')).dropna()
    dl = daily_frame(sym)
    wkey = pd.Index(idx.to_period('W-SUN').start_time)   # ISO weeks, Monday start
    whi = d.groupby(wkey)['high'].max()
    lvl = whi.shift(1).reindex(wkey).values
    key = pd.Index(idx.normalize())
    tf  = (dl['cl'].shift(1) > dl['sma10'].shift(1)).reindex(key).values
    return lvl, wkey.values, tf, np.ones(len(idx), bool)

ASIA_END = 7   # UTC hour: Asian session = 00:00-07:00 UTC, trade window 07:00-21:00
def build_asian(sym, side='high'):
    d = load(sym); idx = d.index
    day = pd.Index(idx.normalize())
    hr = idx.hour
    asia = hr < ASIA_END
    sub = d[asia]
    g = sub.groupby(pd.Index(sub.index.normalize()))
    a_hi = g['high'].max(); a_lo = g['low'].min()
    lvl = (a_hi if side=='high' else a_lo).reindex(day).values
    dl = daily_frame(sym)
    tf = (dl['cl'].shift(1) > dl['sma10'].shift(1)).reindex(day).values
    # blocks = day, but only bars in the trade window are eligible
    win = (hr >= ASIA_END) & (hr < 21)
    return lvl, day.values, tf, win


# ---------------- backtest ----------------------------------------------
RNG = np.random.default_rng(20260724)

def run(sym, level, blocks, tf, window, direction=+1, hold=60, n_rand=200):
    P = prep(sym)
    idx, O,H,L,C, SPRD = P['idx'], P['O'],P['H'],P['L'],P['C'], P['SPRD']
    med, slip = P['med'], P['slip']
    ex, gap = exit_idx(idx, hold)
    b = np.asarray(blocks)
    ch = np.r_[0, np.where(b[1:] != b[:-1])[0]+1, len(b)]
    trades, elig = [], []          # elig: (start,end) of blocks that passed all pre-trade filters
    for k in range(len(ch)-1):
        s0, s1 = ch[k], ch[k+1]
        sel = np.arange(s0, s1)
        sel = sel[window[sel]]
        if len(sel) < hold + 2: continue
        lv = level[sel[0]]
        if not np.isfinite(lv): continue
        if tf is not None:
            t0 = tf[sel[0]]
            if t0 is None or (isinstance(t0,float) and np.isnan(t0)) or not bool(t0): continue
        if direction > 0:
            if C[sel[0]] >= lv: continue
            hit = sel[H[sel] >= lv]
        else:
            if C[sel[0]] <= lv: continue
            hit = sel[L[sel] <= lv]
        elig.append((sel[0], sel[-1]))
        if len(hit) == 0: continue
        i = hit[0]
        if i >= len(idx)-1: continue
        if gap[i] > 3*hold: continue              # weekend / session-gap veto
        px = (lv if O[i] <= lv else O[i]) if direction > 0 else (lv if O[i] >= lv else O[i])
        cost = SPRD[i] + slip
        raw = (C[ex[i]] - px) if direction > 0 else (px - C[ex[i]])
        trades.append(dict(ts=idx[i], year=idx[i].year, i=i, blk=k,
                           off=i - sel[0], b0=sel[0], b1=sel[-1],
                           entry=px, gross_bp=raw/px*1e4,
                           cost_bp=cost/px*1e4, net_bp=(raw-cost)/px*1e4))
    tr = pd.DataFrame(trades)
    if len(tr) == 0:
        return tr, np.nan, np.nan

    # ---- control 1: random minute inside the SAME block (day-drift benchmark)
    r1 = []
    for _, t in tr.iterrows():
        lo, hi = int(t.b0), int(t.b1)
        cand = np.arange(lo, hi+1)
        cand = cand[gap[cand] <= 3*hold]
        if len(cand) == 0: r1.append(np.nan); continue
        j = RNG.choice(cand, size=min(n_rand, len(cand)), replace=len(cand) < n_rand)
        g = (C[ex[j]] - O[j])/O[j]*1e4
        r1.append(np.nanmean(g))
    tr['rand_same_day_bp'] = r1

    # ---- control 2: same clock-offset, a DIFFERENT eligible block (time-of-day matched)
    starts = np.array([e[0] for e in elig]); ends = np.array([e[1] for e in elig])
    r2 = []
    for _, t in tr.iterrows():
        off = int(t.off)
        ok = np.where((ends - starts) >= off)[0]
        if len(ok) < 5: r2.append(np.nan); continue
        pick = RNG.choice(ok, size=min(n_rand, len(ok)), replace=len(ok) < n_rand)
        j = starts[pick] + off
        j = j[(j < len(idx)-1) & (gap[j] <= 3*hold)]
        if len(j) == 0: r2.append(np.nan); continue
        g = (C[ex[j]] - O[j])/O[j]*1e4
        r2.append(np.nanmean(g))
    tr['rand_tod_bp'] = r2
    return tr, np.nanmean(r1), np.nanmean(r2)


def summarise(sym, lname, tr, r1, r2):
    if len(tr) == 0:
        return dict(symbol=sym, level=lname, n=0)
    n = len(tr)
    m = tr.net_bp.mean(); se = tr.net_bp.std(ddof=1)/np.sqrt(n)
    yr = tr.groupby('year').net_bp.agg(['mean','count'])
    yr = yr[yr['count'] >= 10]
    yrs_pos = int((yr['mean'] > 0).sum()); yrs_tot = len(yr)
    # excess over time-of-day matched control, per-trade paired t
    exc = (tr.gross_bp - tr.rand_tod_bp).dropna()
    exc_t = exc.mean()/(exc.std(ddof=1)/np.sqrt(len(exc))) if len(exc) > 3 else np.nan
    return dict(symbol=sym, level=lname, n=n,
                gross_bp=tr.gross_bp.mean(), cost_bp=tr.cost_bp.mean(), net_bp=m,
                t=m/se if se > 0 else np.nan,
                hit=(tr.net_bp > 0).mean()*100,
                rand_day_bp=r1, rand_tod_bp=r2,
                excess_vs_tod=tr.gross_bp.mean() - (r2 if np.isfinite(r2) else np.nan),
                excess_t=exc_t,
                years_pos=f'{yrs_pos}/{yrs_tot}',
                yr_means='|'.join(f'{y}:{v:+.1f}' for y,v in yr['mean'].items()))


LEVELS = {'PDH': build_pdh, 'PWH': build_pwh, 'ASIAN_HI': build_asian}

if __name__ == '__main__':
    hold = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rows = []
    for sym in SYMS:
        for lname, fn in LEVELS.items():
            lvl, blk, tf, win = fn(sym)
            tr, r1, r2 = run(sym, lvl, blk, tf, win, direction=+1, hold=hold)
            rows.append(summarise(sym, lname, tr, r1, r2))
            print('.', end='', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'repl_levels_h{hold}.csv', index=False)
    print()
    pd.set_option('display.width', 250, 'display.max_columns', 40)
    print(df[['symbol','level','n','gross_bp','cost_bp','net_bp','t','hit',
              'rand_day_bp','rand_tod_bp','excess_vs_tod','excess_t','years_pos']]
          .round(2).to_string(index=False))
    print()
    print(df[['symbol','level','yr_means']].to_string(index=False))
