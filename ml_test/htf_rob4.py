"""LEVEL-SPECIFICITY test: is the prior-week high special, or is any level at that
distance equally good? Displace the level by +/- k * prior-week-range and re-run.
If PWH is a 'watched level', breaking PWH should beat breaking PWH +/- 0.3R.
If it is generic breakout momentum, the displaced levels do just as well."""
import numpy as np, pandas as pd, gc
import htf_levels as HL
from htf_levels import backtest, stats

EPOCH_MON = np.datetime64('1970-01-05')

def build(sym):
    d = HL.load(sym); idx = d.index
    t = (idx + pd.Timedelta(hours=3)).values.astype('datetime64[m]')
    wid = ((t-EPOCH_MON).astype('timedelta64[m]').astype(np.int64)//(7*1440))
    did = idx.normalize().values.astype('datetime64[D]').astype(np.int64)
    df = pd.DataFrame({'hi':d['high'].values,'lo':d['low'].values,'cl':d['close'].values,'w':wid,'d':did})
    wk = df.groupby('w',sort=True).agg(hi=('hi','max'),lo=('lo','min'),cl=('cl','last'))
    dy = df.groupby('d',sort=True).agg(hi=('hi','max'),lo=('lo','min'))
    R = lambda s,k: s.reindex(k).values
    pwh, pwl = wk['hi'].shift(1), wk['lo'].shift(1)
    rng = (pwh - pwl)
    lv = {}
    for k in [-0.5,-0.3,-0.15,0.0,0.15,0.3,0.5]:
        lv[f'{k:+.2f}'] = R(pwh + k*rng, wid)
    tf = dict(none=None, wup=R(wk['cl'].shift(1)>wk['cl'].rolling(10).mean().shift(1),wid))
    del df; gc.collect(); return lv, wid, tf

print('LEVEL-SPECIFICITY: level = prior-week-high + k*(prior-week range). k=0 is the real PWH.')
print('If the real level is special, k=0 should stand out. Flat profile => generic momentum.\n')
for sym in ['XAUUSDm','EURUSDm','BTCUSDm']:
    lv, wid, tf = build(sym)
    print(f'--- {sym} ---')
    for h in [60,240]:
        for f in ['none']:
            print(f'  hold={h}m filt={f}')
            for k in ['-0.50','-0.30','-0.15','+0.00','+0.15','+0.30','+0.50']:
                s = stats(backtest(sym, lv[k], wid, +1, h, tf[f]), sym, k, h, f, +1)
                star = '  <== REAL PWH' if k=='+0.00' else ''
                print(f'    k={k}  n={s["n"]:4d} gross={s["gross"]:+7.2f} cost={s["cost"]:5.2f} '
                      f'net={s["net"]:+7.2f} t={s["t"]:+5.2f} hit={s["hit"]:.2f} yrs={s["yrs"]}{star}',flush=True)
    HL._cache.pop(sym,None); del lv,wid,tf; gc.collect()
