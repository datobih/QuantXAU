"""UNCONDITIONAL DRIFT CONTROL.
A long-only breakout trade held H minutes earns the instrument's unconditional drift for free.
Report mean H-minute return over ALL bars, and over a random bar in each week-block,
so candidate net_bp can be compared against 'just be long for H minutes'."""
import numpy as np, pandas as pd
import htf_levels as HL

EPOCH_MON = np.datetime64('1970-01-05')
rng = np.random.default_rng(0)
print(f"{'sym':9s} {'hold':>5s} {'uncond_bp':>10s} {'wkblock_rand_bp':>16s} {'t_rand':>7s} {'tot_ret_%':>10s}")
for sym in HL.POINT:
    d = HL.load(sym); idx = d.index; C = d['close'].values
    ex = np.clip(idx.searchsorted(idx+pd.Timedelta(minutes=1)),0,len(d)-1)
    t = (idx+pd.Timedelta(hours=3)).values.astype('datetime64[m]')
    wid = ((t-EPOCH_MON).astype('timedelta64[m]').astype(np.int64)//(7*1440))
    ch = np.r_[0,np.where(wid[1:]!=wid[:-1])[0]+1,len(wid)]
    tot = (C[-1]/C[0]-1)*100
    for h in [60,240,1440]:
        exh = np.clip(idx.searchsorted(idx+pd.Timedelta(minutes=h)),0,len(d)-1)
        r = (C[exh]-C)/C*1e4
        unc = r.mean()
        # one random entry bar per week block -> matches the candidates' trade structure
        picks=[]
        for k in range(len(ch)-1):
            a,b = ch[k],ch[k+1]
            if b-a < h+2: continue
            picks.append(rng.integers(a,b-h-1))
        picks=np.array(picks); rr=r[picks]
        tr = rr.mean()/(rr.std(ddof=1)/np.sqrt(len(rr)))
        print(f'{sym:9s} {h:5d} {unc:10.2f} {rr.mean():16.2f} {tr:7.2f} {tot:10.1f}')
    HL._cache.pop(sym,None)
