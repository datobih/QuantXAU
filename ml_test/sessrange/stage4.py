"""Stage 4: is the day-clustered t real (two-legged hedge) or an artifact?
   Apply the identical statistic to every symbol as a control."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'c:/Users/David/Downloads/prototype_3-main/prototype_3-main/ml_test')
from sessrange.core import *
pd.set_option('display.width',300,'display.max_rows',600)
def tstat(x):
    x=np.asarray(x,float); return x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))

print('='*100)
print('(1) DIAGNOSIS on USDJPY LONDON h120: why does daily averaging shrink the variance?')
sym='USDJPYm'; hi,lo,blk,w=range_level(sym,7,12,12,21)
up=backtest(sym,hi,blk,+1,120).assign(leg='UP'); dn=backtest(sym,lo,blk,-1,120).assign(leg='DN')
both=pd.concat([up,dn],ignore_index=True); both['day']=both['ts'].dt.normalize()
cnt=both.groupby('day').size()
print(f'  trades={len(both)}  days={len(cnt)}  1-leg days={int((cnt==1).sum())}  2-leg days={int((cnt==2).sum())}')
piv=both.pivot_table(index='day',columns='leg',values='bp')
b2=piv.dropna()
print(f'  corr(UP,DN) on 2-leg days = {b2["UP"].corr(b2["DN"]):+.3f}   (strong negative => the pair is a hedge)')
print(f'  std: UP-leg {both[both.leg=="UP"]["bp"].std():.1f}  DN-leg {both[both.leg=="DN"]["bp"].std():.1f}  '
      f'2-leg day avg {b2.mean(axis=1).std():.1f}  1-leg day {both.groupby("day")["bp"].mean()[cnt==1].std():.1f}')
d1=both.groupby('day')['bp'].mean()[cnt==1]; d2=b2.mean(axis=1)
print(f'  mean bp: 1-leg days {d1.mean():+.2f} (n={len(d1)})   2-leg days {d2.mean():+.2f} (n={len(d2)})')
print(f'  t on 2-leg days only: {tstat(d2.values):+.2f}      t on 1-leg days only: {tstat(d1.values):+.2f}')

print()
print('='*100)
print('(2) CONTROL: same day-clustered statistic on EVERY symbol, London range, h120, both legs')
print('    if clustered-t is inflated everywhere, it is a property of the statistic, not of USDJPY')
rows=[]
for s2 in SYMS:
    h2,l2,bk2,_=range_level(s2,7,12,12,21)
    u=backtest(s2,h2,bk2,+1,120); dd=backtest(s2,l2,bk2,-1,120)
    bb=pd.concat([u,dd],ignore_index=True); bb['day']=bb['ts'].dt.normalize()
    dser=bb.groupby('day')['bp'].mean()
    gser=bb.groupby('day')['gross_bp'].mean()
    rows.append(dict(sym=s2,ntr=len(bb),nday=len(dser),
        gross=round(bb['gross_bp'].mean(),2),cost=round(bb['cost_bp'].mean(),2),
        net=round(bb['bp'].mean(),2),t_trade=round(tstat(bb['bp']),2),
        t_day=round(tstat(dser.values),2),t_day_gross=round(tstat(gser.values),2),
        std_trade=round(bb['bp'].std(),1),std_day=round(dser.std(),1)))
C=pd.DataFrame(rows); print(C.to_string(index=False))

print()
print('='*100)
print('(3) Same control for the ASIAN range, h120')
rows=[]
for s2 in SYMS:
    h2,l2,bk2,_=range_level(s2,0,7,7,20)
    u=backtest(s2,h2,bk2,+1,120); dd=backtest(s2,l2,bk2,-1,120)
    bb=pd.concat([u,dd],ignore_index=True); bb['day']=bb['ts'].dt.normalize()
    dser=bb.groupby('day')['bp'].mean()
    rows.append(dict(sym=s2,ntr=len(bb),nday=len(dser),gross=round(bb['gross_bp'].mean(),2),
        cost=round(bb['cost_bp'].mean(),2),net=round(bb['bp'].mean(),2),
        t_trade=round(tstat(bb['bp']),2),t_day=round(tstat(dser.values),2)))
print(pd.DataFrame(rows).to_string(index=False))

print()
print('='*100)
print('(4) USDJPY London h120 daily-clustered series: per-year mean and t')
dser=both.groupby('day')['bp'].mean()
yy=pd.DataFrame({'bp':dser.values},index=dser.index).groupby(lambda x:x.year)['bp']
print(yy.agg(n='size',mean='mean',t=lambda s:tstat(s.values)).round(2).to_string())
