"""
BB Strategy Results Analyzer
Run: python ml_test/analyze_results.py
Analyzes: data/processed/BB_strategy_tuned.csv
"""
import pandas as pd
import numpy as np

CSV_PATH = 'data/processed/BB_strategy_tuned.csv'

# ============================================================================
# LOAD
# ============================================================================
df = pd.read_csv(CSV_PATH)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['date']     = df['Datetime'].dt.date
df['week']     = df['Datetime'].dt.isocalendar().week.astype(int)
df['year']     = df['Datetime'].dt.year
df['month']    = df['Datetime'].dt.to_period('M')
df['dow']      = df['Datetime'].dt.day_name()
df['hour']     = df['Datetime'].dt.hour

trades = df[df['result'].isin(['WIN', 'LOSS'])].copy()

# Use pnl_r column if present (written by hedging_strategy_bb.py),
# otherwise fall back to RR=2 with a warning
if 'pnl_r' in trades.columns:
    trades['pnl'] = trades['pnl_r']
    wins_r  = trades.loc[trades['result'] == 'WIN', 'pnl'].iloc[0] if (trades['result'] == 'WIN').any() else 2.0
    RR = round(wins_r, 4)
else:
    RR = 2.0
    print('WARNING: pnl_r column not found — using hardcoded RR=2. Re-run hedging_strategy_bb.py to fix.')
    trades['pnl'] = trades['result'].map({'WIN': RR, 'LOSS': -1.0})

trades['is_win'] = (trades['result'] == 'WIN').astype(int)

total   = len(trades)
wins    = trades['is_win'].sum()
losses  = total - wins
wr      = wins / total * 100
net_pnl = trades['pnl'].sum()

print('=' * 70)
print(f'BB STRATEGY RESULTS ANALYSIS  —  R:R {RR}:1')
print('=' * 70)
print(f'File    : {CSV_PATH}')
print(f'Range   : {trades["Datetime"].min().date()} → {trades["Datetime"].max().date()}')
print(f'Trades  : {total}  |  Wins: {wins}  |  Losses: {losses}')
print(f'Win Rate: {wr:.2f}%')
print(f'Net P&L : {net_pnl:.1f}R')
print(f'EV/trade: {net_pnl/total:.4f}R')

# ============================================================================
# DRAWDOWN
# ============================================================================
cum_pnl  = trades['pnl'].cumsum()
peak     = cum_pnl.cummax()
drawdown = cum_pnl - peak
max_dd   = drawdown.min()
max_dd_time = trades['Datetime'].loc[drawdown.idxmin()]

print(f'\n{"─"*70}')
print('DRAWDOWN')
print(f'{"─"*70}')
print(f'Max Drawdown : {max_dd:.1f}R  (at {max_dd_time})')
print(f'Final Equity : {cum_pnl.iloc[-1]:.1f}R')

# ============================================================================
# CONSECUTIVE LOSSES
# ============================================================================
max_consec_loss = max_consec_win = cur = 0
prev = None
for r in trades['result']:
    if r == prev:
        cur += 1
    else:
        cur  = 1
        prev = r
    if r == 'LOSS': max_consec_loss = max(max_consec_loss, cur)
    else:           max_consec_win  = max(max_consec_win,  cur)

print(f'\n{"─"*70}')
print('STREAKS')
print(f'{"─"*70}')
print(f'Max Consecutive Losses : {max_consec_loss}')
print(f'Max Consecutive Wins   : {max_consec_win}')

# ============================================================================
# HELPER
# ============================================================================
def wr_table(grp, label_col, label='Group', preserve_order=False):
    rows = []
    for key, g in grp:
        w = g['is_win'].sum()
        t = len(g)
        p = g['pnl'].sum()
        rows.append({'label': key, 'trades': t, 'wins': w,
                     'wr': w/t*100 if t else 0, 'pnl': p})
    df_out = pd.DataFrame(rows) if preserve_order else pd.DataFrame(rows).sort_values('label')
    print(f'\n  {label:<22} {"Trades":>7} {"Wins":>6} {"WR%":>7} {"P&L(R)":>8}')
    print(f'  {"─"*55}')
    for _, r in df_out.iterrows():
        print(f'  {str(r["label"]):<22} {int(r["trades"]):>7} {int(r["wins"]):>6} '
              f'{r["wr"]:>6.1f}% {r["pnl"]:>8.1f}')
    wrs = df_out['wr']
    print(f'\n  Mean WR: {wrs.mean():.1f}%  |  Std: {wrs.std():.1f}%  |  '
          f'Min: {wrs.min():.1f}%  |  Max: {wrs.max():.1f}%')
    return df_out

# ============================================================================
# MONTHLY
# ============================================================================
print(f'\n{"─"*70}')
print('MONTHLY WIN RATE  (all sessions)')
print(f'{"─"*70}')
wr_table(trades.groupby('month'), 'month', label='Month')

# ============================================================================
# MONTHLY — ASIAN SESSION ONLY
# ============================================================================
print(f'\n{"─"*70}')
print('MONTHLY WIN RATE  (ASIAN session only)')
print(f'{"─"*70}')
asian = trades[trades['session'] == 'ASIAN']
if len(asian):
    wr_table(asian.groupby('month'), 'month', label='Month (Asian)')
    # Asian drawdown
    asian_cum = asian['pnl'].cumsum()
    asian_dd  = (asian_cum - asian_cum.cummax()).min()
    print(f'\n  Asian Net P&L : {asian_cum.iloc[-1]:.1f}R')
    print(f'  Asian Max DD  : {asian_dd:.1f}R')
else:
    print('  No ASIAN trades found.')

# ============================================================================
# WEEKLY
# ============================================================================
print(f'\n{"─"*70}')
print('WEEKLY WIN RATE')
print(f'{"─"*70}')
trades['yw'] = trades['Datetime'].dt.strftime('%Y-W%V')
wr_table(trades.groupby('yw'), 'yw', label='Year-Week')

# ============================================================================
# DAY OF WEEK
# ============================================================================
DOW_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
print(f'\n{"─"*70}')
print('DAY OF WEEK')
print(f'{"─"*70}')
dow_grp = [(d, trades[trades['dow'] == d]) for d in DOW_ORDER if (trades['dow'] == d).any()]
wr_table(iter(dow_grp), 'dow', label='Day', preserve_order=True)

# ============================================================================
# SESSION
# ============================================================================
print(f'\n{"─"*70}')
print('SESSION')
print(f'{"─"*70}')
wr_table(trades.groupby('session'), 'session', label='Session')

# ============================================================================
# DAILY  (summary stats only — too many rows to print individually)
# ============================================================================
print(f'\n{"─"*70}')
print('DAILY WIN RATE  (summary stats)')
print(f'{"─"*70}')
daily = trades.groupby('date').apply(
    lambda g: pd.Series({'trades': len(g), 'wr': g['is_win'].mean()*100, 'pnl': g['pnl'].sum()}),
    include_groups=False,
).reset_index()
print(f'  Trading days  : {len(daily)}')
print(f'  Mean daily WR : {daily["wr"].mean():.1f}%')
print(f'  Std  daily WR : {daily["wr"].std():.1f}%')
print(f'  Min  daily WR : {daily["wr"].min():.1f}%  ({daily.loc[daily["wr"].idxmin(),"date"]})')
print(f'  Max  daily WR : {daily["wr"].max():.1f}%  ({daily.loc[daily["wr"].idxmax(),"date"]})')
print(f'  Profitable days (>0 P&L) : {(daily["pnl"] > 0).sum()} / {len(daily)}')
print(f'  Loss days                : {(daily["pnl"] < 0).sum()} / {len(daily)}')

print('\n' + '=' * 70)
print('DONE')
print('=' * 70)
