"""
BB Tuner — finds optimal Bollinger Band parameters for hedging_strategy_bb.py.

Run:  python ml_test/bb_tune.py
Then copy the printed CONFIGURATION block into hedging_strategy_bb.py.
"""
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# TUNING CONFIG
# ============================================================================
N_TRIALS   = 500
MIN_TRADES = 700   # trials with fewer trades are discarded

TARGET_PCT = 0.001   # 0.1%  take profit  — keep in sync with hedging_strategy_bb.py
STOP_PCT   = 0.0005  # 0.05% stop loss
HORIZON    = 20      # bars to scan forward

# ============================================================================
# FEATURE ENGINEERING  (identical to hedging_strategy_bb.py)
# ============================================================================
def create_microstructure_features(df):
    df = df.copy()

    df['range']     = df['High'] - df['Low']
    df['body']      = df['Close'] - df['Open']
    df['abs_body']  = abs(df['body'])
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['body_pct']  = df['abs_body'] / (df['range'] + 1e-10)

    df['close_position']  = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['directional_flow'] = df['body'] / df['Close']
    df['flow_3']  = df['directional_flow'].rolling(3).sum()
    df['flow_5']  = df['directional_flow'].rolling(5).sum()
    df['flow_10'] = df['directional_flow'].rolling(10).sum()
    df['flow_momentum'] = df['flow_3'] - df['flow_5'].shift(2)

    df['buy_imbalance']  = ((df['body'] > 0) & (df['body_pct'] > 0.6) & (df['close_position'] > 0.7)).astype(float)
    df['sell_imbalance'] = ((df['body'] < 0) & (df['body_pct'] > 0.6) & (df['close_position'] < 0.3)).astype(float)
    df['imbalance_3'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(3).sum()
    df['imbalance_5'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(5).sum()

    df['is_up']        = (df['Close'] > df['Open']).astype(int)
    df['up_count_3']   = df['is_up'].rolling(3).sum()
    df['up_count_5']   = df['is_up'].rolling(5).sum()
    df['consistency_3'] = df['up_count_3'].apply(lambda x: max(x, 3-x))
    df['consistency_5'] = df['up_count_5'].apply(lambda x: max(x, 5-x))

    df['atr_3']  = df['range'].rolling(3).mean()
    df['atr_10'] = df['range'].rolling(10).mean()
    df['atr_20'] = df['range'].rolling(20).mean()
    df['vol_ratio']      = df['atr_3'] / (df['atr_10'] + 1e-10)
    df['vol_expansion']  = (df['range'] > df['atr_10'] * 1.2).astype(int)
    df['vol_contraction'] = (df['range'] < df['atr_10'] * 0.7).astype(int)

    df['ema_8']  = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['trend_align'] = (
        ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int)
      - ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
    )
    df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']

    df['high_10'] = df['High'].rolling(10).max()
    df['low_10']  = df['Low'].rolling(10).min()
    df['at_high'] = (df['Close'] >= df['high_10'].shift(1) * 0.9999).astype(int)
    df['at_low']  = (df['Close'] <= df['low_10'].shift(1) * 1.0001).astype(int)

    df['upper_reject'] = (df['upper_wick'] > df['abs_body'] * 2).astype(int)
    df['lower_reject'] = (df['lower_wick'] > df['abs_body'] * 2).astype(int)

    df['big_body']   = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
    df['small_body'] = (df['abs_body'] < df['abs_body'].rolling(10).mean() * 0.5).astype(int)



    return df.dropna()


def label_outcomes(df, horizon=20, target=0.001, stop=0.0005):
    df = df.copy()
    outcomes = []
    for i in range(len(df) - horizon):
        if i % 20000 == 0:
            print(f'  Labeling {i}/{len(df)-horizon}...')
        entry  = df['Close'].iloc[i]
        future = df.iloc[i+1:i+horizon+1]

        long_hit = False
        for h, l in zip(future['High'], future['Low']):
            if h >= entry * (1 + target): long_hit = True; break
            if l <= entry * (1 - stop):   break

        short_hit = False
        if not long_hit:
            for h, l in zip(future['High'], future['Low']):
                if l <= entry * (1 - target): short_hit = True; break
                if h >= entry * (1 + stop):   break

        outcomes.append(1 if long_hit else (2 if short_hit else 0))

    df = df.iloc[:len(outcomes)].copy()
    df['outcome'] = outcomes
    return df


# ============================================================================
# LOAD & PREPARE
# ============================================================================
print('Loading data...')
df = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t',
                 names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('Datetime', inplace=True)
df = df[['Open','High','Low','Close']].copy()
print(f'  {len(df)} bars loaded')

print('Engineering features...')
df = create_microstructure_features(df)

print('Labeling outcomes...')
df = label_outcomes(df, HORIZON, TARGET_PCT, STOP_PCT)

# Combination features
df['combo_flow_trend']          = df['flow_momentum'] * df['trend_align']
df['combo_vol_imbalance']       = df['vol_ratio'] * df['imbalance_3']
df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
df['combo_body_reject']         = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
df['combo_trend_volatility']    = df['trend_align'] * df['vol_expansion']
df['combo_imbalance_momentum']  = df['imbalance_5'] * df['flow_5']
df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
df['combo_vol_flow']            = df['vol_ratio'] * df['flow_3']

split = int(len(df) * 0.6)
train = df.iloc[:split].copy()
test  = df.iloc[split:].copy()
print(f'  Train: {split} | Test: {len(test)}')

# ============================================================================
# TRAIN RANDOM FOREST  (RF prob is the first gate in the strategy)
# ============================================================================
feature_cols = [
    'flow_3','flow_5','flow_10','flow_momentum',
    'imbalance_3','imbalance_5',
    'consistency_3','consistency_5',
    'vol_ratio','vol_expansion','vol_contraction',
    'trend_align','dist_ema8',
    'at_high','at_low',
    'upper_reject','lower_reject',
    'big_body','small_body',
    'body_pct','close_position',



    
]
combo_features = [
    'combo_flow_trend','combo_vol_imbalance','combo_consistency_position',
    'combo_body_reject','combo_trend_volatility','combo_imbalance_momentum',
    'combo_position_consistency','combo_vol_flow',
]
all_features = feature_cols + combo_features

X_train = train[all_features].fillna(0)
y_train = (train['outcome'] != 0).astype(int)
X_test  = test[all_features].fillna(0)

print('Training Random Forest...')
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
test = test.copy()
test['rf_prob'] = rf.predict_proba(X_test)[:, 1]
print(f'  Done. rf_prob range: {test["rf_prob"].min():.3f} – {test["rf_prob"].max():.3f}')

# ============================================================================
# OPTUNA TUNING  — SL-first bar-by-bar simulation (no label shortcuts)
# ============================================================================
_df_close = df['Close'].values
_t_high   = test['High'].values
_t_low    = test['Low'].values
_t_close  = test['Close'].values
_t_rf     = test['rf_prob'].values
_n_test   = len(test)

def bb_objective(trial):
    bb_period  = trial.suggest_int(  'bb_period',  5,    50)
    bb_std_mul = trial.suggest_float('bb_std',     1.0,  3.0)
    long_thr   = trial.suggest_float('long_thr',   0.05, 0.45)
    short_thr  = trial.suggest_float('short_thr',  0.55, 0.95)
    rf_thr     = trial.suggest_float('rf_thr',     0.65, 0.90)

    close_s = pd.Series(_df_close)
    mid     = close_s.rolling(bb_period).mean()
    std     = close_s.rolling(bb_period).std()
    upper   = mid + bb_std_mul * std
    lower   = mid - bb_std_mul * std
    pct     = ((close_s - lower) / (upper - lower + 1e-10)).values[split:]

    wins = losses = 0
    for pos in range(_n_test):
        if _t_rf[pos] < rf_thr:
            continue
        bb = pct[pos]
        if   bb <= long_thr:  direction = 1
        elif bb >= short_thr: direction = -1
        else:                  continue

        entry = _t_close[pos]
        if direction == 1:
            tp_p = entry * (1 + TARGET_PCT)
            sl_p = entry * (1 - STOP_PCT)
        else:
            tp_p = entry * (1 - TARGET_PCT)
            sl_p = entry * (1 + STOP_PCT)

        hit = False
        for j in range(pos + 1, min(pos + 1 + HORIZON, _n_test)):
            h, l = _t_high[j], _t_low[j]
            if direction == 1:
                if l <= sl_p: losses += 1; hit = True; break
                if h >= tp_p: wins   += 1; hit = True; break
            else:
                if h >= sl_p: losses += 1; hit = True; break
                if l <= tp_p: wins   += 1; hit = True; break
        if not hit:
            losses += 1

    trades = wins + losses
    if trades < MIN_TRADES:
        return -9999.0
    return (wins * 2 - losses) / trades

print(f'\nRunning Optuna ({N_TRIALS} trials, min {MIN_TRADES} trades)...')
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(bb_objective, n_trials=N_TRIALS, show_progress_bar=True)

best    = study.best_params
best_ev = study.best_value

# Rerun best params to get full stats
_bp = best
close_s = pd.Series(_df_close)
mid  = close_s.rolling(_bp['bb_period']).mean()
std  = close_s.rolling(_bp['bb_period']).std()
pct  = ((close_s - (mid - _bp['bb_std']*std)) /
        ((mid + _bp['bb_std']*std) - (mid - _bp['bb_std']*std) + 1e-10)).values[split:]
rr   = TARGET_PCT / STOP_PCT
wins = losses = 0
for pos in range(_n_test):
    if _t_rf[pos] < _bp['rf_thr']: continue
    bb = pct[pos]
    if   bb <= _bp['long_thr']:  d = 1
    elif bb >= _bp['short_thr']: d = -1
    else: continue
    e = _t_close[pos]
    tp_p = e * (1 + TARGET_PCT) if d == 1 else e * (1 - TARGET_PCT)
    sl_p = e * (1 - STOP_PCT)   if d == 1 else e * (1 + STOP_PCT)
    hit = False
    for j in range(pos + 1, min(pos + 1 + HORIZON, _n_test)):
        h, l = _t_high[j], _t_low[j]
        if d == 1:
            if l <= sl_p: losses += 1; hit = True; break
            if h >= tp_p: wins   += 1; hit = True; break
        else:
            if h >= sl_p: losses += 1; hit = True; break
            if l <= tp_p: wins   += 1; hit = True; break
    if not hit: losses += 1
trades   = wins + losses
wr       = wins / trades * 100 if trades else 0
total_pnl = wins * rr - losses

# ============================================================================
# RESULTS — copy-paste into hedging_strategy_bb.py CONFIGURATION section
# ============================================================================
print(f'\nBest params  —  EV: {best_ev:.4f}R/trade  |  WR: {wr:.1f}%  |  Trades: {trades}  |  Total P&L: {total_pnl:.1f}R  |  R:R {rr:.0f}:1\n')
print('=' * 60)
print('Paste into hedging_strategy_bb.py CONFIGURATION section:')
print('=' * 60)
print(f'BB_PERIOD       = {best["bb_period"]}')
print(f'BB_STD_MUL      = {best["bb_std"]:.3f}')
print(f'BB_LONG_THRESH  = {best["long_thr"]:.3f}')
print(f'BB_SHORT_THRESH = {best["short_thr"]:.3f}')
print(f'RF_THRESH_FINAL = {best["rf_thr"]:.3f}')
print('=' * 60)
