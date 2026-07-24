import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import sys, os

# ============================================================================
# CONFIGURATION - Edit these values to test different settings
# ============================================================================
RF_THRESHOLD = 0.70  # RF probability threshold for hedge signals (e.g., 0.70, 0.75, 0.80)

# BB forward test parameters - run bb_tune.py to retune
BB_PERIOD       = 14      # Bollinger Band lookback period
BB_STD_MUL      = 2.793   # Standard deviation multiplier
BB_LONG_THRESH  = 0.110 # %B threshold: LONG  when %B <= this (near lower band)
BB_SHORT_THRESH = 0.923   # %B threshold: SHORT when %B >= this (near upper band)
RF_THRESH_FINAL = 0.694   # RF probability filter for forward test
TARGET_PCT      = 0.001   # 0.1%  take profit (used for training labels)
FWD_TARGET_PCT  = 0.0015  # 0.15% take profit (forward test only)
STOP_PCT        = 0.0005  # 0.05% stop loss (used for training labels)
FWD_STOP_PCT    = 0.0005  # 0.05% stop loss (forward test only)
HORIZON         = 20      # bars to scan forward

def create_microstructure_features(df):
    df = df.copy()
    
    # Price structure
    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']
    df['abs_body'] = abs(df['body'])
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['body_pct'] = df['abs_body'] / (df['range'] + 1e-10)
    
    # Order flow
    df['close_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['directional_flow'] = df['body'] / df['Close']
    df['flow_3'] = df['directional_flow'].rolling(3).sum()
    df['flow_5'] = df['directional_flow'].rolling(5).sum()
    df['flow_10'] = df['directional_flow'].rolling(10).sum()
    df['flow_momentum'] = df['flow_3'] - df['flow_5'].shift(2)
    
    # Imbalance (strong directional pressure)
    df['buy_imbalance'] = ((df['body'] > 0) & (df['body_pct'] > 0.6) & (df['close_position'] > 0.7)).astype(float)
    df['sell_imbalance'] = ((df['body'] < 0) & (df['body_pct'] > 0.6) & (df['close_position'] < 0.3)).astype(float)
    df['imbalance_3'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(3).sum()
    df['imbalance_5'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(5).sum()
    
    # Momentum consistency - is_up = bullish candle (close > open)
    df['is_up'] = (df['Close'] > df['Open']).astype(int)
    df['up_count_3'] = df['is_up'].rolling(3).sum()
    df['up_count_5'] = df['is_up'].rolling(5).sum()
    df['consistency_3'] = df['up_count_3'].apply(lambda x: max(x, 3-x))
    df['consistency_5'] = df['up_count_5'].apply(lambda x: max(x, 5-x))
    
    # Volatility
    df['atr_3'] = df['range'].rolling(3).mean()
    df['atr_10'] = df['range'].rolling(10).mean()
    df['atr_20'] = df['range'].rolling(20).mean()
    df['vol_ratio'] = df['atr_3'] / (df['atr_10'] + 1e-10)
    df['vol_expansion'] = (df['range'] > df['atr_10'] * 1.2).astype(int)
    df['vol_contraction'] = (df['range'] < df['atr_10'] * 0.7).astype(int)
    
    # Trend structure
    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int) - ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
    df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']
    
    # Support/Resistance
    df['high_10'] = df['High'].rolling(10).max()
    df['low_10'] = df['Low'].rolling(10).min()
    df['at_high'] = (df['Close'] >= df['high_10'].shift(1) * 0.9999).astype(int)
    df['at_low'] = (df['Close'] <= df['low_10'].shift(1) * 1.0001).astype(int)
    
    # Rejection patterns
    df['upper_reject'] = (df['upper_wick'] > df['abs_body'] * 2).astype(int)
    df['lower_reject'] = (df['lower_wick'] > df['abs_body'] * 2).astype(int)
    
    # Size patterns
    df['big_body'] = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
    df['small_body'] = (df['abs_body'] < df['abs_body'].rolling(10).mean() * 0.5).astype(int)

    # Bollinger Bands (20-period, 2 std)
    df['bb_mid'] = df['Close'].rolling(20).mean()
    df['bb_std'] = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-10)



    return df.dropna()

def label_outcomes(df, horizon=15, target=0.001, stop=0.0005):
    df = df.copy()
    outcomes = []
    
    for i in range(len(df) - horizon):
        if i % 20000 == 0:
            print(f'  Labeling {i}/{len(df)-horizon}...')
        
        entry = df['Close'].iloc[i]
        future = df.iloc[i+1:i+horizon+1]
        
        # Check LONG
        long_hit = False
        for h, l in zip(future['High'], future['Low']):
            if h >= entry * (1 + target):
                long_hit = True
                break
            if l <= entry * (1 - stop):
                break
        
        # Check SHORT
        short_hit = False
        if not long_hit:
            for h, l in zip(future['High'], future['Low']):
                if l <= entry * (1 - target):
                    short_hit = True
                    break
                if h >= entry * (1 + stop):
                    break
        
        outcomes.append(1 if long_hit else (2 if short_hit else 0))
    
    df = df.iloc[:len(outcomes)].copy()
    df['outcome'] = outcomes
    return df

print('='*80)
print('XAUUSD 1MIN FEATURE CORRELATION ANALYSIS')
print('='*80)

print('\nLoading data...')
# New format: Date, Time, Open, High, Low, Close, TickVol, Vol, Spread
df = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t', 
                 names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('Datetime', inplace=True)
df = df[['Open','High','Low','Close']].copy()  # Keep only OHLC
print(f'Loaded {len(df)} bars (1-minute timeframe)')

print('\nEngineering features...')
df = create_microstructure_features(df)

print(f'\nLabeling outcomes ({HORIZON} bars, {TARGET_PCT*100:.2f}% target, {STOP_PCT*100:.3f}% stop)...')
df = label_outcomes(df, HORIZON, TARGET_PCT, STOP_PCT)

# Create combination features on full dataset
print('\nCreating combination features...')
df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']

# Split 60:40 for more test data
split = int(len(df) * 0.6)
test = df.iloc[split:].copy()

print(f'\nTrain set: {split} bars (60%)')
print(f'Test set: {len(test)} bars (40%)')
print(f'Base success rate: {(test["outcome"] != 0).sum() / len(test) * 100:.1f}%')

# Analyze correlations
print('\n' + '='*80)
print('FEATURE CORRELATION WITH SUCCESSFUL TRADES')
print('='*80)

safe_trades = test['outcome'] != 0

feature_cols = [
    'flow_3', 'flow_5', 'flow_10', 'flow_momentum',
    'imbalance_3', 'imbalance_5',
    'consistency_3', 'consistency_5',
    'vol_ratio', 'vol_expansion', 'vol_contraction',
    'trend_align', 'dist_ema8',
    'at_high', 'at_low',
    'upper_reject', 'lower_reject',
    'big_body', 'small_body',
    'body_pct', 'close_position',

  
   

]

correlations = []
for feat in feature_cols:
    if feat in test.columns:
        corr = test[feat].corr(safe_trades.astype(int))
        safe_mean = test[safe_trades][feat].mean()
        noise_mean = test[~safe_trades][feat].mean()
        diff = abs(safe_mean - noise_mean)
        correlations.append((feat, corr, safe_mean, noise_mean, diff))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(f'\n{"Feature":<20} {"Correlation":<12} {"Safe Mean":<12} {"Noise Mean":<12} {"Difference"}')
print('-'*80)
for feat, corr, safe_m, noise_m, diff in correlations[:15]:
    print(f'{feat:<20} {corr:>11.4f} {safe_m:>11.4f} {noise_m:>11.4f} {diff:>11.4f}')

# Create combination features
print('\n' + '='*80)
print('ANALYZING COMBINATION FEATURE CORRELATIONS')
print('='*80)

combo_features = [
    'combo_flow_trend', 'combo_vol_imbalance', 'combo_consistency_position',
    'combo_body_reject', 'combo_trend_volatility', 'combo_imbalance_momentum',
    'combo_position_consistency', 'combo_vol_flow'
]

print(f'\n{"Combination Feature":<35} {"Correlation":<12} {"Safe Mean":<12} {"Noise Mean":<12} {"Difference"}')
print('-'*100)
combo_correlations = []
for feat in combo_features:
    corr = test[feat].corr(safe_trades.astype(int))
    safe_mean = test[safe_trades][feat].mean()
    noise_mean = test[~safe_trades][feat].mean()
    diff = abs(safe_mean - noise_mean)
    combo_correlations.append((feat, corr, safe_mean, noise_mean, diff))
    print(f'{feat:<35} {corr:>11.4f} {safe_mean:>11.4f} {noise_mean:>11.4f} {diff:>11.4f}')

combo_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

# ============================================================================
# MULTI-FEATURE MODELS: Find optimal weighted combinations
# ============================================================================
print('\n' + '='*80)
print('TRAINING MODELS TO FIND OPTIMAL FEATURE COMBINATIONS')
print('='*80)

# Prepare train/test data
train = df.iloc[:split].copy()
train_safe = train['outcome'] != 0

# Select all available features
all_features = feature_cols + combo_features
X_train = train[all_features].fillna(0)
y_train = train_safe.astype(int)
X_test = test[all_features].fillna(0)
y_test = safe_trades.astype(int)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'\nTraining on {len(X_train)} samples with {len(all_features)} features')
print(f'Testing on {len(X_test)} samples')

# 1. Logistic Regression (linear combination of all features)
print('\n--- LOGISTIC REGRESSION ---')
print('Finding optimal linear combination of all features...')
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print(f'AUC-ROC: {lr_auc:.4f}')

# Show most important features by coefficient
feature_importance = list(zip(all_features, lr.coef_[0]))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
print(f'\nTop 10 features by coefficient:')
for feat, coef in feature_importance[:10]:
    print(f'  {feat:<35} {coef:>8.4f}')

# Test different probability thresholds
print(f'\nFiltered trading results by probability threshold:')
print(f'{"Threshold":<12} {"Success %":<12} {"Frequency %":<12} {"Count"}')
print('-'*60)
for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    filtered = test[lr_probs >= threshold]
    if len(filtered) > 0:
        success_rate = (filtered['outcome'] != 0).sum() / len(filtered) * 100
        frequency = len(filtered) / len(test) * 100
        print(f'{threshold:<12.2f} {success_rate:>11.1f} {frequency:>11.2f} {len(filtered):>11d}')

# 2. Random Forest (non-linear combinations)
print('\n--- RANDOM FOREST ---')
print('Finding optimal non-linear feature combinations...')
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
print(f'AUC-ROC: {rf_auc:.4f}')

# Show most important features
feature_importance_rf = list(zip(all_features, rf.feature_importances_))
feature_importance_rf.sort(key=lambda x: x[1], reverse=True)
print(f'\nTop 10 features by importance:')
for feat, importance in feature_importance_rf[:10]:
    print(f'  {feat:<35} {importance:>8.4f}')

# Test different probability thresholds
print(f'\nFiltered trading results by probability threshold:')
print(f'{"Threshold":<12} {"Success %":<12} {"Frequency %":<12} {"Count"}')
print('-'*60)
for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    filtered = test[rf_probs >= threshold]
    if len(filtered) > 0:
        success_rate = (filtered['outcome'] != 0).sum() / len(filtered) * 100
        frequency = len(filtered) / len(test) * 100
        print(f'{threshold:<12.2f} {success_rate:>11.1f} {frequency:>11.2f} {len(filtered):>11d}')

# Highlight RF >= 0.75 specifically
print(f'\n*** RF >= 0.75 RESULTS ***')
filtered_75 = test[rf_probs >= 0.75]
if len(filtered_75) > 0:
    success_75 = (filtered_75['outcome'] != 0).sum() / len(filtered_75) * 100
    print(f'Success Rate: {success_75:.1f}%')
    print(f'Trade Count: {len(filtered_75)}')
    print(f'Frequency: {len(filtered_75) / len(test) * 100:.2f}%')

# Save probabilities to test set for further analysis
test['lr_prob'] = lr_probs
test['rf_prob'] = rf_probs

# Find best combinations
print('\n' + '='*80)
print('TESTING FEATURE COMBINATIONS (FILTERED SETUPS)')
print('='*80)

results = []

# Test combinations
combo1 = test[(test['imbalance_3'].abs() >= 2) & (test['consistency_5'] >= 4)]
if len(combo1) > 0:
    results.append(('Strong imbalance + high consistency', 
                   (combo1['outcome'] != 0).sum() / len(combo1) * 100,
                   len(combo1) / len(test) * 100, len(combo1)))

combo2 = test[(test['flow_momentum'] > 0.0005) & (test['trend_align'] == 1) & (test['vol_expansion'] == 1)]
if len(combo2) > 0:
    results.append(('Flow accel + trend align + vol expansion', 
                   (combo2['outcome'] != 0).sum() / len(combo2) * 100,
                   len(combo2) / len(test) * 100, len(combo2)))

combo3 = test[(test['big_body'] == 1) & (test['consistency_3'] == 3) & (test['close_position'] > 0.7)]
if len(combo3) > 0:
    results.append(('Big body + 3 up + strong close', 
                   (combo3['outcome'] != 0).sum() / len(combo3) * 100,
                   len(combo3) / len(test) * 100, len(combo3)))

# Test combinations using the new combo features
combo4 = test[test['combo_vol_imbalance'] > test['combo_vol_imbalance'].quantile(0.95)]
if len(combo4) > 0:
    results.append(('Top 5% vol*imbalance', 
                   (combo4['outcome'] != 0).sum() / len(combo4) * 100,
                   len(combo4) / len(test) * 100, len(combo4)))

combo5 = test[test['combo_flow_trend'] > test['combo_flow_trend'].quantile(0.95)]
if len(combo5) > 0:
    results.append(('Top 5% flow*trend', 
                   (combo5['outcome'] != 0).sum() / len(combo5) * 100,
                   len(combo5) / len(test) * 100, len(combo5)))

combo6 = test[test['combo_imbalance_momentum'] > test['combo_imbalance_momentum'].quantile(0.90)]
if len(combo6) > 0:
    results.append(('Top 10% imbalance*momentum', 
                   (combo6['outcome'] != 0).sum() / len(combo6) * 100,
                   len(combo6) / len(test) * 100, len(combo6)))

# Add model-based filters
combo7 = test[test['lr_prob'] >= 0.70]
if len(combo7) > 0:
    results.append(('LogReg prob >= 0.70', 
                   (combo7['outcome'] != 0).sum() / len(combo7) * 100,
                   len(combo7) / len(test) * 100, len(combo7)))

combo8 = test[test['rf_prob'] >= 0.70]
if len(combo8) > 0:
    results.append(('RandomForest prob >= 0.70', 
                   (combo8['outcome'] != 0).sum() / len(combo8) * 100,
                   len(combo8) / len(test) * 100, len(combo8)))

combo9 = test[(test['lr_prob'] >= 0.75) & (test['rf_prob'] >= 0.75)]
if len(combo9) > 0:
    results.append(('Both models >= 0.75', 
                   (combo9['outcome'] != 0).sum() / len(combo9) * 100,
                   len(combo9) / len(test) * 100, len(combo9)))

results.sort(key=lambda x: x[1], reverse=True)

print(f'\n{"Setup":<45} {"Success %":<12} {"Frequency %":<12} {"Count"}')
print('-'*80)
for setup, success, freq, count in results:
    if count >= 10:
        print(f'{setup:<45} {success:>11.1f} {freq:>11.2f} {count:>11d}')

print(f'\nBaseline: {(test["outcome"] != 0).sum() / len(test) * 100:.1f}%')

# Save models and scaler for live trading
print('\n' + '='*80)
print('SAVING MODELS FOR LIVE TRADING')
print('='*80)

import pickle
os.makedirs('models', exist_ok=True)

with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)
print('Saved: models/logistic_regression.pkl')

with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
print('Saved: models/random_forest.pkl')

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print('Saved: models/scaler.pkl')

# Save feature names for reference
with open('models/feature_names.txt', 'w') as f:
    f.write('\n'.join(all_features))
print('Saved: models/feature_names.txt')

# Print manual trading criteria
print('\n' + '='*80)
print('MANUAL TRADING CRITERIA (NO MODEL NEEDED)')
print('='*80)

# Analyze successful trades to find typical feature ranges
high_prob_trades = test[test['rf_prob'] >= 0.85]
successful = high_prob_trades[high_prob_trades['outcome'] != 0]

if len(successful) > 0:
    print(f'\n85%+ Win Rate Setups ({len(high_prob_trades)} total, {len(successful)} successful):')
    print(f'\nFeature ranges for high-probability trades:')
    print('-'*80)
    
    key_features = ['vol_ratio', 'imbalance_3', 'consistency_5', 'big_body', 
                    'vol_expansion', 'flow_momentum', 'trend_align', 'close_position']
    
    for feat in key_features:
        if feat in successful.columns:
            min_val = successful[feat].quantile(0.25)
            max_val = successful[feat].quantile(0.75)
            median = successful[feat].median()
            print(f'{feat:<25} Median: {median:>8.4f}  Range: {min_val:>8.4f} to {max_val:>8.4f}')

test.to_csv('data/processed/XAUUSD1_feature_analysis.csv')
print(f'\nSaved test results: data/processed/XAUUSD1_feature_analysis.csv')

# ============================================================================
# BOLLINGER BAND STRATEGY - FORWARD TEST
# ============================================================================
print('\n' + '='*80)
print('BOLLINGER BAND STRATEGY - FORWARD TEST')
print('='*80)
print(f'bb_period={BB_PERIOD}, bb_std={BB_STD_MUL:.3f}, '
      f'long<={BB_LONG_THRESH:.3f}, short>={BB_SHORT_THRESH:.3f}, '
      f'rf>={RF_THRESH_FINAL:.3f}')

# Recompute BB with best params
mid_opt   = df['Close'].rolling(BB_PERIOD).mean()
std_opt   = df['Close'].rolling(BB_PERIOD).std()
upper_opt = mid_opt + BB_STD_MUL * std_opt
lower_opt = mid_opt - BB_STD_MUL * std_opt
pct_opt   = (df['Close'] - lower_opt) / (upper_opt - lower_opt + 1e-10)
width_opt = (upper_opt - lower_opt) / (mid_opt + 1e-10)

def get_session(hour):
    if 0 <= hour < 8:
        return 'ASIAN'
    elif 8 <= hour < 13:
        return 'LONDON'
    elif 13 <= hour < 17:
        return 'NY_OVERLAP'
    elif 17 <= hour < 22:
        return 'NY'
    else:
        return 'LATE'

test_hedge = test.reset_index()
test_hedge['hour']      = test_hedge['Datetime'].dt.hour
test_hedge['session']   = test_hedge['hour'].apply(get_session)
test_hedge['bb_pct']    = pct_opt.loc[test.index].values
test_hedge['bb_width']  = width_opt.loc[test.index].values

hedge_setups = test_hedge[test_hedge['rf_prob'] >= RF_THRESH_FINAL].copy()

print(f'\nTotal test bars: {len(test_hedge)}')
print(f'High probability setups (RF >= {RF_THRESH_FINAL:.2f}): '
      f'{len(hedge_setups)} ({len(hedge_setups)/len(test_hedge)*100:.2f}%)')

rr_mult    = FWD_TARGET_PCT / FWD_STOP_PCT
bb_results = []

for idx, row in hedge_setups.iterrows():
    entry  = row['Close']
    bb_pct = row['bb_pct']

    if bb_pct <= BB_LONG_THRESH:
        direction = 'LONG'
    elif bb_pct >= BB_SHORT_THRESH:
        direction = 'SHORT'
    else:
        direction = 'SKIP'

    if direction == 'SKIP':
        result = 'SKIP'
    else:
        if direction == 'LONG':
            tp_price = entry * (1 + FWD_TARGET_PCT)
            sl_price = entry * (1 - FWD_STOP_PCT)
        else:  # SHORT
            tp_price = entry * (1 - FWD_TARGET_PCT)
            sl_price = entry * (1 + FWD_STOP_PCT)

        result = 'LOSS'  # default: expired at horizon or SL hit
        end_pos = min(idx + 1 + HORIZON, len(test_hedge))

        for _, fb in test_hedge.iloc[idx + 1 : end_pos].iterrows():
            h, l = fb['High'], fb['Low']
            if direction == 'LONG':
                if l <= sl_price:   # SL checked first - pessimistic
                    result = 'LOSS'
                    break
                if h >= tp_price:
                    result = 'WIN'
                    break
            else:  # SHORT
                if h >= sl_price:   # SL checked first - pessimistic
                    result = 'LOSS'
                    break
                if l <= tp_price:
                    result = 'WIN'
                    break

    if result == 'WIN':
        pnl_r = rr_mult
    elif result == 'LOSS':
        pnl_r = -1.0
    else:
        pnl_r = 0.0

    bb_results.append({
        'Datetime':  row['Datetime'],
        'session':   row['session'],
        'rf_prob':   row['rf_prob'],
        'bb_pct':    round(bb_pct, 4),
        'bb_width':  round(row['bb_width'], 4),
        'entry':     entry,
        'direction': direction,
        'result':    result,
        'pnl_r':     pnl_r,
    })

bb_df = pd.DataFrame(bb_results)

if len(bb_df) > 0:
    taken    = bb_df[bb_df['result'] != 'SKIP']
    skipped  = (bb_df['result'] == 'SKIP').sum()
    wins     = (taken['result'] == 'WIN').sum()
    losses   = (taken['result'] == 'LOSS').sum()
    win_rate = wins / len(taken) * 100 if len(taken) > 0 else 0

    total_pnl = wins * rr_mult + losses * (-1)
    avg_pnl   = total_pnl / len(taken) if len(taken) > 0 else 0

    print(f'\n--- OVERALL PERFORMANCE ---')
    print(f'TP: {FWD_TARGET_PCT*100:.3f}%  SL: {FWD_STOP_PCT*100:.3f}%  R:R = {rr_mult:.2f}:1')
    print(f'Total signals:      {len(bb_df)}')
    print(f'Trades taken:       {len(taken)}')
    print(f'Skipped (mid-band): {skipped}')
    print(f'Wins:               {wins}')
    print(f'Losses:             {losses}')
    print(f'Win Rate:           {win_rate:.1f}%')
    print(f'Net P&L:            {total_pnl:.1f}R  ({rr_mult:.0f}R win / 1R loss)')
    print(f'Average per trade:  {avg_pnl:.3f}R')
    print(f'Expected Value:     {avg_pnl:.3f}R per trade')

    print(f'\n--- DIRECTION BREAKDOWN ---')
    for d in ['LONG', 'SHORT', 'SKIP']:
        d_rows   = bb_df[bb_df['direction'] == d]
        d_wins   = (d_rows['result'] == 'WIN').sum()
        d_losses = (d_rows['result'] == 'LOSS').sum()
        d_taken  = d_wins + d_losses
        d_wr     = d_wins / d_taken * 100 if d_taken > 0 else 0
        d_pnl    = d_wins * rr_mult + d_losses * (-1)
        print(f'{d:<8} Signals: {len(d_rows):<6} Taken: {d_taken:<6} WR: {d_wr:>5.1f}%  P&L: {d_pnl:>7.1f}R')

    print(f'\n--- SESSION BREAKDOWN ---')
    for sess in bb_df['session'].unique():
        s_data   = bb_df[bb_df['session'] == sess]
        s_taken  = s_data[s_data['result'] != 'SKIP']
        s_wins   = (s_taken['result'] == 'WIN').sum()
        s_losses = (s_taken['result'] == 'LOSS').sum()
        s_trades = s_wins + s_losses
        s_wr     = s_wins / s_trades * 100 if s_trades > 0 else 0
        s_pnl    = s_wins * rr_mult + s_losses * (-1)
        print(f'{sess:<12} Signals: {len(s_data):<6} Trades: {s_trades:<6} WR: {s_wr:>5.1f}%  P&L: {s_pnl:>7.1f}R')

    bb_df['Datetime'] = bb_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    bb_df['rf_prob']  = bb_df['rf_prob'].round(3)
    bb_df['entry']    = bb_df['entry'].round(2)
    bb_df.to_csv('data/processed/BB_strategy_tuned.csv', index=False)
    print(f'\nSaved: data/processed/BB_strategy_tuned.csv')

    print('\n--- SAMPLE BB SETUPS (first 20) ---')
    print(bb_df.head(20).to_string(index=False))
else:
    print('\nNo setups generated.')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)