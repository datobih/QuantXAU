import pandas as pd
import numpy as np
import os

# ============================================================================
# RANGING REGIME DETECTOR — XAUUSD 15-minute
#
# Four indicators, each measuring ranging from a different angle:
#   1. Choppiness Index (CI)    — purpose-built ranging measure
#   2. Efficiency Ratio (ER)    — directional efficiency (low = ranging)
#   3. Hurst Exponent (H)       — mean reversion tendency (H < 0.5 = ranging)
#   4. Lag-1 Autocorrelation    — direct mean reversion test (negative = ranging)
#   5. ADX                      — directional strength (low = ranging)
#
# All five normalized to [0,1] ranging score then combined.
# Session bias applied (Asian/Late sessions are structurally more ranging).
#
# Output:
#   ranging_score : 0.0 (strong trend) to 1.0 (strong range)
#   regime        : RANGING / TRENDING / UNCERTAIN
#   Saved: data/processed/XAUUSD15_regimes.csv
# ============================================================================

DATA_15 = 'data/raw/XAUUSD15.csv'
OUT_PATH = 'data/processed/XAUUSD15_regimes.csv'

# Thresholds — tune these after reviewing the distribution plots
RANGING_THRESHOLD  = 0.62   # score >= this to RANGING
TRENDING_THRESHOLD = 0.40   # score <= this to TRENDING
# scores in between to UNCERTAIN (grid stays off)

# Indicator periods (in 15-min bars)
# All kept short to minimise lag — regime should flip within 1-2 hours
CI_PERIOD   = 7    # 1.75 hours  (was 14)
ER_PERIOD   = 10   # 2.5  hours  (was 20)
BB_PERIOD   = 10   # 2.5  hours  (replaces Hurst — BB squeeze = ranging)
BB_NORM_WIN = 50   # rolling window for BB width percentile normalisation
ATR_FAST    = 7    # 1.75 hours  \  fast/slow ATR ratio
ATR_SLOW    = 28   # 7    hours  /  (replaces AC1)
ADX_PERIOD  = 7    # 1.75 hours  (was 14)

# Composite weights (must sum to 1.0)
W_CI    = 0.30
W_ER    = 0.25
W_BB    = 0.20   # was W_HURST
W_ATR   = 0.15   # was W_AC
W_ADX   = 0.10

# Session bias adjustments (applied to ranging_score before classification)
# Positive = nudge toward RANGING, negative = nudge toward TRENDING
SESSION_BIAS = {
    'ASIAN':      +0.05,   # quiet, structurally ranging
    'LONDON':     -0.08,   # breakout tendency
    'NY_OVERLAP': -0.08,   # high volatility, directional
    'NY':         -0.03,
    'LATE':       +0.03,   # quiet, drift toward ranging
}


# ============================================================================
# LOAD DATA
# ============================================================================
def load_data(path):
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.strip('<>') for c in df.columns]
    df['Datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    df = df.set_index('Datetime').sort_index()
    df = df.rename(columns={
        'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low',
        'CLOSE': 'Close', 'TICKVOL': 'TickVol'
    })
    return df[['Open', 'High', 'Low', 'Close', 'TickVol']]


# ============================================================================
# SESSION LABEL
# ============================================================================
def label_session(dt_index):
    """Label each bar by trading session (UTC hours)."""
    hour = dt_index.hour
    session = pd.Series('LATE', index=dt_index)
    session[((hour >= 0)  & (hour < 7))]  = 'ASIAN'
    session[((hour >= 7)  & (hour < 12))] = 'LONDON'
    session[((hour >= 12) & (hour < 17))] = 'NY_OVERLAP'
    session[((hour >= 17) & (hour < 21))] = 'NY'
    session[((hour >= 21) | (hour < 0))]  = 'LATE'
    return session


# ============================================================================
# INDICATOR 1: CHOPPINESS INDEX
# Range: 0-100. >61.8 = choppy/ranging. <38.2 = trending.
# CHOP = 100 * log10(SUM(ATR1, n) / (HH - LL)) / log10(n)
# ============================================================================
def choppiness_index(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_sum = tr.rolling(period).sum()
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    hl_range = (hh - ll).replace(0, np.nan)

    ci = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
    return ci.clip(0, 100)


# ============================================================================
# INDICATOR 2: EFFICIENCY RATIO (Perry Kaufman)
# Range: 0-1. Near 0 = lots of back-and-forth = ranging. Near 1 = trending.
# ER = |close[n] - close[0]| / SUM(|close[i] - close[i-1]|, n)
# ============================================================================
def efficiency_ratio(close, period=20):
    direction = close.diff(period).abs()
    noise = close.diff().abs().rolling(period).sum()
    er = (direction / noise.replace(0, np.nan)).clip(0, 1)
    return er


# ============================================================================
# INDICATOR 3: BOLLINGER BAND WIDTH  (replaces Hurst)
# Narrow bands = low volatility = ranging.  Expands fast when a trend starts.
# Normalised using rolling percentile so it adapts to price level.
# ============================================================================
def bollinger_band_width(close, period=10):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (2 * std / sma.replace(0, np.nan))   # fraction of price


# ============================================================================
# INDICATOR 4: ATR RATIO  (replaces AC1)
# fast ATR / slow ATR < 1 means volatility is contracting = ranging.
# Expands immediately when a new directional move begins.
# ============================================================================
def atr_ratio(high, low, close, fast=7, slow=28):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_f = tr.ewm(span=fast, adjust=False).mean()
    atr_s = tr.ewm(span=slow, adjust=False).mean()
    return (atr_f / atr_s.replace(0, np.nan)).clip(0, 3)


# ============================================================================
# INDICATOR 5: ADX (Average Directional Index)
# Low ADX = no directional strength = ranging.
# ============================================================================
def compute_adx(high, low, close, period=14):
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = high - prev_high
    dm_minus = prev_low - low

    dm_plus  = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)

    atr      = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)

    di_sum = (di_plus + di_minus).replace(0, np.nan)
    dx     = 100 * (di_plus - di_minus).abs() / di_sum
    adx    = dx.ewm(span=period, adjust=False).mean()
    return adx


# ============================================================================
# NORMALIZE TO RANGING SCORE [0, 1]
# 1.0 = strong ranging signal, 0.0 = strong trending signal
# ============================================================================
def normalize_ci(ci):
    """CI 38.2-61.8 scale. Above 61.8 = fully ranging."""
    return ((ci - 38.2) / (61.8 - 38.2)).clip(0, 1)

def normalize_er(er):
    """ER 0-1. Low ER = ranging. Invert."""
    return (1.0 - er).clip(0, 1)

def normalize_bb_width(bb_width, norm_window=50):
    """Narrow bands = ranging (score 1). Wide bands = trending (score 0).
    Uses rolling min-max so it adapts to the current volatility regime."""
    rmax = bb_width.rolling(norm_window).max()
    rmin = bb_width.rolling(norm_window).min()
    score = 1.0 - (bb_width - rmin) / (rmax - rmin + 1e-9)
    return score.clip(0, 1)

def normalize_atr_ratio(ratio):
    """ratio < 1 = volatility contracting = ranging (score 1).
    ratio > 1.5 = expanding = trending (score 0)."""
    return (1.5 - ratio).clip(0, 1)

def normalize_adx(adx):
    """ADX < 20 = ranging, ADX > 45 = strong trend."""
    return ((45 - adx) / (45 - 15)).clip(0, 1)


# ============================================================================
# MAIN
# ============================================================================
print('=' * 70)
print('RANGING REGIME DETECTOR — XAUUSD 15-minute')
print('=' * 70)

print(f'\nLoading {DATA_15}...')
df = load_data(DATA_15)
print(f'Loaded {len(df):,} bars  |  {df.index[0]} to {df.index[-1]}')

# Session labels
df['session'] = label_session(df.index)

# -- Compute raw indicators ------------------------------------------------
print('\nComputing indicators...')

df['CI']       = choppiness_index(df['High'], df['Low'], df['Close'], CI_PERIOD)
df['ER']       = efficiency_ratio(df['Close'], ER_PERIOD)
df['BB_Width'] = bollinger_band_width(df['Close'], BB_PERIOD)
df['ATR_Ratio']= atr_ratio(df['High'], df['Low'], df['Close'], ATR_FAST, ATR_SLOW)
df['ADX']      = compute_adx(df['High'], df['Low'], df['Close'], ADX_PERIOD)

# -- Normalize to [0,1] ranging scores ------------------------------------
df['ci_score']  = normalize_ci(df['CI'])
df['er_score']  = normalize_er(df['ER'])
df['bb_score']  = normalize_bb_width(df['BB_Width'], BB_NORM_WIN)
df['atr_score'] = normalize_atr_ratio(df['ATR_Ratio'])
df['adx_score'] = normalize_adx(df['ADX'])

# -- Composite ranging score -----------------------------------------------
df['ranging_score'] = (
    W_CI  * df['ci_score'] +
    W_ER  * df['er_score'] +
    W_BB  * df['bb_score'] +
    W_ATR * df['atr_score'] +
    W_ADX * df['adx_score']
)

# -- Session bias ----------------------------------------------------------
df['session_bias'] = df['session'].map(SESSION_BIAS)
df['ranging_score_biased'] = (df['ranging_score'] + df['session_bias']).clip(0, 1)

# -- Regime classification -------------------------------------------------
df['regime'] = 'UNCERTAIN'
df.loc[df['ranging_score_biased'] >= RANGING_THRESHOLD,  'regime'] = 'RANGING'
df.loc[df['ranging_score_biased'] <= TRENDING_THRESHOLD, 'regime'] = 'TRENDING'

# -- Drop warmup bars (NaN period) -----------------------------------------
warmup = max(CI_PERIOD, ER_PERIOD, BB_PERIOD + BB_NORM_WIN, ATR_SLOW, ADX_PERIOD)
df_clean = df.iloc[warmup:].copy()

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
total = len(df_clean)
regime_counts = df_clean['regime'].value_counts()

print(f'\n{"-"*50}')
print(f'REGIME DISTRIBUTION  ({total:,} bars)')
print(f'{"-"*50}')
for regime in ['RANGING', 'UNCERTAIN', 'TRENDING']:
    n   = regime_counts.get(regime, 0)
    pct = n / total * 100
    bar = '#' * int(pct / 2)
    print(f'  {regime:<12} {n:>6,}  ({pct:5.1f}%)  {bar}')

print(f'\n{"-"*50}')
print(f'INDICATOR MEANS BY REGIME')
print(f'{"-"*50}')
print(f'{"Regime":<12} {"CI":>7} {"ER":>7} {"BB_W":>7} {"ATR_R":>7} {"ADX":>7} {"Score":>7}')
print(f'{"-"*60}')
for regime in ['RANGING', 'UNCERTAIN', 'TRENDING']:
    sub = df_clean[df_clean['regime'] == regime]
    if len(sub) == 0:
        continue
    print(
        f'{regime:<12} '
        f'{sub["CI"].mean():>7.1f} '
        f'{sub["ER"].mean():>7.3f} '
        f'{sub["BB_Width"].mean():>7.4f} '
        f'{sub["ATR_Ratio"].mean():>7.3f} '
        f'{sub["ADX"].mean():>7.1f} '
        f'{sub["ranging_score_biased"].mean():>7.3f}'
    )

print(f'\n{"-"*50}')
print(f'RANGING PERIODS BY SESSION')
print(f'{"-"*50}')
ranging_only = df_clean[df_clean['regime'] == 'RANGING']
print(f'{"Session":<14} {"Total bars":>11} {"Ranging bars":>13} {"Ranging %":>10}')
print(f'{"-"*52}')
for sess in ['ASIAN', 'LONDON', 'NY_OVERLAP', 'NY', 'LATE']:
    total_sess   = (df_clean['session'] == sess).sum()
    ranging_sess = (ranging_only['session'] == sess).sum()
    pct          = ranging_sess / total_sess * 100 if total_sess > 0 else 0
    print(f'  {sess:<12} {total_sess:>11,} {ranging_sess:>13,} {pct:>9.1f}%')

# -- Realized range analysis -----------------------------------------------
# For each bar, compute the actual price range over the next 8 bars (2 hours)
# as a % of current ATR. High = trending. Low = ranging.
FORWARD_N = 8
df_clean['fwd_range'] = df_clean['High'].rolling(FORWARD_N).max().shift(-FORWARD_N) - \
                        df_clean['Low'].rolling(FORWARD_N).min().shift(-FORWARD_N)
df_clean['atr14'] = choppiness_index(df_clean['High'], df_clean['Low'], df_clean['Close'], 14)

# Use raw price range normalized by recent volatility as validation
df_clean['fwd_range_pct'] = df_clean['fwd_range'] / df_clean['Close'] * 100

print(f'\n{"-"*50}')
print(f'VALIDATION — FORWARD REALIZED RANGE (next {FORWARD_N} bars)')
print(f'Lower range in RANGING periods = detector is working')
print(f'{"-"*50}')
print(f'{"Regime":<12} {"Avg fwd range (%)":>18} {"Median":>8}')
print(f'{"-"*42}')
for regime in ['RANGING', 'UNCERTAIN', 'TRENDING']:
    sub = df_clean[df_clean['regime'] == regime]['fwd_range_pct'].dropna()
    if len(sub) == 0:
        continue
    print(f'  {regime:<12} {sub.mean():>16.4f}%  {sub.median():>7.4f}%')

# ============================================================================
# SAVE
# ============================================================================
os.makedirs('data/processed', exist_ok=True)

save_cols = [
    'Open', 'High', 'Low', 'Close', 'TickVol',
    'session',
    'CI', 'ER', 'BB_Width', 'ATR_Ratio', 'ADX',
    'ci_score', 'er_score', 'bb_score', 'atr_score', 'adx_score',
    'ranging_score', 'session_bias', 'ranging_score_biased',
    'regime',
]
df_clean[save_cols].to_csv(OUT_PATH)
print(f'\nSaved: {OUT_PATH}  ({len(df_clean):,} bars)')

print('\n' + '=' * 70)
print('REGIME DETECTION COMPLETE')
print('=' * 70)
