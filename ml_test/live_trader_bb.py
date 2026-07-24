"""
BB Strategy Live Trader -- MetaTrader 5
========================================
Uses the RF model trained in hedging_strategy_bb.py.
Bollinger Band %%B determines trade direction.


Signal logic (all three gates must pass):
  1. RF probability >= RF_THRESH_FINAL  (high-quality setup filter)
  2. BB %%B <= BB_LONG_THRESH           -> LONG  (price near lower band)
     BB %%B >= BB_SHORT_THRESH          -> SHORT (price near upper band)
  3. Current UTC hour in [00:00, 08:00)

Run hedging_strategy_bb.py first to train and save the model.
Run bb_tune.py to retune BB params, then update CONFIGURATION below.

Usage:
  python ml_test/live_trader_bb.py --dry    # Dry run (no real trades)
  python ml_test/live_trader_bb.py --live   # LIVE TRADING (real money!)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import time
import logging
import argparse
import os
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION  -- keep in sync with hedging_strategy_bb.py
# ============================================================================
SYMBOL        = "XAUUSDz"       # Adjust suffix for your broker
TIMEFRAME     = mt5.TIMEFRAME_M1
LOT_SIZE      = 0.01
MAX_POSITIONS = 2

# BB forward test params (from hedging_strategy_bb.py CONFIGURATION)
BB_PERIOD       = 14
BB_STD_MUL      = 2.793
BB_LONG_THRESH  = 0.110   # %%B <= this -> LONG
BB_SHORT_THRESH = 0.923   # %%B >= this -> SHORT
RF_THRESH_FINAL = 0.694
TARGET_PCT      = 0.0015   # 0.2%  take profit (matches FWD_TARGET_PCT in forward test)
STOP_PCT        = 0.0005  # 0.05% stop loss

MAGIC_LONG  = 237001
MAGIC_SHORT = 237002

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR)
RF_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'random_forest.pkl')
FEATURES_PATH = os.path.join(PROJECT_ROOT, 'models', 'feature_names.txt')
LOG_PATH      = os.path.join(SCRIPT_DIR, 'live_trader_bb.log')

MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE ENGINEERING  -- must match hedging_strategy_bb.py exactly
# ============================================================================
def create_microstructure_features(df):
    df = df.copy()

    df['range']      = df['High'] - df['Low']
    df['body']       = df['Close'] - df['Open']
    df['abs_body']   = abs(df['body'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['body_pct']   = df['abs_body'] / (df['range'] + 1e-10)

    df['close_position']   = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
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
    df['consistency_3'] = df['up_count_3'].apply(lambda x: max(x, 3 - x))
    df['consistency_5'] = df['up_count_5'].apply(lambda x: max(x, 5 - x))

    df['atr_3']  = df['range'].rolling(3).mean()
    df['atr_10'] = df['range'].rolling(10).mean()
    df['atr_20'] = df['range'].rolling(20).mean()
    df['vol_ratio']       = df['atr_3'] / (df['atr_10'] + 1e-10)
    df['vol_expansion']   = (df['range'] > df['atr_10'] * 1.2).astype(int)
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

    # Deeper flow
    df['flow_15']         = df['directional_flow'].rolling(15).sum()
    df['flow_20']         = df['directional_flow'].rolling(20).sum()
    df['flow_accel']      = df['flow_3'] - df['flow_3'].shift(3)
    df['flow_accel_5']    = df['flow_5'] - df['flow_5'].shift(5)
    df['abs_flow_3']      = df['flow_3'].abs()
    df['abs_flow_5']      = df['flow_5'].abs()
    df['abs_flow_10']     = df['flow_10'].abs()
    df['flow_divergence'] = (df['flow_3'] * df['flow_10'] < 0).astype(int)

    # Flow quality
    df['consecutive_up']   = df['is_up'].groupby((df['is_up'] != df['is_up'].shift()).cumsum()).cumcount() + 1
    df['consecutive_up']   = df['consecutive_up'] * df['is_up']
    df['consecutive_down'] = (1 - df['is_up']).groupby(((1 - df['is_up']) != (1 - df['is_up']).shift()).cumsum()).cumcount() + 1
    df['consecutive_down'] = df['consecutive_down'] * (1 - df['is_up'])
    df['max_consecutive']  = df[['consecutive_up', 'consecutive_down']].max(axis=1)
    df['flow_efficiency']  = df['abs_body'] / (df['range'] + 1e-10)
    df['flow_eff_3']       = df['flow_efficiency'].rolling(3).mean()
    df['flow_eff_5']       = df['flow_efficiency'].rolling(5).mean()

    # Volatility regime
    df['atr_roc']       = (df['atr_3'] - df['atr_3'].shift(3)) / (df['atr_3'].shift(3) + 1e-10)
    df['vol_breakout']  = df['range'] / (df['atr_20'] + 1e-10)
    df['range_min_5']   = df['range'].rolling(5).min()
    df['range_max_5']   = df['range'].rolling(5).max()
    df['range_squeeze'] = df['range_min_5'] / (df['range_max_5'] + 1e-10)

    # Distance features
    df['abs_dist_ema8']  = df['dist_ema8'].abs()
    df['dist_ema21']     = (df['Close'] - df['ema_21']) / df['Close']
    df['abs_dist_ema21'] = df['dist_ema21'].abs()
    df['ema_spread']     = (df['ema_8'] - df['ema_21']) / df['Close']
    df['abs_ema_spread'] = df['ema_spread'].abs()

    # Combo features
    df['combo_abs_flow_vol']     = df['abs_flow_5'] * df['vol_ratio']
    df['combo_eff_flow']         = df['flow_eff_3'] * df['abs_flow_3']
    df['combo_consecutive_body'] = df['max_consecutive'] * df['body_pct']
    df['combo_atr_roc_accel']    = df['atr_roc'] * df['flow_accel'].abs()
    df['combo_squeeze_flow']     = (1 - df['range_squeeze']) * df['abs_flow_3']
    df['combo_dist_flow']        = df['abs_dist_ema8'] * df['abs_flow_5']

    return df.dropna()


def add_combo_features(df):
    """Original 8 combo features computed outside create_microstructure_features in training."""
    df = df.copy()
    df['combo_flow_trend']           = df['flow_momentum'] * df['trend_align']
    df['combo_vol_imbalance']        = df['vol_ratio'] * df['imbalance_3']
    df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
    df['combo_body_reject']          = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
    df['combo_trend_volatility']     = df['trend_align'] * df['vol_expansion']
    df['combo_imbalance_momentum']   = df['imbalance_5'] * df['flow_5']
    df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
    df['combo_vol_flow']             = df['vol_ratio'] * df['flow_3']
    return df


# ============================================================================
# MT5 HELPERS
# ============================================================================
def connect_mt5():
    if MT5_PATH and os.path.exists(MT5_PATH):
        if not mt5.initialize(path=MT5_PATH):
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False
    else:
        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False
    logger.info(f"MT5 connected: version={mt5.version()}")
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        logger.error(f"Symbol {SYMBOL} not found")
        return False
    if not info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            logger.error(f"Failed to select {SYMBOL}")
            return False
    logger.info(f"Symbol {SYMBOL} ready  |  point={info.point}  digits={info.digits}")
    return True


def get_filling_mode():
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return None
    filling = info.filling_mode
    if filling & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:
        return mt5.ORDER_FILLING_IOC
    return None


def get_bars(count=300):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to get rates: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df['Datetime'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'open': 'Open', 'high': 'High',
        'low': 'Low',   'close': 'Close',
    }, inplace=True)
    df.set_index('Datetime', inplace=True)
    return df[['Open', 'High', 'Low', 'Close']]


def wait_for_next_minute():
    now = datetime.now(timezone.utc)
    seconds = 60 - now.second - (now.microsecond / 1_000_000) + 2
    if seconds > 0:
        logger.info(f"Waiting {seconds:.1f}s for next candle close...")
        time.sleep(seconds)
    return datetime.now(timezone.utc)


def get_open_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return []
    return [p for p in positions if p.magic in (MAGIC_LONG, MAGIC_SHORT)]


def get_session_label():
    h = datetime.now(timezone.utc).hour
    if 0  <= h < 8:  return 'ASIAN'
    if 8  <= h < 13: return 'LONDON'
    if 13 <= h < 17: return 'NY_OVERLAP'
    if 17 <= h < 22: return 'NY'
    return 'LATE'


# ============================================================================
# ORDER EXECUTION
# ============================================================================
def place_order(direction, lot_size):
    filling = get_filling_mode()
    if filling is None:
        logger.error("Cannot determine filling mode — skipping order")
        return None

    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        logger.error(f"Failed to get tick: {mt5.last_error()}")
        return None

    digits = mt5.symbol_info(SYMBOL).digits

    if direction == 'LONG':
        price      = tick.ask
        tp         = round(price * (1 + TARGET_PCT), digits)
        sl         = round(price * (1 - STOP_PCT),   digits)
        order_type = mt5.ORDER_TYPE_BUY
        magic      = MAGIC_LONG
        comment    = "BB_LONG"
    else:
        price      = tick.bid
        tp         = round(price * (1 - TARGET_PCT), digits)
        sl         = round(price * (1 + STOP_PCT),   digits)
        order_type = mt5.ORDER_TYPE_SELL
        magic      = MAGIC_SHORT
        comment    = "BB_SHORT"

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       lot_size,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    20,
        "magic":        magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    result = mt5.order_send(request)
    if result is None:
        logger.error(f"order_send returned None: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed: retcode={result.retcode}  comment={result.comment}")
        return None

    logger.info(
        f"ORDER FILLED: {direction} @ {price:.2f}  "
        f"SL={sl:.2f}  TP={tp:.2f}  ticket={result.order}"
    )
    return result


# ============================================================================
# SIGNAL DETECTION
# ============================================================================
def check_signal(df_feat, model, feature_names):
    """
    Evaluate the last completed bar.
    Returns (direction, rf_prob, reason_str).
    direction is 'LONG', 'SHORT', or None.
    """
    if len(df_feat) < 30:
        return None, 0.0, "Not enough bars"

    row = df_feat.iloc[-2]  # Last COMPLETED bar (iloc[-1] is still forming)

    X = pd.DataFrame([row[feature_names].values], columns=feature_names).fillna(0)
    rf_prob = model.predict_proba(X)[0, 1]

    if rf_prob < RF_THRESH_FINAL:
        return None, rf_prob, f"RF={rf_prob:.3f} < threshold {RF_THRESH_FINAL}"

    # Bollinger Band %%B on last completed bar
    close_s = df_feat['Close']
    mid    = close_s.rolling(BB_PERIOD).mean()
    std    = close_s.rolling(BB_PERIOD).std()
    upper  = mid + BB_STD_MUL * std
    lower  = mid - BB_STD_MUL * std
    bb_pct = ((close_s - lower) / (upper - lower + 1e-10)).iloc[-2]

    if bb_pct <= BB_LONG_THRESH:
        direction = 'LONG'
    elif bb_pct >= BB_SHORT_THRESH:
        direction = 'SHORT'
    else:
        return None, rf_prob, f"RF={rf_prob:.3f}  BB%={bb_pct:.3f} (mid-band, skip)"

    return direction, rf_prob, f"RF={rf_prob:.3f}  BB%={bb_pct:.3f}"


# ============================================================================
# MAIN LOOP
# ============================================================================
def run(live_mode=False):
    banner = "LIVE TRADING" if live_mode else "DRY RUN"
    logger.info("=" * 72)
    logger.info(f"  BB STRATEGY LIVE TRADER — {banner}")
    logger.info("=" * 72)

    # Load model and feature list
    try:
        with open(RF_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(FEATURES_PATH, 'r') as f:
            feature_names = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Run hedging_strategy_bb.py first to generate model files.")
        return

    logger.info(f"  Model loaded  |  features={len(feature_names)}")
    logger.info(f"  RF threshold: {RF_THRESH_FINAL}  |  BB period: {BB_PERIOD}  std: {BB_STD_MUL}")
    logger.info(f"  Long <=  BB%={BB_LONG_THRESH}   Short >= BB%={BB_SHORT_THRESH}")
    logger.info(f"  TP: {TARGET_PCT*100:.2f}%  SL: {STOP_PCT*100:.3f}%")
    logger.info(f"  Sessions: ALL")
    logger.info(f"  Symbol: {SYMBOL}  |  Lot: {LOT_SIZE}  |  Max positions: {MAX_POSITIONS}")
    logger.info("-" * 72)

    if not connect_mt5():
        return

    last_bar_time = None
    trade_count   = 0

    wait_for_next_minute()

    try:
        while True:
            session = get_session_label()

            df_raw = get_bars(300)
            if df_raw is None:
                time.sleep(60)
                continue

            current_bar_time = df_raw.index[-1]
            if current_bar_time == last_bar_time:
                time.sleep(5)
                continue
            last_bar_time = current_bar_time

            price    = df_raw['Close'].iloc[-2]
            open_pos = get_open_positions()
            n_open   = len(open_pos)

            # Engineer features
            df_feat = create_microstructure_features(df_raw)
            df_feat = add_combo_features(df_feat)
            if len(df_feat) < 30:
                logger.warning("Not enough bars after feature warmup — skipping")
                wait_for_next_minute()
                continue

            direction, rf_prob, reason = check_signal(df_feat, model, feature_names)

            if direction is not None:
                logger.info("=" * 72)
                logger.info(f"  SIGNAL: {direction} @ {price:.2f}  [{session}]  |  {reason}")

                if n_open >= MAX_POSITIONS:
                    logger.info(f"  Skipped — max positions ({n_open}/{MAX_POSITIONS})")
                elif live_mode:
                    result = place_order(direction, LOT_SIZE)
                    if result:
                        trade_count += 1
                        logger.info(f"  Trade #{trade_count} executed")
                    else:
                        logger.error("  Order execution failed")
                else:
                    if direction == 'LONG':
                        tp = round(price * (1 + TARGET_PCT), 2)
                        sl = round(price * (1 - STOP_PCT),   2)
                    else:
                        tp = round(price * (1 - TARGET_PCT), 2)
                        sl = round(price * (1 + STOP_PCT),   2)
                    logger.info(
                        f"  [DRY RUN] Would place {direction} @ {price:.2f}  "
                        f"SL={sl:.2f}  TP={tp:.2f}"
                    )
                    trade_count += 1

                logger.info("=" * 72)
            else:
                logger.info(
                    f"[{session}] {current_bar_time}  |  {price:.2f}  "
                    f"|  {reason}  |  pos={n_open}/{MAX_POSITIONS}"
                )

            wait_for_next_minute()

    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C)...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        mt5.shutdown()
        logger.info(f"MT5 closed. Trades placed this session: {trade_count}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BB Strategy Live Trader')
    parser.add_argument('--live', action='store_true', help='Enable LIVE trading (real money!)')
    parser.add_argument('--dry',  action='store_true', help='Dry run (default)')
    args = parser.parse_args()

    if args.live:
        print("=" * 72)
        print("  WARNING: LIVE TRADING MODE — REAL MONEY!")
        print(f"  Symbol: {SYMBOL}  |  Lot: {LOT_SIZE}")
        print(f"  TP: {TARGET_PCT*100:.2f}%  |  SL: {STOP_PCT*100:.3f}%")
        print(f"  Sessions: ALL")
        print("=" * 72)
        confirm = input("Type 'YES' to confirm: ")
        if confirm.strip() == "YES":
            run(live_mode=True)
        else:
            print("Aborted.")
    else:
        run(live_mode=False)
