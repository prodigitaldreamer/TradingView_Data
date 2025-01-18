# indicators.py

import pandas as pd
import numpy as np

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    # RSI calculation: 
    # Compute diff, positive gains (up), negative losses(down)
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    # Get average gain, average loss
    # Use EWM or SMA. Classic RSI uses SMA, but we can use ewm for smoothing or SMA for simplicity.
    # We'll use SMA here.
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    # Avoid division by zero
    rs = roll_up / roll_down
    rsi = 100 - (100/(1+rs))
    return rsi

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series, length=20, std_dev=2):
    # Middle = SMA of length
    middle = series.rolling(length).mean()
    # std dev
    std = series.rolling(length).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower

def add_indicators_to_csv(csv_path):
    data = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    # Ensure data sorted by time
    data = data.sort_index()
    close = data['close']

    # EMAs
    data['EMA20'] = ema(close, 20)
    data['EMA50'] = ema(close, 50)
    data['EMA100'] = ema(close, 100)
    data['EMA200'] = ema(close, 200)

    # RSI 14
    data['RSI14'] = rsi(close, 14)
    # RSI 7
    data['RSI7'] = rsi(close, 7)

    # Moving averages for RSI:
    data['RSI14_MA14'] = data['RSI14'].rolling(14).mean()
    data['RSI7_MA7'] = data['RSI7'].rolling(7).mean()

    # MACD
    macd_line, macd_signal, macd_hist = macd(close, 12, 26, 9)
    data['MACD'] = macd_line
    data['MACD_Signal'] = macd_signal
    data['MACD_Hist'] = macd_hist

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = bollinger_bands(close, 20, 2)
    data['BB_upper'] = bb_upper
    data['BB_middle'] = bb_middle
    data['BB_lower'] = bb_lower

    # Overwrite the csv file with indicators added
    data.to_csv(csv_path)
    print(f"Indicators added to {csv_path}")