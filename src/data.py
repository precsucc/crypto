import pandas as pd
import numpy as np
import os

def load_coin(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['timeOpen'], unit='ms')
    df = df.sort_values('date').set_index('date')
    
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_days).ffill()
    
    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_features(df: pd.DataFrame, feature_set: str = 'basic', window_size: int = 90) -> pd.DataFrame:
    df = df.copy()
    
    df['log_ret'] = np.log(df['priceClose'] / df['priceClose'].shift(1))
    
    if feature_set == 'basic':
        df['rsi'] = compute_rsi(df['priceClose'])
        df['sma_20'] = df['priceClose'].rolling(window=20).mean()
    elif feature_set == 'gbm':
        df['gbm_drift'] = df['log_ret'].rolling(window=window_size).mean()
        df['gbm_vol'] = df['log_ret'].rolling(window=window_size).std()
        sma_20 = df['priceClose'].rolling(window=20).mean()
        df['dist_sma'] = (df['priceClose'] - sma_20) / sma_20
    
    df.dropna(inplace=True)
    return df
