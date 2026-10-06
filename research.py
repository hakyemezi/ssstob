"""One weekly feature/target pipeline shared by training and inference."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import yfinance as yf

FEATURES = ['return_1w', 'return_4w', 'volatility_12w', 'distance_sma_12w', 'volume_change']
SCHEMA = 'weekly-close-v1'
ROOT = Path(__file__).resolve().parent


def eligible_tickers():
    listings = pd.read_excel(ROOT / 'data/ilkislem.xlsx', header=None, skiprows=[0])
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
    listed = listings[pd.to_datetime(listings.iloc[:, 4], errors='coerce', dayfirst=True) < cutoff]
    codes = set(listed.iloc[:, 1].astype(str).str.split('.').str[0])
    universe = pd.read_excel(ROOT / 'data/hisse20.xlsx').iloc[:, 0].astype(str)
    return sorted(codes & {value.split('.E')[0] for value in universe if '.E' in value})


def normalize_prices(frame, as_of=None):
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty or not {'Close', 'Volume'}.issubset(frame.columns):
        raise ValueError('No usable weekly prices')
    index = pd.to_datetime(frame.index)
    if index.tz is not None:
        index = index.tz_convert('Europe/Istanbul').tz_localize(None)
    frame.index = index.normalize()
    frame = frame.sort_index()
    if frame.index.duplicated().any():
        raise ValueError('Duplicate weekly timestamps')
    # Exclude the whole current week: weekly bars are labelled at week start.
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz='Europe/Istanbul').tz_localize(None)
    now = now.tz_localize(None) if now.tzinfo is not None else now
    monday = now.normalize() - pd.Timedelta(days=now.weekday())
    frame = frame.loc[frame.index < monday, ['Close', 'Volume']]
    frame = frame.apply(pd.to_numeric, errors='coerce')
    frame.loc[frame['Close'] <= 0, 'Close'] = np.nan
    return frame


def download_prices(ticker, as_of=None):
    frame = yf.download(ticker + '.IS', start='2015-01-01', interval='1wk',
                        auto_adjust=True, progress=False, timeout=30)
    return normalize_prices(frame, as_of)


def prepare_features(prices):
    result = prices.copy()
    close = result['Close']
    result['return_1w'] = close.pct_change(fill_method=None)
    result['return_4w'] = close.pct_change(4, fill_method=None)
    result['volatility_12w'] = result['return_1w'].rolling(12).std()
    result['distance_sma_12w'] = close / close.rolling(12).mean() - 1
    result['volume_change'] = result['Volume'].pct_change(fill_method=None)
    result['next_week_return'] = close.shift(-1) / close - 1
    result['target_date'] = pd.Series(result.index, index=result.index).shift(-1)
    result['target'] = (result['next_week_return'] > 0).where(result['next_week_return'].notna())
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def chronological_split(frame, fraction=0.8):
    labelled = frame.dropna(subset=['target', 'target_date'])
    if len(labelled) < 60:
        raise ValueError('At least 60 labelled weekly observations are required')
    if not 0.5 <= fraction <= 0.9:
        raise ValueError('Training fraction must be between 0.5 and 0.9')
    split = int(len(labelled) * fraction)
    test = labelled.iloc[split:]
    # Purge any training label whose outcome touches the holdout boundary.
    train = labelled.iloc[:split]
    train = train[train['target_date'] < test.index.min()]
    if train['target'].nunique() < 2:
        raise ValueError('Training requires both target classes')
    return train, test


def load_model(ticker, directory=None):
    import joblib
    directory = Path(directory or ROOT / 'trained_models')
    path = directory / f'{ticker}.joblib'
    metadata = json.loads((directory / f'{ticker}.json').read_text())
    if (metadata.get('schema') != SCHEMA or metadata.get('features') != FEATURES
            or metadata.get('frequency') != '1wk' or metadata.get('ticker') != ticker):
        raise ValueError('Incompatible model metadata; retrain with train.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get('sha256'):
        raise ValueError('Model checksum differs from its metadata')
    return joblib.load(path), metadata


def out_of_sample(frame, metadata):
    # Every prediction shown must be after every outcome used for training.
    return frame.loc[frame.index > pd.Timestamp(metadata['train_target_end'])]
