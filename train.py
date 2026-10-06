"""Train a reproducible weekly baseline with a chronological holdout."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research import FEATURES, ROOT, SCHEMA, chronological_split, download_prices, prepare_features


def train(ticker, prices, directory):
    frame = prepare_features(prices)
    training, holdout = chronological_split(frame)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          LogisticRegression(random_state=42, max_iter=2000))
    model.fit(training[FEATURES], training['target'].astype(int))
    probabilities = model.predict_proba(holdout[FEATURES])[:, 1]
    labels = holdout['target'].astype(int)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f'{ticker}.joblib'
    joblib.dump(model, model_path)
    # Keep the evaluated model; do not silently refit on its holdout.
    metadata = {
        'ticker': ticker, 'schema': SCHEMA, 'features': FEATURES, 'frequency': '1wk',
        'target': 'next completed weekly close / current weekly close - 1 > 0',
        'train_start': training.index.min().isoformat(),
        'train_end': training.index.max().isoformat(),
        'train_target_end': training['target_date'].max().isoformat(),
        'holdout_start': holdout.index.min().isoformat(), 'holdout_end': holdout.index.max().isoformat(),
        'train_rows': len(training), 'holdout_rows': len(holdout),
        'accuracy': accuracy_score(labels, probabilities >= 0.5),
        'balanced_accuracy': balanced_accuracy_score(labels, probabilities >= 0.5),
        'brier_score': brier_score_loss(labels, probabilities),
        'majority_baseline_accuracy': float((labels == training['target'].mode().iloc[0]).mean()),
        'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    prices.to_csv(directory / f'{ticker}_weekly_prices.csv')
    metadata['data_sha256'] = hashlib.sha256((directory / f'{ticker}_weekly_prices.csv').read_bytes()).hexdigest()
    (directory / f'{ticker}.json').write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tickers', nargs='+')
    parser.add_argument('--output', type=Path, default=ROOT / 'trained_models')
    parser.add_argument('--prices-csv', type=Path, help='Saved weekly Close/Volume snapshot; one ticker only')
    args = parser.parse_args()
    if args.prices_csv and len(args.tickers) != 1:
        parser.error('--prices-csv requires exactly one ticker')
    for ticker in args.tickers:
        if not ticker.isalnum():
            parser.error('Ticker must contain letters and numbers only')
        prices = (pd.read_csv(args.prices_csv, index_col=0, parse_dates=True)
                  if args.prices_csv else download_prices(ticker))
        print(json.dumps(train(ticker, prices, args.output), indent=2))
