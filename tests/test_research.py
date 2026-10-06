from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from research import FEATURES, chronological_split, normalize_prices, prepare_features, load_model, out_of_sample
from train import train


def prices():
    rng = np.random.default_rng(42)
    return pd.DataFrame({'Close': 100 * np.exp(np.cumsum(rng.normal(0, .03, 180))),
                         'Volume': rng.integers(100, 1000, 180)},
                        index=pd.date_range('2020-01-06', periods=180, freq='W-MON'))


def test_current_incomplete_week_excluded_and_multiindex_normalized():
    frame = prices().iloc[:4]
    frame.columns = pd.MultiIndex.from_product([frame.columns, ['AAA.IS']])
    actual = normalize_prices(frame, as_of='2020-01-29')
    assert actual.index.max() == pd.Timestamp('2020-01-20')
    assert actual.columns.tolist() == ['Close', 'Volume']


def test_feature_values_do_not_depend_on_future_prices():
    full = prepare_features(prices())
    short = prepare_features(prices().iloc[:100])
    pd.testing.assert_frame_equal(full.loc[short.index, FEATURES], short[FEATURES])
    assert pd.isna(short['target'].iloc[-1])
    assert short['return_1w'].iloc[-1] == pytest.approx(prices().Close.iloc[99] / prices().Close.iloc[98] - 1)


def test_split_purges_overlapping_labels():
    training, holdout = chronological_split(prepare_features(prices()))
    assert training.target_date.max() < holdout.index.min()


def test_training_metadata_and_reload_predictions(tmp_path):
    metadata = train('AAA', prices(), tmp_path)
    model, saved = load_model('AAA', tmp_path)
    assert saved['frequency'] == '1wk'
    eligible = out_of_sample(prepare_features(prices()), metadata)
    assert eligible.index.min() > pd.Timestamp(saved['train_target_end'])
    assert np.isfinite(model.predict_proba(eligible[FEATURES])).all()
    assert 0 <= saved['brier_score'] <= 1
    assert (tmp_path / 'AAA_weekly_prices.csv').exists()
    path = tmp_path / 'AAA.json'
    broken = json.loads(path.read_text()); broken['frequency'] = '1d'
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='Incompatible'): load_model('AAA', tmp_path)


def test_app_does_not_fetch_on_start(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import research
    def unexpected(*args, **kwargs): raise AssertionError('Unexpected download')
    monkeypatch.setattr(research, 'download_prices', unexpected)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'st.py').run()
    assert not app.exception
    assert not app.error
