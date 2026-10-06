# SSStoB: weekly stock research

A Streamlit research application for completed weekly stock prices and a reproducible baseline for next-week direction. Training and inference use the same weekly Close/Volume pipeline. The current unfinished week is excluded.

## Run

Python 3.11+:

```sh
python -m pip install -r requirements.txt
streamlit run st.py
```

Click **Load weekly data** to start. Navigating tabs does not download data or clear the cache. Download failures are shown per stock; successfully loaded stocks remain available. Prices are cached for one hour, with an explicit refresh option.

## Legacy models and migration

The existing `models/*.pkl` files are preserved for provenance, but their training dates, feature conventions and input frequency were not recorded in this repository. They are not loaded by the updated prediction interface. The old daily-input/weekly-history mixture has been removed.

The new model is an explicit **logistic-regression baseline**, not a reconstruction or validated replacement of the legacy models. Its features are weekly returns, trailing volatility, distance from a trailing average and volume change. It does not use the old technical-indicator feature set.

Train models before displaying predictions:

```sh
python train.py AKBNK ASELS
```

This writes a model, metadata and the downloaded weekly price snapshot to `trained_models/`. These generated files are ignored by Git. To reproduce training from a saved snapshot:

```sh
python train.py AKBNK --prices-csv trained_models/AKBNK_weekly_prices.csv --output rerun_models
```

Use the same dependency versions recorded in the metadata for reproducibility. The model checksum ties metadata to a specific artifact; only load model files you trust.

## Evaluation contract

- Target: next completed weekly close divided by the current weekly close, minus one; positive returns form class 1.
- Chronological 80/20 split; training rows whose outcome reaches the holdout boundary are purged.
- The scaler and classifier are fitted on training data only. The evaluated model is saved without refitting on holdout data.
- Metadata records training/holdout dates, row counts, accuracy, balanced accuracy, Brier score and a training-majority baseline.
- The UI displays predictions only after the last training outcome. A missing or incompatible metadata file disables that stock's predictions while leaving price charts available.
- Historical results are a fixed holdout replay, **not** a walk-forward backtest. Probabilities are not separately calibrated. Provider-adjusted prices can be revised.
- The spreadsheet universe is a present-day selection; survivorship/selection bias, trading costs and execution timing are not modelled. Holdout metrics do not establish profitable trading performance.

No live models were trained as part of the reliability changes. Synthetic tests verify the pipeline, separation of training/holdout, and artifact round-trip; they do not measure market performance.

## Tests

```sh
python -m pip install pytest
python -m pytest -q tests
```

A GitHub Actions example is provided at `docs/ci-tests.example.yml`. It is not active: enabling it requires copying it to `.github/workflows/tests.yml` with credentials that can manage workflows. Local tests use the commands above.
