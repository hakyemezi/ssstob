import pandas as pd
import plotly.express as px
import streamlit as st

from research import FEATURES, download_prices, eligible_tickers, load_model, out_of_sample, prepare_features

st.set_page_config(page_title='SSStoB research', layout='wide')
st.title('SSStoB — weekly stock research')
st.caption('Completed weekly bars only. Probabilities are research model outputs, not calibrated guarantees.')


@st.cache_data(ttl=3600)
def load_prices(ticker):
    return download_prices(ticker)


run, current, historical, charts = st.tabs(['Run research', 'Current predictions', 'Historical holdout', 'Stock analysis'])
with run:
    st.info('Legacy models have no recorded training period or input frequency. Train documented models with train.py before using predictions. Price charts work without a model.')
    refresh = st.checkbox('Refresh downloaded prices')
    if st.button('Load weekly data', type='primary'):
        if refresh:
            load_prices.clear()
        frames, failures = {}, {}
        try:
            tickers = eligible_tickers()
        except Exception as error:
            st.error(f'Could not read the stock universe: {error}')
            tickers = []
        for ticker in tickers:
            try:
                frame = prepare_features(load_prices(ticker))
                if frame.empty:
                    raise ValueError('Not enough complete observations')
                frames[ticker] = frame
            except Exception as error:
                failures[ticker] = str(error)
        st.session_state['frames'] = frames
        st.session_state['failures'] = failures
    frames = st.session_state.get('frames', {})
    st.write(f'{len(frames)} stocks loaded.')
    for ticker, error in st.session_state.get('failures', {}).items():
        st.warning(f'{ticker}: {error}')

predictions, unavailable = [], {}
for ticker, frame in st.session_state.get('frames', {}).items():
    try:
        model, metadata = load_model(ticker)
        eligible = out_of_sample(frame, metadata)
        if eligible.empty:
            raise ValueError('No observations after the recorded training outcomes')
        result = eligible[['Close', 'return_1w', 'next_week_return']].copy()
        result['Probability of positive next week'] = model.predict_proba(eligible[FEATURES])[:, 1]
        result['Stock'] = ticker
        result['Date'] = result.index
        predictions.append(result)
    except Exception as error:
        unavailable[ticker] = str(error)

with current:
    for ticker, error in unavailable.items():
        st.warning(f'{ticker}: {error}')
    if predictions:
        latest = pd.concat([frame.iloc[[-1]] for frame in predictions], ignore_index=True)
        latest = latest.rename(columns={'return_1w': 'Previous weekly return'})
        st.dataframe(latest.drop(columns=['next_week_return']), hide_index=True)
    else:
        st.info('Load data and train compatible models to see predictions.')

with historical:
    st.caption('Only observations after training outcomes are shown. This is a fixed holdout replay, not a walk-forward backtest; prices can be revised by the provider.')
    if predictions:
        past = pd.concat(predictions, ignore_index=True).dropna(subset=['next_week_return'])
        if not past.empty:
            date = st.selectbox('Completed week', sorted(past['Date'].unique(), reverse=True))
            st.dataframe(past[past['Date'] == date], hide_index=True)
        else:
            st.info('No realised holdout outcomes yet.')

with charts:
    frames = st.session_state.get('frames', {})
    if frames:
        ticker = st.selectbox('Stock', sorted(frames))
        st.plotly_chart(px.line(frames[ticker], y='Close', title=f'{ticker}: adjusted weekly close'), use_container_width=True)
