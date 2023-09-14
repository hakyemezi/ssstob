import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import time
from IPython.display import clear_output
from ta import add_all_ta_features
import warnings
import joblib
import plotly.express as px

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

################################################
# Functions
################################################
@st.cache_data
def process_stock_data(n_years=5):
    df1 = pd.read_excel("data/ilkislem.xlsx", header=None, skiprows=[0])
    today = pd.Timestamp.today()
    n_years_ago = today - pd.DateOffset(years=n_years)
    df1 = df1[pd.to_datetime(df1.iloc[:, 4], errors="coerce", dayfirst=True) < n_years_ago]
    code_list = df1.iloc[:, 1].apply(lambda x: str(x).split(".")[0]).tolist()
    df2 = pd.read_excel('data/hisse20.xlsx')
    result = [x.split('.E')[0] for x in df2.iloc[:, 0].astype(str) if '.E' in x]
    code_list = list(set(code_list) & set(result))
    return code_list

@st.cache_data
def get_stock_prices(tickers):
    all_data = pd.DataFrame()
    test_data = pd.DataFrame()
    no_data = []
    for i in tickers:
        try:
            test_data = yf.download(i + '.IS', start=dt.datetime(2015, 1, 1), end=dt.date.today(), interval="1wk")
            test_data['Symbol'] = i
            all_data = pd.concat([all_data, test_data])
            clear_output(wait=True)
        except:
            no_data.append(i)
            print(f"no_data run for {i}")
        clear_output(wait=True)
    unique_symbols = all_data['Symbol'].unique()
    print(f"Successfully retrieved {len(unique_symbols)} different stock data.")
    print(f"Failed to retrieve prices {len(no_data)} stocks")
    print(dt.date.today())
    return all_data, no_data

@st.cache_data
def get_stock_prices_for_test(tickers):
    all_data = pd.DataFrame()
    test_data = pd.DataFrame()
    no_data = []
    for i in tickers:
        try:
            test_data = yf.download(i + '.IS', start=dt.datetime(2023, 1, 1), end=dt.date.today() + dt.timedelta(days=1))
            test_data['Symbol'] = i
            all_data = pd.concat([all_data, test_data])
            clear_output(wait=True)
        except:
            no_data.append(i)
            print(f"no_data run for {i}")
        clear_output(wait=True)
    unique_symbols = all_data['Symbol'].unique()
    print(f"Successfully retrieved {len(unique_symbols)} different stock data.")
    print(f"Failed to retrieve prices {len(no_data)} stocks")
    return all_data, no_data

@st.cache_data
def get_all_prices(interval="1wk"):
    print("Data retrieval started...")
    start_time = time.time()
    tickers = process_stock_data()
    all_data, no_data = get_stock_prices(tickers)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("Data retrieval ended...")
    print("Elapsed time of the code: {} minutes {} seconds".format(minutes, seconds))
    return all_data, tickers

@st.cache_data
def get_all_prices_for_test(interval="1d"):
    print("Data retrieval started...")
    start_time = time.time()
    tickers = process_stock_data()
    all_data, no_data = get_stock_prices_for_test(tickers)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("Data retrieval ended...")
    print(all_data)
    print("Elapsed time of the code: {} minutes {} seconds".format(minutes, seconds))
    return all_data

def calculate_returns(dataframe, symbol_col='Symbol', close_col='Close', open_col='Open', close_shift=True):
    dataframe['Actual_Return_Last_7_Day'] = dataframe.groupby(symbol_col)[close_col].pct_change()
    if close_shift:
        dataframe['Close_Shifted'] = dataframe.groupby(symbol_col)[close_col].transform(lambda x: x.shift(-1))
    else:
        dataframe['Close_Shifted'] = dataframe.groupby(symbol_col)[close_col].transform(lambda x: x.shift(0))
    dataframe['Actual_Return_Next_7_Days'] = (
                (dataframe['Close_Shifted'] - dataframe[open_col]) / dataframe[open_col] * 100)
    dataframe['Actual_Direction_Next_7_Days'] = np.where(dataframe['Actual_Return_Next_7_Days'] > 0, 1, 0)
    # last_row = dataframe.iloc[-1]
    dataframe = dataframe.dropna().copy()
    # last_index = dataframe.index.max()
    # new_index = last_index + pd.DateOffset(days=1)
    # dataframe.loc[new_index] = last_row
    return dataframe

@st.cache_data
def calculate_technical_indicators(dataframe, long_window=20, short_window=5, window=14, high_column='High',
                                   low_column='Low', close_column='Close', volume_column='Volume'):
    symbol_list = dataframe['Symbol'].unique().tolist()
    stock_data = pd.DataFrame()
    for stock_code in symbol_list:
        data = dataframe.groupby('Symbol').get_group(stock_code)
        data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume",
                                   fillna=True)
        print(f"Calculated technical indicators for {stock_code} ...")
        stock_data = pd.concat([stock_data, data])
        clear_output(wait=True)
    print("Finished.")
    return stock_data

@st.cache_data
def calculate_technical_indicators_for_test(dataframe, long_window=20, short_window=5, window=14, high_column='High',
                                   low_column='Low', close_column='Close', volume_column='Volume'):
    symbol_list = dataframe['Symbol'].unique().tolist()
    stock_data = pd.DataFrame()
    for stock_code in symbol_list:
        data = dataframe.groupby('Symbol').get_group(stock_code)
        data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume",
                                   fillna=False)
        data.fillna(0, inplace=True)
        print(f"Calculated technical indicators for {stock_code} ...")
        stock_data = pd.concat([stock_data, data])
        clear_output(wait=True)
    print("Finished.")
    return stock_data
################################################
# Functions
################################################
menu1 = "Run SSStoB"
menu2 = "See Current Predictions"
menu3 = "See Past Predictions"
menu4 = "Stock Analysis"
menu5 = "Rerun Application"
features = ['others_dr',
            'volatility_dch',
            'trend_ichimoku_conv',
            'volatility_kcc',
            'trend_ichimoku_b',
            'volume_vwap',
            'trend_visual_ichimoku_a',
            'trend_ema_slow',
            'trend_visual_ichimoku_b',
            'volatility_dcm',
            'volatility_bbh',
            'trend_psar_down',
            'others_dlr',
            'momentum_ppo',
            'trend_aroon_ind',
            'trend_sma_fast',
            'volume_obv',
            'volatility_kcl',
            'volume_nvi',
            'volatility_kch']

# Main menu
menu_options = [menu1, menu2, menu3, menu4]
menu_selection = st.sidebar.radio('', menu_options)

# Submenu
if menu_selection == menu1:
    st.sidebar.title(menu1)
    st.text("1. Started SSStoB")
    main_data, tickers = get_all_prices()
    st.text("2. Downloaded Prices")
    main_data = calculate_returns(main_data)
    st.text("3. Calculated Returns")
    main_data = calculate_technical_indicators(main_data)
    st.text("4. Calculated Indicators")
    st.session_state['main_data'] = main_data
    st.session_state['tickers'] = tickers
    print(tickers)
    for ticker in tickers:
        a = f"models/{ticker}.pkl"
        st.session_state[f"m_{ticker}"] = joblib.load(a)
    st.text("5. Models Have Setup for Each Stock")
    test_data = get_all_prices_for_test(interval="1d")
    st.text("6. Downloaded Daily Prices for Test")
    test_data = calculate_returns(test_data, close_shift=False)
    st.text("7. Calculated Returns for Test")
    test_data = calculate_technical_indicators_for_test(test_data)
    st.text("8. Calculated Indicators for Test")
    st.session_state['test_data'] = test_data
    st.success("Successfully Completed!")
    st.warning("Please use the menu options on the left side of the screen to view the predictions.")

elif menu_selection == menu2:
    st.sidebar.title(menu2)
    if 'main_data' not in st.session_state:
        st.warning("Please run SSStoB first!")
    else:
        results = pd.DataFrame(columns=["Stock", "Prediction", "Daily Return", "SMA", "EMA", "MACD", "RSI"])
        # selected_date = pd.Timestamp.today() - timedelta(days=pd.Timestamp.today().weekday())
        print(st.session_state["test_data"])
        selected_date = st.session_state["test_data"].index.max()
        selected_date = selected_date.replace(hour=0, minute=0, second=0, microsecond=0)
        st.title(f"Pedictions as of {selected_date.strftime('%d.%m.%Y')}")
        for ticker in st.session_state['tickers']:
            test = st.session_state['test_data'].groupby('Symbol').get_group(ticker)
            ham_data = test
            test = test[features]
            model = st.session_state[f"m_{ticker}"]
            test = test[test.index == selected_date]
            x = model.predict_proba(test)
            result = {"Stock": ticker,
                      "Prediction": x[0][1],
                      "Daily Return": ham_data.loc[selected_date, "Actual_Return_Last_7_Day"]*100,
                      "SMA": ham_data.loc[selected_date, "trend_sma_slow"],
                      "EMA": ham_data.loc[selected_date, "trend_ema_slow"],
                      "MACD": ham_data.loc[selected_date, "trend_macd"],
                      "RSI": ham_data.loc[selected_date, "momentum_rsi"]}

            result = pd.DataFrame(result, index=[0])
            results = pd.concat([results, result], ignore_index=True)
        results = results.sort_values("Prediction", ascending=False).reset_index(drop=True)
        results.index = results.index + 1
        st.dataframe(results.head(7))
        comments = {
            'Prediction Range': ['0.65 or higher', 'Below 0.5', 'Between 0.5 and 0.65'],
            'Recommendation': ['Positive Direction', 'Negative Direction', 'Neutral'],
            'Description': [
                "The model indicates a positive outlook signal with a prediction of 0.65 or higher, suggesting a high probability of positive performance in the near future.",
                "The model's prediction falls below 0.5, indicating a negative outlook for the short-term performance of the asset. Caution may be warranted.",
                "The model provides a neutral outlook for the asset with a prediction between 0.5 and 0.65. Monitoring market conditions is advisable."
            ]
        }
        explanation = pd.DataFrame(comments)
        st.title('Model Prediction Explanations')
        st.markdown(explanation.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        st.title('Disclosure')
        st.text_area("Disclosure", "The information provided is not investment advice. It is general in nature and may not be suitable for your specific financial situation and preferences. Therefore, making investment decisions solely based on this information may not meet your expectations. The information is for general informational purposes only and does not constitute a buy-sell recommendation or return promise for any investment instrument. It is important to note that this content may not provide sufficient information to support trading decisions. The content owner is not liable for the outcomes of future investments or commercial transactions based on the information and opinions provided. The accuracy and completeness of the prices, data, and information cannot be guaranteed, and the content is subject to change without notice. The data is sourced from believed reliable sources, and any errors resulting from their use are not the responsibility of the content producer.",
                     200)


elif menu_selection == menu3:
    st.sidebar.title(menu3)
    if 'main_data' not in st.session_state:
        st.warning("Please run SSStoB first!")
    else:
        results = pd.DataFrame(columns=["Stock", "Prediction","Actual", "Actual Return", "Past Period Return", "SMA", "EMA", "MACD", "RSI"])
        selected_date = st.selectbox("Select a date (Beginning of the weeks)", st.session_state["main_data"].index.unique().sort_values(ascending=False))
        st.title(f"Historical Pedictions as of {selected_date.strftime('%d.%m.%Y')}")
        for ticker in st.session_state['tickers']:
            test = st.session_state["main_data"].groupby('Symbol').get_group(ticker)
            ham_data = test
            test = test[features]
            model = st.session_state[f"m_{ticker}"]
            test = test[test.index == selected_date]
            x = model.predict_proba(test)
            result = {"Stock": ticker,
                      "Prediction": x[0][1],
                      "Actual": ham_data.loc[selected_date, "Actual_Direction_Next_7_Days"],
                      "Actual Return": ham_data.loc[selected_date, "Actual_Return_Next_7_Days"],
                      "Past Period Return": ham_data.loc[selected_date, "Actual_Return_Last_7_Day"] * 100,
                      "SMA": ham_data.loc[selected_date, "trend_sma_slow"],
                      "EMA": ham_data.loc[selected_date, "trend_ema_slow"],
                      "MACD": ham_data.loc[selected_date, "trend_macd"],
                      "RSI": ham_data.loc[selected_date, "momentum_rsi"]}
            result = pd.DataFrame(result, index=[0])
            results = pd.concat([results, result], ignore_index=True)
        #results = results.append(result, ignore_index=True)
        results = results.sort_values("Prediction", ascending=False).reset_index(drop=True)
        results.index = results.index + 1
        st.dataframe(results.head(7))
        st.text_area("Disclosure", "The information provided is not investment advice. It is general in nature and may not be suitable for your specific financial situation and preferences. Therefore, making investment decisions solely based on this information may not meet your expectations. The information is for general informational purposes only and does not constitute a buy-sell recommendation or return promise for any investment instrument. It is important to note that this content may not provide sufficient information to support trading decisions. The content owner is not liable for the outcomes of future investments or commercial transactions based on the information and opinions provided. The accuracy and completeness of the prices, data, and information cannot be guaranteed, and the content is subject to change without notice. The data is sourced from believed reliable sources, and any errors resulting from their use are not the responsibility of the content producer.",
                     400)

elif menu_selection == menu4:
    st.sidebar.title(menu4)
    if 'main_data' not in st.session_state:
        st.warning("Please run SSStoB first!")
    else:
        submenu_options = st.session_state['tickers']
        submenu_selection = st.sidebar.radio('Select an option', submenu_options)
        test = st.session_state["test_data"].groupby('Symbol').get_group(submenu_selection)
        fig = px.line(test["Close"], title=submenu_selection)
        fig.update_layout(width=1500, height=600)
        st.plotly_chart(fig, use_container_width=True)

elif menu_selection == menu5:
    st.sidebar.title(menu5)
    st.text("1. Started SSStoB")
    st.cache_data.clear()
    main_data, tickers = get_all_prices()
    st.text("2. Downloaded Prices")
    main_data = calculate_returns(main_data)
    st.text("3. Calculated Returns")
    main_data = calculate_technical_indicators(main_data)
    st.text("4. Calculated Indicators")
    st.session_state['main_data'] = main_data
    st.session_state['tickers'] = tickers
    print(tickers)
    for ticker in tickers:
        a = f"models/{ticker}.pkl"
        st.session_state[f"m_{ticker}"] = joblib.load(a)
    st.text("5. Models Have Setup for Each Stock")
    test_data = get_all_prices_for_test(interval="1d")
    st.text("6. Downloaded Daily Prices for Test")
    test_data = calculate_returns(test_data, close_shift=False)
    st.text("7. Calculated Returns for Test")
    test_data = calculate_technical_indicators_for_test(test_data)
    st.text("8. Calculated Indicators for Test")
    st.session_state['test_data'] = test_data
    st.success("Successfully Completed!")
    st.warning("Please use the menu options on the left side of the screen to view the predictions.")
