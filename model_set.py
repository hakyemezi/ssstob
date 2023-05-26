import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import time
from IPython.display import clear_output
from ta import add_all_ta_features
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
import joblib
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV, cross_validate, TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

def process_stock_data(n_years=5):
    df1 = pd.read_excel("ilkislem.xlsx", header=None, skiprows=[0])

    today = pd.Timestamp.today()
    n_years_ago = today - pd.DateOffset(years=n_years)
    df1 = df1[pd.to_datetime(df1.iloc[:, 4], errors="coerce", dayfirst=True) < n_years_ago]

    code_list = df1.iloc[:, 1].apply(lambda x: str(x).split(".")[0]).tolist()

    df2 = pd.read_excel('hisse20.xlsx')
    result = [x.split('.E')[0] for x in df2.iloc[:, 0].astype(str) if '.E' in x]

    code_list = list(set(code_list) & set(result))

    return code_list

def get_stock_prices(tickers):
    all_data = pd.DataFrame()
    test_data = pd.DataFrame()
    no_data = []

    for i in tickers:
        try:
            test_data = yf.download(i + '.IS', start=dt.datetime(1990, 1, 1), end=dt.date.today(), interval="1wk")
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

def get_all_prices():
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

def calculate_returns(dataframe, symbol_col='Symbol', close_col='Close', open_col='Open'):
    dataframe['Actual_Return_Last_7_Day'] = dataframe.groupby(symbol_col)[close_col].pct_change()
    dataframe['Close_Shifted'] = dataframe.groupby(symbol_col)[close_col].transform(lambda x: x.shift(-1))
    dataframe['Actual_Return_Next_7_Days'] = ((dataframe['Close_Shifted'] - dataframe[open_col]) / dataframe[open_col] * 100)
    dataframe['Actual_Direction_Next_7_Days'] = np.where(dataframe['Actual_Return_Next_7_Days'] > 0, 1, 0)
    dataframe = dataframe.dropna().copy()
    return dataframe

def calculate_technical_indicators(dataframe, long_window=20, short_window=5,window=14, high_column='High', low_column='Low',
                                   close_column='Close', volume_column='Volume'):
    symbol_list = dataframe['Symbol'].unique().tolist()
    stock_data = pd.DataFrame()
    for stock_code in symbol_list:
        data = dataframe.groupby('Symbol').get_group(stock_code)
        data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        print(f"Calculated technical indicators for {stock_code} ...")
        stock_data = pd.concat([stock_data, data])
        clear_output(wait=True)
    print("Finished.")
    return stock_data

all_data, tickers = get_all_prices()
stock_data = all_data.copy()
stock_data = calculate_returns(stock_data)
stock_data = calculate_technical_indicators(stock_data)

target_variables = ['volume_adi', 'volume_obv',
       'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em', 'volume_vpt',
       'volume_vwap', 'volume_mfi', 'volume_nvi', 'volatility_bbm',
       'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp',
       'volatility_bbhi', 'volatility_bbli', 'volatility_kcc',
       'volatility_kch', 'volatility_kcl', 'volatility_kcw', 'volatility_kcp',
       'volatility_kchi', 'volatility_kcli', 'volatility_dcl',
       'volatility_dch', 'volatility_dcm', 'volatility_dcw', 'volatility_dcp',
       'volatility_atr', 'volatility_ui', 'trend_macd', 'trend_macd_signal',
       'trend_macd_diff', 'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast',
       'trend_ema_slow', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg',
       'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index', 'trend_dpo',
       'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
       'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
       'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
       'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
       'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
       'trend_psar_down', 'trend_psar_up_indicator',
       'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
       'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
       'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
       'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
       'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
       'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
       'others_cr']

mms = MinMaxScaler()
for ticker in tickers:
    test = stock_data.groupby('Symbol').get_group(ticker)
    for target in target_variables:
        target_data = test[target].values.reshape(-1, 1)
        scaled_data = mms.fit_transform(target_data)
        test[target] = scaled_data.flatten()

def plot_importance(model, features, num=26, save=False):

    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)

    if save:
        plt.savefig('Importances.png')




def get_importance(model, features, num=22):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    feature_imp_sorted = feature_imp.sort_values(by="Value", ascending=False).head(num)
    return feature_imp_sorted

importance_df = pd.DataFrame()
for ticker in tickers:
    test = stock_data.groupby('Symbol').get_group(ticker)
    top_features = []
    for target in target_variables:
        target_data = test[target].values.reshape(-1, 1)
        scaled_data = mms.fit_transform(target_data)
        test[target] = scaled_data.flatten()

    split_date = '2022-12-31'
    train_data = test[test.index <= split_date]
    test_data = test[test.index > split_date]
    X_train = train_data[target_variables]
    y_train = train_data['Actual_Direction_Next_7_Days']
    X_test = test_data[target_variables]
    y_test = test_data['Actual_Direction_Next_7_Days']

    xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
    xgboost_model.fit(X_train, y_train)
    cv_results = cross_validate(xgboost_model, X_train, y_train, cv=TimeSeriesSplit(n_splits=3))

    # Modelin performansını değerlendirme
    accuracy = xgboost_model.score(X_test, y_test)
    predictions = xgboost_model.predict(X_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)

    print(f"as all features for {ticker} ...")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)

    # Değişken önem derecelerini DataFrame'e ekleme
    plot_importance(xgboost_model, X_train, num=len(X_train), save=True)
    plot_importance(xgboost_model, X_train, num=37, save = True)
    feature_imp = pd.DataFrame({'Value': xgboost_model.feature_importances_, 'Feature': X_train.columns})
    feature_imp = feature_imp.sort_values(by="Value", ascending=False).head(10)
    feature_imp['Ticker'] = ticker
    importance_df = pd.concat([importance_df, feature_imp])
    top_features = importance_df['Feature'].value_counts().nlargest(20).index.tolist()
    clear_output(wait=True)

for ticker in tickers:
    test = stock_data.groupby('Symbol').get_group(ticker)
    for target in top_features:
        target_data = test[target].values.reshape(-1, 1)
        scaled_data = mms.fit_transform(target_data)
        test[target] = scaled_data.flatten()

    split_date = '2022-12-31'
    train_data = test[test.index <= split_date]
    test_data = test[test.index > split_date]
    X_train = train_data[top_features]
    y_train = train_data['Actual_Direction_Next_7_Days']
    X_test = test_data[top_features]
    y_test = test_data['Actual_Direction_Next_7_Days']

    xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
    xgboost_model.fit(X_train, y_train)
    cv_results = cross_validate(xgboost_model, X_train, y_train, cv=TimeSeriesSplit(n_splits=3))

    # Modelin performansını değerlendirme
    accuracy = xgboost_model.score(X_test, y_test)
    predictions = xgboost_model.predict(X_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)

    print(f"as top features for {ticker} ...")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)

    xgboost_params = {"learning_rate": [0.01, 0.05, 0.1, 0.3, 0.5],
                      "max_depth": [3, 5, 7, 9],
                      "subsample" : [0.5, 0.7, 0.9],
                      "n_estimators": [100, 500, 700, 1000],
                      "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
                      "gamma": [0, 0.1, 0.5, 0.8]}

    xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=TimeSeriesSplit(n_splits=3), n_jobs=-1, verbose=True).fit(X_train, y_train)

    xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X_train, y_train)

    cv_results = cross_validate(xgboost_final, X_train, y_train, cv=TimeSeriesSplit(n_splits=3))

    # Modelin performansını değerlendirme
    accuracy = xgboost_final.score(X_test, y_test)
    predictions = xgboost_final.predict(X_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)

    # Modelin performansını yazdırma
    print(f"After Hyperparameters for {ticker} ...")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)

    joblib.dump(xgboost_final, f"{ticker}.pkl")

    clear_output(wait=True)
