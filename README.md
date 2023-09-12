# SSStoB: Smart Stock Selector To Buy
SSStoB is a Streamlit application that utilizes a machine learning model to predict the direction of stock prices and visualizes the prediction results. This application allows you to view prediction results on a specific date and explore historical prediction outcomes.

**Access the Live Application:** [SSStoB Live Application](https://ssstob.streamlit.app/)

## How It Works
The SSStoB application consists of the following key steps:

### Data Preparation: 
Initially, several data processing functions are employed to prepare the stock market data. The data is limited to specific years, and eligible stocks are identified.

### Fetching Stock Prices: 
The price data for the identified stocks is retrieved using the Yahoo Finance API. The downloaded data is then processed and made ready for use.

### Calculation of Technical Indicators: 
Technical analysis indicators are computed and added to the data frame. These indicators aid in making predictions.

### Preparation of Machine Learning Models: 
A machine learning model is prepared for each stock, and these models are used to predict the price direction of the respective stock.

### Displaying Predictions: 
The application presents users with current prediction results based on the selected date. These results are sorted by the predicted price direction and displayed to the user.

### Exploring Historical Prediction Results: 
Users can examine historical prediction outcomes made on specific dates. This is useful for evaluating past model performance.

### Graphical Analysis: 
Users can visualize the price chart of a selected stock. This chart visually displays the price movements of the stock.

## Usage
Once the application starts, you can select your desired option from the menu on the left side:

### Run SSStoB: 
Prepares machine learning models and processes data for predictions.
### See Current Predictions: 
Displays prediction results as of the current date.
### See Past Predictions: 
Allows you to select a date to review historical prediction results.
### Stock Analysis: 
Provides stock price chart analysis for a specific stock.

## Notes and Disclaimers
This application is not investment advice. It is for general informational purposes only.
Accuracy and completeness of the data cannot be guaranteed.
Always seek expert advice before making any investment decisions.

## Contribution
If you wish to contribute to this project, please fork the repository and submit your improvements as pull requests. Any contributions and feedback are highly appreciated.
