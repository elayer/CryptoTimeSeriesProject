## Cryptocurrency Time Series Modeling Project - Overview:

* Created functions to automatically scrape Yahoo! Finance cryptocurrency tickers that the user chooses, collecting data from the current day back to 
January 1st, 2017. 

* Explored various cryptocurrency trends and their possible influence by Russia's invasion of Ukraine.

* Using BTC data (BitCoin), explored various time series algorithms, such as AR, MA, ARCH and ARIMA to investigate the best model to graph the data. Eventually found that utilizing exogenous data with Auto ARIMA generates models that follow the data very closely.

* As a follow up, I then made an LSTM model with PyTorch to forecast cryptocurrency values.

* Lastly, I constructed a StreamLit app allowing users to create Auto ARIMA and LSTM models and juxtapose their predictive power. The app allows users to choose date ranges to collect data, which crypto tickers to analyze, and from a chosen date to make predictions.


## Code and Resources Used:

**Python Version:** 3.8.5

**Packages:** numpy, pandas, requests, beautiful soup, matplotlib, seaborn, statsmodels, streamlit, pyTorch, arch, scipy, yahoo_fin, pmdarima, datetime, sklearn

**Web Framework Requirements Command:** ```pip install -r requirements.txt```

## References:

* Various project structure and process elements were learned from Ken Jee's YouTube series: 
https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

* Helpful medium article on construacting an LSTM model using PyTorch:
https://medium.com/analytics-vidhya/pytorch-lstms-for-time-series-forecasting-of-indian-stocks-8a49157da8b9

* A lot of knowledge I gained from learning about time series data and models used to operate on it I learned from the following online resource:
https://365datascience.com

## Yahoo! Finance Scraping & Functionizing:

Created functions that prompt the user to enter what tickers they wish to analyze as well as between daily and business day frequency. Depending on what tickers are chosen, the function returns the following data from the earliest possible date where data exists for all tickers chosen:
*   Returns
*   Closing Price
*   Volume

## EDA
In the first picture, along with a few other crypto tickers I have uploaded to this repo, is the plot of a cryptocurrency ticker with respect to the date when Russia invaded Ukraine. Generally, a lot of the crypto tickers have stalled since or around this time. Of course there are other factors as well that lead to cryptocurrencies losing their value, such as inflation and supply chain issues. 

I also include an example of predictions for BTC using Auto ARIMA with exogenous variables. This approach works far better than any simple AR or MA model had. I also include candlestick plots in the StreamLit app to analyze differences, increases and declines for the chosen ticker to analyze.

![alt text](https://github.com/elayer/CryptoTimeSeriesProject/blob/main/btc_russia.png "BTC data around Russian Invasion Date")
![alt text](https://github.com/elayer/CryptoTimeSeriesProject/blob/main/BTC-USD_Close_pred_plot.png "BTC Auto ARIMA Model")
![alt text](https://github.com/elayer/CryptoTimeSeriesProject/blob/main/examplecandlestick.png "BTC Candlestick Plot")

## Model Building 
Using the tickers chosen in unison, I set up Auto ARIMA to analyze an endogenous variable of choice using exogenous variables of choice. 

I followed suit for the LSTM model by giving the user the choice to choose date ranges for which to collect data, a ticker to analyze, and a date from which to make predictions.

## StreamLit Application
This is the primary focus of the project. I allow for the user to input various parameters for both constructing an Auto ARIMA as well as LSTM models to plot predictions of cryptocurrencies using their model of choice. I include the posting of the dataframe to use in the construction of models using color to gauge the trend in value of tickers used in analysis.

![alt text](https://github.com/elayer/CryptoTimeSeriesProject/blob/main/CryptoAppTopPage.png "StreamLit App Top")
![alt text](https://github.com/elayer/CryptoTimeSeriesProject/blob/main/autoarima-btc-eth%26usdt.png "Example BTC Auto ARIMA App")
![alt text](https://github.com/elayer/CryptoTimeSeriesProject/blob/main/LSTMBTCexampleplot.png "Example BTC LSTM App")

## Future Improvements
I allow for the user to input some parameters for the Auto ARIMA model, but I could perhaps add more customization to this half of the application such as the max amount of AR and MA components to use within the model. 

In addition, some UI elements could also be improved if possible on Streamlit, such as how the charts from model runs are displayed on the screen.
