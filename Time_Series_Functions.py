import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import time
from datetime import datetime
from datetime import date
from datetime import timedelta

import yfinance
from yahoo_fin.stock_info import get_data

def get_close_column(crypto_dict):
    
    """
    This function returns the closing prices for each chosen cryptocurrency ticker
    
    args:
        crypto_dict (dict): The dictionary containing all cryptocurrency data returned from Yahoo! finance
        
    returns:
        data (DataFrame): A dataframe containing only the closing prices of the chosen cryptocurrencies
    
    """
    keys = crypto_dict.keys()
    
    data = pd.DataFrame()
    
    for key in keys:
        data[key+'_Close'] = crypto_dict[key]['close']
        
    return data


def get_volume_column(crypto_dict):
    
    """
    This function returns the volume for each chosen cryptocurrency ticker
    
    args:
        crypto_dict (dict): The dictionary containing all cryptocurrency data returned from Yahoo! finance
        
    returns:
        data (DataFrame): A dataframe containing only the volumes of the chosen cryptocurrencies
    
    """
    keys = crypto_dict.keys()
    
    data = pd.DataFrame()
    
    for key in keys:
        data[key+'_Volume'] = crypto_dict[key]['volume']
        
    return data

def create_ticker_table(start_date="2017-01-01", interval = '1d'):
    
    """
    This function creates a table that contains closing prices and estimated returns for choice cryptocurrencies
    to be used for analysis, plotting, forecasting, etc.
    
    args:
        tickers (list): A default list containing the choice cryptocurrencies to evaluate
        start_date (date): The beginning date to start the table from
        end_date (date): The last date to generate data up to
        interval (str): The time interval to construct the table from
        
    returns:
        df (DataFrame): The data table containing the data of cryptocurrency closing prices and returns
    
    """
    
    tickers = []
    today = date.today()
    prev_day = pd.to_datetime(today - timedelta(days=1)).strftime("%Y-%m-%d")
    
    stopper = True
    num_closes = 0
    #print('Note: Bitcoin is already in the table by default') #In order to make the function diverse, it has been removed
    
    while stopper:
        crypto_str = input('Enter a ticker from Yahoo! Finance. Enter "No" to continue: ')
        if crypto_str.upper() == 'NO':
            break
            stopper = False
        else:
            tickers.append(crypto_str)
            
    crypto_data = {}
    
    rate = input("Do you want daily or business day data? Enter 'd' for daily and 'b' for business: ")
    rate = rate.lower() #Using lower method for assurance
    
    while rate != 'b' and rate != 'd':
        rate = input("Please enter again: ")
        rate = rate.lower()

    for ticker in tickers:
        crypto_data[ticker] = get_data(ticker, start_date = start_date, end_date = prev_day, index_as_date = True, interval = interval)
    
    closes = get_close_column(crypto_data)
    volumes = get_volume_column(crypto_data)
    
    df_crypto = pd.concat([closes, volumes], axis=1)
    
    close_cols_width = closes.shape[1]
    
    df_crypto = df_crypto.iloc[1:]
    df_crypto = df_crypto.asfreq(rate) #User may decide the rate between d and b
    df_crypto = df_crypto.fillna(method='ffill')
    
    types = closes.columns.tolist()
    
    #squared returns = volatility
    
    for tick in types:
        df_crypto[tick.split('_')[0]+'_Return'] = df_crypto[tick].pct_change(1).mul(100)
        
        #Squared Returns:
        #df_crypto['Squared_'+tick.split(':')[0]+'_Return'] = df_crypto[tick.split(':')[0]+'_Return'].mul(df_crypto[tick.split(':')[0]+'_Return'])
        
    null_elim = df_crypto.iloc[:,:close_cols_width].notna().idxmax().max()
    
    trunc_df_crypto = df_crypto.loc[null_elim:, :].copy() 
    trunc_df_crypto.fillna(0.0, inplace=True)

    print()
    print('********** Table Head **********')
    display(trunc_df_crypto.head(15))
    print('********** Table Tail **********')
    display(trunc_df_crypto.tail(15))
    
    return trunc_df_crypto