#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import plotly.figure_factory as ff
sns.set()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
#import datetime as dt

import time
from datetime import datetime
from datetime import date
from datetime import timedelta

import streamlit as st

from pmdarima.arima import auto_arima

import yfinance
from yahoo_fin.stock_info import get_data
from Time_Series_Functions import get_close_column, get_volume_column, create_ticker_table
from Auto_Time_Series_Functions import get_close_column, get_volume_column, auto_create_ticker_table

#importing created functions
from crypto_data_loader import crypto_data_loader
from CryptoDataset import CryptoDataset
from CryptoModel import CryptoModel
from generate_data import generate_data
from get_preds import get_preds
from standardize_data import standardize_data
from train_val_split import train_val_split
from visualize_results import visualize_results


import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Cryptocurrency Analyzer")

#We can stick with StandardScaler for now
def arima_standardize(df, scale = None, train = True):
    '''Standardizes the data used within the Auto ARIMA model. Returns the 
       scaler object in order to inverse transform the data for plotting
       
       args:
           df (Series): The column to be transformed
           scale (scaler Object): The scaler to use in order to transform test data
           train (boolean): The boolean flag to use when transforming train or test data
       returns:
           scaled_df (Series): The transformed data
           sc (scaler Object): The scaler object to use when transforming test data
       
    '''
    if train:
        sc = StandardScaler()
        trans = sc.fit_transform(df.values.reshape(-1, 1))
        trans_values = [float(item) for item in trans]
        date_indices = df.index
                
        scaled_df = pd.Series(trans_values, index=date_indices)
        return scaled_df, sc
            
    else:
        trans = scale.transform(df.values.reshape(-1, 1))
        trans_values = [float(item) for item in trans]
        date_indices = df.index
                
        scaled_df = pd.Series(trans_values, index=date_indices)
        return scaled_df

def log_and_shift(series):
    #Performs log and difference transformations on data
    series_log = np.log(series)
    series_diff_log = series_log.shift()
    return series_diff_log[1:]

def rev_log_and_shift(series):
    #Reverses the transformations made from the log_and_shift function
    series_rev_exp = np.exp(series.shift(-1))
    series_rev_exp.dropna(axis=0, inplace=True)
    return series_rev_exp

def train_loop(train_data_loader, valid_data_loader, model, loss_fn, optimizer, n_epochs = 100):
    # Track the losses across epochs
    train_losses = []
    valid_losses = []
    
    #looping through each epoch
    for epoch in range(1, n_epochs + 1):
        ls = 0
        valid_ls = 0
        #Load the data from the data loader
        for xb, yb in train_data_loader:
            #Forward pass operation
            ips = xb.unsqueeze(0)
            targs = yb
            op = model(ips)
            
            #Backpropagate the errors through the network
            optimizer.zero_grad()
            loss = loss_fn(op, targs)
            loss.backward()
            optimizer.step()
            ls += (loss.item() / ips.shape[1])
        
        #Load data from the validation data loader and make predictions
        for xb, yb in valid_data_loader:
            ips = xb.unsqueeze(0)
            ops = model.predict(ips)
            vls = loss_fn(ops, yb)
            valid_ls += (vls.item() / xb.shape[1])

        #Using lambda function to generate mean squared error for the model
        rmse = lambda x: round(np.sqrt(x * 1.000), 3)
        train_losses.append(str(rmse(ls)))
        valid_losses.append(str(rmse(valid_ls)))
        
        #Print the total loss for every tenth epoch
        if (epoch % 10 == 0) or (epoch == 1):
            print(f"Epoch {str(epoch):<4}/{str(n_epochs):<4} | Train Loss: {train_losses[-1]:<8}| Validation Loss: {valid_losses[-1]:<8}")

    # Make predictions on train, validation and test data and plot 
    # the predictions along with the true values 
    to_numpy = lambda x, y: (x.squeeze(0).numpy(), y.squeeze(0).numpy())
    train_preds, train_labels = get_preds(train_data_loader, model)
    train_preds, train_labels = to_numpy(train_preds, train_labels)

    val_preds, val_labels = get_preds(valid_data_loader, model)
    val_preds, val_labels = to_numpy(val_preds, val_labels)
    
    return train_preds, train_labels, val_preds, val_labels

#arima_container = st.container()
#lstm_container = st.container()
st.title("Cryptocurrency Analyzer")

arima_col, lstm_col = st.columns([6,6])

#We may have to make some kind of boolean for which models to process

with arima_col:

    st.header("Auto ARIMA Model")

    #all top listed cryptocurrency tickers on Yahoo! Finance
    tickers = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD', 'HEX-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD', 'BUSD-USD', 'UST-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'SHIB-USD', 'WBTC-USD', 'LUNA1-USD', 'WTRX-USD', 'STETH-USD', 'TRX-USD', 'DAI-USD', 'MATIC-USD', 'NEAR-USD', 'CRO-USD', 'LTC-USD']


    #Choosing a start date to generate data from
    start_date = st.date_input(label = "Select start date from which to get historical data", 
                               value = date(2020, 1, 1),
                               min_value = date(2017, 1, 1),
                               max_value = date.today(),
                               key=1)

    #Selecting tickers to use for the Auto ARIMA model
    symbols = st.multiselect("Select tickers to analyze: ",
                             options = tickers,
                             key=3)
    
    
    #st.write(symbols)
    
    
    try:
        df = auto_create_ticker_table(symbols = symbols, start_date = start_date)
        close_cols = df.shape[1] // 3

        display_df = pd.DataFrame()
        for j in range(close_cols):
            temp = df.iloc[:,j]
            display_df = pd.concat([display_df, temp], axis=1)

        #Using style bars to allow for users to make decisions with the tickers they choose
        display_df.style.background_gradient(cmap='GnBu')
        st.dataframe(display_df.style.bar(color=["tomato", "lime"], align="mid"))

        #Keeping only the close columns created from the original df
        new_df = pd.DataFrame()
        for j in range(close_cols):
            transformed = df.iloc[:,j] #log_and_shift(df.iloc[:,j]) df.iloc[:,j] #should we try to StandardScaler it?
            new_df = pd.concat([new_df, transformed], axis=1)
    
    except TypeError:
        st.write('Please enter tickers')
        
    #Try/Except statements below are placeholders for errors popping up if the user hasn't entered data yet
    try:
        selection_cols = new_df.columns.tolist()
    except NameError:
        st.write('Waiting for tickers...')

    try:
        endo = st.selectbox("Choose one column to be Endogenous: ",
                            options = selection_cols)
        
        #Choosing the date from which predictions are made 
        valid_date = st.date_input(label = "Select the Date from which to make Predictions: ", 
                                   value = date(2020, 1, 1),
                                   min_value = start_date,
                                   max_value = date.today())
    except NameError:
        st.write('Waiting for tickers...')
        
    #Including option for user to choose if Auto ARIMA considers seasonal components
    seasonal_options = [True, False]    
    seasonal_switch = st.selectbox("Include seasonal component in Auto ARIMA?",
                                options = seasonal_options,
                                key=5)
    
    seasonal_d = st.slider(label = "Set the order of seasonal differencing: ",
                        value = 0,
                        min_value = 0,
                        max_value = 5,
                        step = 1)
    
    seasonal_vals = [1, 4, 12]
    
    seasonal_p = st.selectbox("Choose the period for seasonal differencing: ",
                            options = seasonal_vals)
    
        
    if st.button('Click to run Auto Arima'):
    
        endo_df = new_df[endo]
        new_df.drop(columns=endo, inplace=True)

        #Create and standardize the endogenous variable data
        df_train, df_test = endo_df[:valid_date], endo_df[valid_date:]
        
        df_train, sc = arima_standardize(df_train)
        df_test = arima_standardize(df_test, sc, False)
    
        #Initializing the dataframes for exogenous variables
        exog_data = pd.DataFrame(index = df_train.index)
        exog_test = pd.DataFrame(index = df_test.index)

        exog_range = new_df.shape[1]
        exog_columns = new_df.columns.tolist()
        
        #list to store scaler objects for each exogenous variable
        scalers = [] 

        #Transforming all exogenous columns to be on the same scale as the endogenous variable.
        for col in exog_columns:
            train_temp = new_df[col][:valid_date]
            scaler = StandardScaler()                 
            train_temp = scaler.fit_transform(train_temp.values.reshape(-1, 1)) 
            train_val_list = [float(elem) for elem in train_temp]
            train_temp = pd.Series(train_val_list)
            train_temp.index = df_train.index
            scalers.append(scaler)                         
            exog_data = pd.concat([exog_data, train_temp], axis=1)
    
        for i, col in enumerate(exog_columns):
            test_temp = new_df[col][valid_date:]
            scaler = scalers[i]                         
            test_temp = scaler.transform(test_temp.values.reshape(-1, 1))      
            test_val_list = [float(elem) for elem in test_temp]
            test_temp = pd.Series(test_val_list, index = df_test.index)
            exog_test = pd.concat([exog_test, test_temp], axis=1)
            
        img_pth = f"{endo}_pred_plot.png"

        #Creating the model with a wide range of parameters for more models to be generated
        auto_model = auto_arima(df_train, exogenous = exog_data, 
                                start_p = 1, start_q = 1, 
                                test = 'adf',
                                d = None,
                                seasonal = seasonal_switch, #True
                                start_P = 0,
                                D = seasonal_d,
                                trace = True,
                                error_action = 'ignore',
                                suppress_warnings = True,
                                stepwise = True,
                                m = seasonal_p, max_q = 8, max_P = 8, max_Q = 8)

        df_auto_pred = pd.DataFrame(auto_model.predict(n_periods = len(df_test),
                                                       exogenous = exog_test),
                                                       index = df_test.index)

        #df_auto_pred = np.exp(df_auto_pred.shift(-1))
        #df_test = np.exp(df_test.shift(-1))

        #df_auto_pred.dropna(axis=0, inplace=True)
        #df_test.dropna(axis=0, inplace=True)
        
        #Reversing the transform to scale back the data to its original scale
        df_auto_pred_rev = sc.inverse_transform(df_auto_pred.values.reshape(-1, 1))
        df_test_rev = sc.inverse_transform(df_test.values.reshape(-1, 1))

        pred_values = [float(item) for item in df_auto_pred_rev]
        test_values = [float(item) for item in df_test_rev]
        
        date_indices = df_auto_pred.index
        
        df_auto_pred = pd.Series(pred_values, index=date_indices)
        df_test = pd.Series(test_values, index=date_indices)
        
        #new stuff:###############
        df_train_hold, df_test_hold = endo_df[:valid_date], endo_df[valid_date:]
        
        df_test = pd.concat([df_train_hold, df_test], axis=0)
        df_auto_pred = pd.concat([df_train_hold, df_auto_pred], axis=0)
        ###############
        

        #Combining the true values and predictions
        plotly_df = pd.concat([df_test, df_auto_pred], axis=1)
        plotly_df.columns = ['True Values', 'Predicted Values']
        

        #Creating the plotly graph to plot in the application
        fig = px.line(plotly_df, title=f'Auto Arima Model Predictions for {endo[:3]}', width=1100, height=600,
                      labels = {'index':'Date', 'value':'Price (USD)'})
        
        fig.add_scatter(x=df_train_hold.index, y=df_train_hold.values, mode='lines', marker=dict(color="green"), name='Pre-predicted Data')
        
        fig.add_vline(x=valid_date, line_width=3, line_color="magenta")
        
        st.plotly_chart(fig)
        
        today = date.today()
        prev_day = pd.to_datetime(today - timedelta(days=1)).strftime("%Y-%m-%d")
        candleplot = get_data(endo[:7], start_date = start_date, end_date = prev_day, index_as_date = True, interval = '1d')
        
        dates = candleplot.index
    
        figstick = go.Figure(data=[go.Candlestick(x=candleplot.index,
                                              open=candleplot['open'],
                                              high=candleplot['high'],
                                              low=candleplot['low'],
                                              close=candleplot['close'])])
        figstick.update_layout(
                    width=1100,
                    height=600)
        figstick.update_xaxes(title_text='Date')
        figstick.update_yaxes(title_text='Price (USD)')
        figstick.update_layout(xaxis_rangeslider_visible=True,
                 title = f"{endo[:3]} Candlestick Plot")
        
        st.plotly_chart(figstick)
 

with lstm_col:   
    
    st.header("LSTM Model")

    tickers_lstm = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD', 'HEX-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD', 'BUSD-USD', 'UST-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'SHIB-USD', 'WBTC-USD', 'LUNA1-USD', 'WTRX-USD', 'STETH-USD', 'TRX-USD', 'DAI-USD', 'MATIC-USD', 'NEAR-USD', 'CRO-USD', 'LTC-USD']


    start_date_lstm = st.date_input(label = "Select start date from which to get historical data", 
                                    value = date(2020, 1, 1),
                                    min_value = date(2017, 1, 1),
                                    max_value = date.today(),
                                    key=2)

    symbol_lstm = st.selectbox("Select tickers to analyze: ",
                                options = tickers_lstm,
                                key=4)
    
    n_lags = st.slider(label = "Number of lags for which predictions are based on: ",
                        value = 5,
                        min_value = 1,
                        max_value = 25,
                        step = 5)

    train_pct = st.slider(label = "Percentage of data to use for training: ",
                        value = 0.8,
                        min_value = 0.7,
                        max_value = 0.9,
                        step = 0.1)

    hidden_dim = st.slider(label = "Number of hidden dimensions included in the LSTM model: ",
                        value = 80,
                        min_value = 20,
                        max_value = 150,
                        step = 5)
    
    rnn_layers = st.slider(label = "Number of RNN layers used within the model: ",
                        value = 3,
                        min_value = 1,
                        max_value = 10,
                        step = 1)

    dropout = st.slider(label = "Percentage of neurons to drop during training: ",
                        value = 0.3,
                        min_value = 0.1,
                        max_value = 0.7,
                        step = 0.1)

    n_epochs = st.slider(label = "Number of epochs to use for model training: ",
                        value = 500,
                        min_value = 100,
                        max_value = 3000,
                        step = 100)    
    
    batch_size = st.slider(label = "Batch size of the model for each iteration of training: ",
                        value = 8,
                        min_value = 2,
                        max_value = 32,
                        step = 2)

    lrs = [0.001, 0.005, 0.007, 0.01, 0.05, 0.07]
    learning_rate = st.selectbox("Select a learning rate value for the gradient descent calculations: ",
                                options = lrs,
                                key=6)
    
    
    symbol_lstm_list = []
    symbol_lstm_list.append(symbol_lstm)

    if st.button('Click to run LSTM'):
    

        df_lstm = auto_create_ticker_table(symbols = symbol_lstm_list, start_date = start_date_lstm)
        close_cols_lstm = df_lstm.shape[1] // 3

        display_df_lstm = pd.DataFrame()
        for j in range(close_cols_lstm):
            temp_lstm = df_lstm.iloc[:,j]
            display_df_lstm = pd.concat([display_df_lstm, temp_lstm], axis=1)

        #Using style bars to allow for users to make decisions with the tickers they choose
        display_df_lstm.style.background_gradient(cmap='PuBu')
        st.write(display_df_lstm.style.bar(color=["tomato", "lime"], align="mid"))
        
        df_lstm['Date'] = df_lstm.index.date
        
        #right now it has hard coded 5 lags
        inputs, labels, dates = generate_data(df_lstm, symbol_lstm_list[0]+'_Close', 'Date', n_lags) 

        N = len(inputs)

        #right now has hard coded 0.8 split of data
        X_train, y_train, X_val, y_val = train_val_split(inputs, labels, train_pct) 
        
        config = {'start_date': start_date_lstm,
          'end_date': date.today(),
          'hidden_dim': hidden_dim,
          'rnn_layers': rnn_layers,
          'dropout': dropout,
          'n_epochs': n_epochs,
          'batch_size': batch_size,
          'n_lags': n_lags,
          'learning_rate': learning_rate } #not 0.001
        
        params = {"batch_size": config['batch_size'],
         "shuffle": False,
         "num_workers": 0}
        device = "cpu"
        
        # Standardize the data to bring the inputs on a uniform scale
        X_train, scaler_ = standardize_data(X_train, train = True)
        X_val = standardize_data(X_val, scaler_) 

        # Create dataloaders for both training and validation datasets
        train_data_loader = crypto_data_loader(X_train, y_train, params)
        valid_data_loader = crypto_data_loader(X_val, y_val, params)

        # Create the model
        model = CryptoModel(config['n_lags'], config['hidden_dim'], config['rnn_layers'], config['dropout']).to(device)

        # Define the loss function and the optimizer
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'])
        
        train_preds, train_labels, val_preds, val_labels = train_loop(train_data_loader, valid_data_loader, model, loss_fn, optimizer, n_epochs = config['n_epochs'])
        
        preds, labels = (train_preds, val_preds), (train_labels, val_labels)
        
        train_preds, val_preds = preds[0], preds[1]
        train_labels, val_labels = labels[0], labels[1]

        # Format the predictions into a dataframe and save them to a file in the predictions folder
        all_preds = np.concatenate((train_preds,val_preds))
        all_labels = np.concatenate((train_labels,val_labels))
        flags = ["train"] * len(train_labels) + ["valid"] * len(val_labels)

        df = pd.DataFrame([(x[0], y[0]) for x, y in zip(all_preds, all_labels)], columns = ["Predicted Values", "True Values"])
        df["Type"] = flags
        df.index = dates
        #df.to_csv(pred_pth)
        #st.write("Predictions for the last five timestamps...")
        #st.dataframe(df.tail(5), width = 600, height = 800)

        # Find out the first element which belongs to validation dataset to depict the same manually
        dt = None
        for idx, item in enumerate(df.Type):
            if item == "valid":
                dt = df.index[idx]
                break
    
        # Create the plot and save it to the path provided as an argument above
        img_pth_lstm = "pred_plot_lstm.png"
        
        lstm_plotly_df = df[['True Values', 'Predicted Values']]
        
        fig = px.line(lstm_plotly_df, title=f'LSTM Model Predictions for {symbol_lstm}', width=1100, height=600,
                      labels = {'index':'Date', 'value':'Price (USD)'})
        
        fig.add_vline(x=dt, line_width=3, line_color="magenta")

        #with auto_arima_plot:
        st.plotly_chart(fig)
        
        today = date.today()
        prev_day = pd.to_datetime(today - timedelta(days=1)).strftime("%Y-%m-%d")
        candleplot = get_data(symbol_lstm, start_date = start_date_lstm, end_date = prev_day, index_as_date = True, interval = '1d')
        
        dates = candleplot.index
    
        figstick = go.Figure(data=[go.Candlestick(x=candleplot.index,
                                              open=candleplot['open'],
                                              high=candleplot['high'],
                                              low=candleplot['low'],
                                              close=candleplot['close'])])
        figstick.update_layout(
                    width=1100,
                    height=600)
        figstick.update_xaxes(title_text='Date')
        figstick.update_yaxes(title_text='Price (USD)')
        figstick.update_layout(xaxis_rangeslider_visible=True,
                 title = f"{symbol_lstm} Candlestick Plot")
        
        st.plotly_chart(figstick)
    
