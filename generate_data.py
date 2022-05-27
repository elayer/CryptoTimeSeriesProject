

def generate_data(df, price_col, date_col, n_steps):
    """Reads the dataset and based on n_steps/lags to consider in the time series, creates input output pairs
    Args:
        df (DataFrame): The dataframe to acquire the selected column of data from
        price_col (str): [The name of column in the dataframe that holds the closing price for the stock]
        date_col (str): [The nameo oc column in the dataframe which holds dates values]
        n_steps (int): [Number of steps/ lags based on which prediction is made]
    """

    for idx in range(n_steps):
        df[f"lag_{idx + 1}"] = df[price_col].shift(periods = (idx + 1))
    
    # Create a dataframe which has only the lags and the date
    new_df = df[[date_col, price_col] + [f"lag_{x + 1}" for x in range(n_steps)]]
    new_df = new_df.iloc[n_steps:-1, :]

    # Get a list of dates for which these inputs and outputs are
    dates = list(new_df[date_col])

    # Create input and output pairs out of this new_df
    inputs = []
    outputs = []
    for entry in new_df.itertuples():
        i = entry[-n_steps:][::-1]
        o = entry[-(n_steps + 1)]
        inputs.append(i)
        outputs.append(o)

    return (inputs, outputs, dates)

