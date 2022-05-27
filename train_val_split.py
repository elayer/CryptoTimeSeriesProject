
def train_val_split(x, y, train_pct = 0.8):
    """Given the input x and output labels y, splits the dataset into train, validation and test datasets
    Args:
        x (list): A list of all the input sequences
        y (list): A list of all the outputs (floats)
        train_pct (float): [% of data in the test set]
    """
    # Perform a train test split (It will be sequential here since we're working with time series data)
    N = len(x)
    
    X_train = x[:int(train_pct * N)]
    y_train = y[:int(train_pct * N)]

    X_val = x[int(train_pct * N):]
    y_val = y[int(train_pct * N):]

    return (X_train, y_train, X_val, y_val)

