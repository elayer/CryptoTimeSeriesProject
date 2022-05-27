
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def standardize_data(X, scaler = None, train = False):
    """Given a list of input features, standardizes them to bring them onto a homogenous scale
    Args:
        X (dataframe): A dataframe of all the input values
        scaler (object, optional): A StandardScaler object that holds mean and std of a standardized dataset. 
        train (bool, optional): If False, means validation set to be loaded and SS needs to be passed to scale it. 
    """
    if train:
        scaler = StandardScaler()   
        new_X = scaler.fit_transform(X)
        return (new_X, scaler)
    else:
        new_X = scaler.transform(X)
        return new_X 

