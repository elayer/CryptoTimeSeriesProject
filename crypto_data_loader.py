
from torch.utils.data import DataLoader
from CryptoDataset import CryptoDataset

def crypto_data_loader(x, y, params):
    """Given the inputs, labels and dataloader parameters, returns a pytorch dataloader
    Args:
        x (list): [inputs list]
        y (list): [target variable list]
        params (dict): [Parameters pertaining to dataloader eg. batch size]
    """
    training_set = CryptoDataset(x, y)
    data_loader = DataLoader(training_set, **params) #torch.utils.data.DataLoader
    return data_loader

