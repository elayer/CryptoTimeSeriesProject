
import torch
import numpy as np
from torch.utils.data import DataLoader

def get_preds(generator, model):
    """Given a pytorch neural network model and a generator object, extracts predictions and returns the same
    Args:
        generator (object): A pytorch dataloader which holds inputs on which we wanna predict
        model (object): A pytorch model with which we will predict stock prices on input data
    """
    all_preds = []
    all_labels = []
    all_inputs = []
    
    for xb, yb in generator:
        i = xb.unsqueeze(0)
        o = model.predict(i)
        all_preds.append(o)
        all_inputs.append(i)
        all_labels.append(yb)
    return (torch.cat(all_preds), torch.cat(all_labels))

