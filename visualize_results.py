
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_results(preds, labels, dates):
    """Given predictions and labels for training and validation datasets, visualizes them in a plot
    Args:
        preds (list): Predicted values of the stock prices
        labels (list): True values of the stock prices
        dates (list): a list of dates as strings
    """
    train_preds, val_preds = preds[0], preds[1]
    train_labels, val_labels = labels[0], labels[1]

    #Format the predictions into a dataframe and save them to a file in the predictions folder
    all_preds = np.concatenate((train_preds,val_preds))
    all_labels = np.concatenate((train_labels,val_labels))
    flags = ["train"] * len(train_labels) + ["valid"] * len(val_labels)

    df = pd.DataFrame([(x[0], y[0]) for x, y in zip(all_preds, all_labels)], columns = ["Predictions", "Ground Truth"])
    df["Type"] = flags
    df.index = dates
    #df.to_csv(pred_pth)
    #st.write("Predictions for the last five timestamps...")
    #st.dataframe(df.tail(5), width = 600, height = 800)

    #Find out the first element which belongs to validation dataset to depict the same manually
    dt = None
    for idx, item in enumerate(df.Type):
        if item == "valid":
            dt = df.index[idx]
            break
    
    #Create the plot and save it to the path provided as an argument above
    
    plt.figure(figsize = (24,11))
    plt.plot(df.index, df["Predictions"], color = 'red')
    plt.plot(df.index, df["Ground Truth"], color = 'blue')
    plt.legend(["Predicted Values", "True Values"], fontsize = 16)
    plt.axvline(x = dt, c='magenta')
    plt.xticks(rotation = 90)
    plt.xlabel("Dates", fontsize = 16, weight='bold')
    plt.ylabel("Price", fontsize = 16, weight='bold')
    plt.title("Cryptocurrency LSTM Predictions (Train) | (Validation)", fontsize = 16, weight='bold')
    
    plt.show()
    

