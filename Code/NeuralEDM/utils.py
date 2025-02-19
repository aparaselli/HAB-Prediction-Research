import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_losses(train_losses, val_losses, fname):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument). 

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        fname (str): Name of the file to save the plot (without extension).

    Returns:
        None
    """

    # Create 'plots' directory if it doesn't exist

    if not os.path.isdir('plots'):
        os.mkdir('plots')

    # Plotting training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend() 

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/" + fname + ".png")

def get_data(data, E, tau, target=None):
    embd = EDM_embedding(E, tau, target)
    X,y = embd(data)
    return X, y

class EDM_embedding():
    def __init__(self, E, tau, target=None):
        self.E = E
        self.tau = tau
        self.target = target

    def create_embeddings(self, data, E, tau, target=None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"target column '{target}' not found in data")
        if E < 1:
            raise ValueError("E must be at least 1")
        
        X = []
        y = []
        
        target_values = data[target].values
        
        tau = abs(tau)
        
        history_needed = (E - 1) * tau
        
        for i in range(len(data)):
            if i < history_needed:
                continue
                
            embedding = []
            
            for e in range(E):
                lag_idx = i - (e * tau)
                if lag_idx < 0:  
                    continue
                embedding.append(target_values[lag_idx])
            
            if len(embedding) == E:
                X.append(embedding)
                if i < len(data) - 1 and (target is not None):  # Ensure we have a next value
                    y.append(target_values[i + 1])
        X = np.array(X)
        y = np.array(y)
        
        return X, y


    def total_embeddings(self, data, E, tau, target):
        """
        Create time delay embeddings from time series data.
        
        Parameters:
        data (DataFrame): Input time series data where columns are features
        E (int): Maximum embedding dimension (number of time steps back)
        tau (lis of ints): Time delays between steps (absolute value will be used)
        target (str): Name of the target column in data
        
        Returns:
        results (list): List of embedded vectors, where each vector contains all possible lags
        y (list): List of target values corresponding to each embedding
        """
        First_l = True
        for t in tau:
            X, y = self.create_embeddings(data, E, t, target)
            for col in data.columns:
                if col != target:
                    X_cur, _ = self.create_embeddings(data, E, t, col)
                    X = np.hstack((X, X_cur))
            #Keep overlaps
            if First_l:
                result = X
                First_l = False
            else:
                result = np.hstack((result[len(result)-len(X):], X))
        return result, y
    
    def __call__(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        return self.total_embeddings(data,self.E,self.tau,self.target)
