import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import random_split
from config import *

from utils import *
from nn_edm import NNEDMModel
from gru_nn import GRUEDMModel
from HABs_dataset import HABsDataset
from train import train, eval

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config_file_path',
        type=str
    )

    args = parser.parse_args()

    input_file_path = '../Data/cleaned_data.csv'
    target = 'Avg_Chloro'
    config = load_config(args.config_file_path)

    print('ENCODED DATA')
    
    #PAPER DATA
    data = pd.read_csv(input_file_path)
    data = data.set_index('time (UTC)')
    data['Time'] = data.index.astype(int)
    data = data.drop(columns=['Time'])

    tau_lengths = [-1,-2,-3]
    E = 6
    X, y = get_data(data, E, tau_lengths, target=target) #returns numpy arrays 
    print(len(X))
    embd_sz = len(data.columns) * E * len(tau_lengths)

    print('CREATED EMBEDDINGS')

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X[:633], dtype=torch.float) #Train with 532 data points
    y_tensor = torch.tensor(y[:633], dtype=torch.float)

    len_data = len(y_tensor)

    train_frac, val_frac, test_frac = 0.8, 0.2, 0.0
    train_size = int(train_frac * len_data)
    val_size = int(val_frac * len_data)
    test_size = len_data - train_size - val_size  


    #assert set(train_indices).isdisjoint(set(val_indices)) and set(train_indices).isdisjoint(set(test_indices)) and set(val_indices).isdisjoint(set(test_indices))
    print('PERFORMED TRAIN/VAL/TEST SPLIT')
    torch.manual_seed(0)
    indices = torch.randperm(len_data)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    assert set(train_indices).isdisjoint(set(val_indices)) and set(train_indices).isdisjoint(set(test_indices)) and set(val_indices).isdisjoint(set(test_indices))
    print('PERFORMED TRAIN/VAL/TEST SPLIT')

    # Index tensors to get non-overlapping splits
    X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
    X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]
    X_test, y_test = X_tensor[test_indices], y_tensor[test_indices]


    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


    train_dataset = HABsDataset(X_train, y_train)
    #print(train_dataset.__getitem__(0))
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = HABsDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    test_dataset = HABsDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config['model'] == 'Base':
        model = NNEDMModel(embd_sz, hidden_size=config['hidden_size'])
    elif config['model'] == 'GRU':
        model = GRUEDMModel(embd_sz,hidden_size=config['hidden_size'],num_layers=config['num_layers'])
    else:
        print('INVALID MODEL SELECTED')
        return


    #config = load_config(args.config_file_path)
    #THIS IS WHERE I LEFT OFF
    ## TRAIN - will save best model weights
    train(model=model,
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,epochs=config['epochs'],patience=config['patience'],save_m_path=config['saved_model_path'])

    ## INFERENCE
    eval(model=model, device=device,
    val_dataloader=val_dataloader)


if __name__ == '__main__':
    main()