import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import random_split

from utils import *
from nn_edm import NNEDMModel
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

    print('ENCODED DATA')
    '''
    PAPER DATA
    paper_data = pd.read_csv(input_file_path)
    paper_data = paper_data.set_index('time (UTC)')
    paper_data['Time'] = paper_data.index.astype(int)
    paper_data = paper_data.drop(columns=['Time'])
    '''
    data = pd.read_csv('lorenz_with_extra_vars.csv')
    data = data.drop(columns=['time','y'])
    tau_lengths = [-1,-2,-3]
    E = 6
    target = 'x'
    X, y = get_data(data, E, tau_lengths, target=target) #returns numpy arrays 
    embd_sz = len(data.columns) * E * len(tau_lengths)

    print('CREATED EMBEDDINGS')

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.float)

    len_data = len(y_tensor)

    train_frac, val_frac, test_frac = 0.5, 0.3, 0.2  
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
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    val_dataset = HABsDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    test_dataset = HABsDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NNEDMModel(embd_sz, hidden_size=1000)


    #config = load_config(args.config_file_path)
    #THIS IS WHERE I LEFT OFF
    ## TRAIN - will save best model weights
    train(model=model,
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader)

    ## INFERENCE
    eval(model=model, device=device,
    val_dataloader=test_dataloader)


if __name__ == '__main__':
    main()