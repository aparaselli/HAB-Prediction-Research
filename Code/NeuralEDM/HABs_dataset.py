import torch
from torch.utils.data import Dataset
from utils import *

class HABsDataset(Dataset):

    def __init__(self, X, y): #X, E, tau, target=None
        #self.embd = EDM_embedding(E, tau, target=None)
        #self.X, self.y  = self.embd(X)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]