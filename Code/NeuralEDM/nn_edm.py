import torch
import torch.nn as nn
from utils import *

# Define RNN Model
class NNEDMModel(nn.Module):
    def __init__(self,embd_sz, hidden_size=100):
        super(NNEDMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embd_sz = embd_sz
        self.relu = nn.ReLU()
        self.fw = nn.Linear(self.embd_sz,self.hidden_size)
        self.output = nn.Linear(self.hidden_size,1)
    
    def forward(self, x):
        x = self.fw(x)
        x = self.relu(x)
        return self.output(x).squeeze()

        
        ## TODO