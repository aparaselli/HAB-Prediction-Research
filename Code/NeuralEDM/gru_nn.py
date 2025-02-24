import torch
import torch.nn as nn

class GRUEDMModel(nn.Module):
    def __init__(self, embd_sz, hidden_size=100, num_layers=2, dropout=0.2):
        super(GRUEDMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embd_sz = embd_sz
        
        self.gru = nn.GRU(input_size=self.embd_sz, 
                          hidden_size=self.hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True, 
                          dropout=dropout)
        
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        x = attn_out

        
        return self.fc(x).squeeze()
