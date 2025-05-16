import torch
from torch import nn
from torch.nn.utils import weight_norm

# ——— Residual TCN Block ————————————————————————————————
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=self.pad, dilation=dilation)
        )
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size,
                      padding=self.pad, dilation=dilation)
        )
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # 1×1 downsample if channels change
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, in_ch, seq_len)
        res = self.downsample(x)

        out = self.conv1(x)
        # chop off the extra padding at the end
        out = out[:, :, :x.size(2)]
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu2(out)
        out = self.drop2(out)

        return self.final_relu(out + res)



class TCNEncoder(nn.Module):
    def __init__(self, in_ch, channels, kernel_size=3, dropout=0.2):
        """
        channels: list of hidden dims, e.g. [64,64,64]
        """
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            layers.append(
                TCNBlock(
                    in_ch if i == 0 else channels[i - 1],
                    ch,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len, in_ch)
        x = x.transpose(1, 2)           # → (batch, in_ch, seq_len)
        y = self.net(x)
        return y.transpose(1, 2)        # → (batch, seq_len, channels[-1])


# ——— Hybrid TAB with Multi-Head Attention —————————————————————
class HybridTAB(nn.Module):
    def __init__(
        self,
        input_size: int,
        tcn_channels: list[int] = [64, 64],
        mha_heads: int = 4,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        # 1) TCN encoder
        self.tcn = TCNEncoder(input_size, tcn_channels, kernel_size=3, dropout=dropout)

        # 2) Multi-head self-attention
        d_model = tcn_channels[-1]
        self.mha = nn.MultiheadAttention(embed_dim=d_model,
                                         num_heads=mha_heads,
                                         dropout=dropout,
                                         batch_first=True)
        self.attn_ln = nn.LayerNorm(d_model)

        # 3) BiLSTM
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True
        )
        self.lstm_ln = nn.LayerNorm(lstm_hidden*2)

        # 4) Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden*2, lstm_hidden),
            nn.ReLU(),
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # — TCN →
        tcn_out = self.tcn(x)                      # (batch, seq_len, d_model)

        # — Multi-Head Attention →
        # (attn requires (batch, seq, embed))
        attn_out, _ = self.mha(tcn_out, tcn_out, tcn_out)
        # residual + layer-norm
        attn_out = self.attn_ln(attn_out + tcn_out)  # (batch, seq_len, d_model)

        # — BiLSTM →
        lstm_out, _ = self.bilstm(attn_out)        # (batch, seq_len, 2*hidden)
        # take last timestep
        last = lstm_out[:, -1, :]                 # (batch, 2*hidden)
        last = self.lstm_ln(last)

        # — Classifier →
        logits = self.classifier(last)            # (batch, num_classes)
        return logits
