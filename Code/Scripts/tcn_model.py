import torch
from torch import nn
from torch.nn.utils import weight_norm

# ======================= Trust Gate (per-expert mask) =======================

def _logit(p, eps=1e-6):
    p = torch.clamp(torch.as_tensor(p, dtype=torch.float32), eps, 1 - eps)
    return torch.log(p / (1 - p))

class TrustGate(nn.Module):
    """
    Per-channel gate in (0,1) applied to time series features.
    Input x: (batch, seq_len, n_channels)
    """
    def __init__(self, n_channels: int, init_probs=None, temperature: float = 8.0):
        super().__init__()
        self.n = n_channels
        self.temperature = float(temperature)
        if init_probs is None:
            init_probs = torch.full((n_channels,), 0.5)
        init_logits = _logit(init_probs)
        self.gate_logits = nn.Parameter(init_logits)

    def forward(self, x):
        g = torch.sigmoid(self.gate_logits * self.temperature)  # (n,)
        return x * g.view(1, 1, self.n)

    @torch.no_grad()
    def set_probs(self, probs):
        self.gate_logits.copy_(_logit(probs))

    @torch.no_grad()
    def gate_probs(self):
        return torch.sigmoid(self.gate_logits).cpu()

# ============================ Residual TCN stack ============================

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=self.pad, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=self.pad, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.final_relu = nn.ReLU()

    def forward(self, x):  # x: (B, C_in, T)
        res = self.downsample(x)

        out = self.conv1(x)
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
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            layers.append(TCNBlock(in_ch if i == 0 else channels[i-1],
                                   ch, kernel_size=kernel_size,
                                   dilation=2**i, dropout=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, T, C_in)
        x = x.transpose(1, 2)     # -> (B, C_in, T)
        y = self.net(x)
        return y.transpose(1, 2)  # -> (B, T, C_out)

# =================== Hybrid TAB (TCN → MHA → BiLSTM → Head) =================

class HybridTAB(nn.Module):
    def __init__(
        self,
        input_size: int,
        tcn_channels: list[int] = [64, 64],
        mha_heads: int = 4,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.tcn = TCNEncoder(input_size, tcn_channels, kernel_size=3, dropout=dropout)

        d_model = tcn_channels[-1]
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=mha_heads,
                                         dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(d_model)

        self.bilstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden,
                              num_layers=lstm_layers, batch_first=True,
                              dropout=dropout if lstm_layers > 1 else 0.0,
                              bidirectional=True)
        self.lstm_ln = nn.LayerNorm(lstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x):  # (B, T, C_in)
        tcn_out, _ = self.tcn(x), None                 # (B, T, d_model)
        attn_out, _ = self.mha(tcn_out, tcn_out, tcn_out)
        attn_out = self.attn_ln(attn_out + tcn_out)
        lstm_out, _ = self.bilstm(attn_out)            # (B, T, 2H)
        last = self.lstm_ln(lstm_out[:, -1, :])        # (B, 2H)
        return self.classifier(last)                   # (B, num_classes)

# ============= Wrapper: TrustGate → HybridTAB (for easy use) ================

class HybridTABWithGate(nn.Module):
    def __init__(
        self,
        input_size: int,
        tcn_channels: list[int] = [64, 64],
        mha_heads: int = 4,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        gate_init_probs=None,
        gate_temperature: float = 8.0,
    ):
        super().__init__()
        self.gate = TrustGate(input_size, init_probs=gate_init_probs, temperature=gate_temperature)
        self.core = HybridTAB(input_size=input_size,
                              tcn_channels=tcn_channels,
                              mha_heads=mha_heads,
                              lstm_hidden=lstm_hidden,
                              lstm_layers=lstm_layers,
                              num_classes=num_classes,
                              dropout=dropout)

    def forward(self, x):
        return self.core(self.gate(x))
