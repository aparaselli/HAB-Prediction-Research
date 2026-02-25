"""
Load the 0.78+ EDM-LSTM model and generate:
1. ROC curve plot (test set)
2. CSV with val/test results
"""

import sys, os, math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from forecast import load_yaml, clean_data, process_parameters, create_model
from tcn_model import HybridTABWithGate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========================= Classes =========================

class EDM_bank:
    def __init__(self, lib_sz, params, target, samp, n):
        self.lib_sz, self.params, self.target = int(lib_sz), params, target
        self.samp, self.n = int(samp), int(n)
    def forecast(self, x):
        lib = '1 ' + str(self.lib_sz)
        pred = str(self.lib_sz + 1) + ' ' + str(x.shape[0])
        pdf = create_model(x, self.params, self.target, self.samp, lib, pred, ensemble_sz=self.n)
        model_preds = np.stack([np.asarray(pdf["pred"].iloc[i][1:-1]) for i in range(self.n)])
        return torch.as_tensor(model_preds.T, dtype=torch.float32)
    def __call__(self, x): return self.forecast(x)

class Embedd_EDM:
    def __init__(self, EDM, data):
        self.targets = torch.tensor(data[EDM.target].iloc[EDM.lib_sz+1:].to_numpy(), dtype=torch.float32)
        self.embedded_inputs = EDM(data)
    def get_data(self): return self.embedded_inputs, self.targets

class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X, self.y, self.seq_len = X, y, seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len]

class _IndexedSlice(Dataset):
    def __init__(self, subs, picks):
        self.subs, self.picks = subs, picks
    def __len__(self): return len(self.picks)
    def __getitem__(self, i):
        s_i, k = self.picks[i]
        return self.subs[s_i][k]

# ========================= Main =========================

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "lstm_carter_script.yaml")
    config = load_yaml(config_path)
    # Config paths are relative to Code/Scripts/ (parent of this script's directory)
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    config['data_path'] = os.path.normpath(os.path.join(scripts_dir, config['data_path']))
    config['parameters_path'] = os.path.normpath(os.path.join(scripts_dir, config['parameters_path']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data pipeline (same as training)
    data = clean_data(config['data_path'])
    parameters = process_parameters(config['parameters_path'])
    data_bank_sz = config['data_bank_sz']
    train_sz = int(config['train_sz'])
    val_len = int(config.get("val_len", 100))
    seq_len = config['seq_len']

    print("Creating EDM embeddings...")
    data_bank = EDM_bank(data_bank_sz, parameters, config['target'], config['samp'], config['n'])
    embedder = Embedd_EDM(data_bank, data)
    X, y = embedder.get_data()

    weeks = pd.to_datetime(data['DATE']).dt.isocalendar().week.to_numpy()
    phi = 2 * np.pi * weeks / 52.0
    season = torch.from_numpy(np.stack([np.sin(phi), np.cos(phi)], axis=1)).float()
    season = season[data_bank_sz + 1:]
    X = torch.cat([X, season], dim=1)

    mu = X[:train_sz].mean(dim=0)
    sigma = X[:train_sz].std(dim=0).clamp_min(1e-6)
    X = (X - mu) / sigma
    input_dim = X.shape[1]

    train_sz_adj = train_sz - val_len
    t2 = float(np.percentile(y[:train_sz_adj].numpy(), 95))
    bins = torch.tensor([t2], dtype=torch.float32)
    y = torch.bucketize(y, bins)

    orig = data[config['target']]
    mask = ~orig.iloc[data_bank_sz + 1:].isna().to_numpy()

    # Val/Test interleave (same split as training)
    post_start = train_sz_adj + seq_len
    mask_seg = mask[post_start:len(X)]
    good_post = np.where(mask_seg)[0]
    cuts = np.where(np.diff(good_post) != 1)[0] + 1
    runs_post = [r for r in np.split(good_post, cuts) if len(r) > seq_len]

    all_subs = [SequenceDataset(X[r + post_start], y[r + post_start], seq_len) for r in runs_post if len(r) > seq_len]
    all_picks = [(s_i, k) for s_i, sub in enumerate(all_subs) for k in range(len(sub))]

    val_picks = [all_picks[i] for i in range(0, len(all_picks), 2)]
    test_picks = [all_picks[i] for i in range(1, len(all_picks), 2)]

    val_ds = _IndexedSlice(all_subs, val_picks)
    test_ds = _IndexedSlice(all_subs, test_picks)

    batch_sz = 16
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

    print(f"Input dim: {input_dim}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Load model (hidden=64, tcn=64, mha=4, layers=2, dropout=0.5)
    model = HybridTABWithGate(
        input_size=input_dim,
        tcn_channels=[64, 64],
        mha_heads=4,
        lstm_hidden=64,
        lstm_layers=2,
        num_classes=2,
        dropout=0.5,
        gate_temperature=6.0,
    ).to(device)

    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded weights from {weights_path}")

    # Evaluate on val and test
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Results")

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        y_true, y_score = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                probs = torch.softmax(model(Xb.to(device)), dim=1)[:, 1]
                y_true.extend(yb.numpy())
                y_score.extend(probs.cpu().numpy())

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        auc_val = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        print(f"{split_name.upper()} AUC: {auc_val:.4f}")

        # ROC plot
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'EDM-LSTM ROC Curve ({split_name.capitalize()} Set)', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.tight_layout()
        roc_path = os.path.join(results_dir, f"ROC_{split_name}.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"  ROC plot saved to {roc_path}")

    # CSV with ROC curve data (FPR, TPR, threshold) for test set
    yt, ys = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            probs = torch.softmax(model(Xb.to(device)), dim=1)[:, 1]
            yt.extend(yb.numpy()); ys.extend(probs.cpu().numpy())
    fpr_t, tpr_t, thr_t = roc_curve(np.array(yt), np.array(ys))
    test_auc = roc_auc_score(np.array(yt), np.array(ys))

    # Save FPR/TPR CSV to auc_plots (matching existing format)
    auc_plots_dir = os.path.join(results_dir, "auc_plots")
    os.makedirs(auc_plots_dir, exist_ok=True)
    csv_path = os.path.join(auc_plots_dir, "EDM_LSTM_BEST_TEST_FPR_TPR.csv")
    roc_df = pd.DataFrame({"FPR": fpr_t, "TPR": tpr_t})
    roc_df.to_csv(csv_path, index=False)
    print(f"Test ROC data CSV saved to {csv_path} ({len(roc_df)} points, AUC={test_auc:.4f})")

    # Also save the model to the standard save path
    std_path = os.path.join(results_dir, "Lingulodinium_LSTM.pth")
    torch.save(model.state_dict(), std_path)
    print(f"Model also saved to {std_path}")

    print("\nDone!")
