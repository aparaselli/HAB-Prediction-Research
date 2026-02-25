"""
Standalone Coov-LSTM training script.
Uses Conv2d + LSTM architecture on base (non-EDM) features.
Selects best model by validation AUC; reports test AUC at that checkpoint.
"""
import sys, os, random, math
import numpy as np
import pandas as pd
import yaml
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Datasets --------------------

class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X, self.y, self.seq_len = X, y, seq_len
    def __len__(self): return max(0, len(self.X) - self.seq_len)
    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len]

class NoisySequenceDataset(Dataset):
    def __init__(self, X, y, seq_len, noise_pct=0.02, indices=None):
        self.X, self.y, self.seq_len, self.noise_pct = X, y, seq_len, noise_pct
        self.range = X.max(dim=0).values - X.min(dim=0).values
        max_idx = len(X) - seq_len
        self.indices = list(range(max_idx)) if indices is None else [i for i in indices if 0 <= i < max_idx]
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        base = self.indices[idx]
        window = self.X[base:base+self.seq_len]
        noise = torch.randn_like(window) * (self.range * self.noise_pct)
        return window + noise, self.y[base+self.seq_len]

class _IndexedSlice(Dataset):
    def __init__(self, subs, picks):
        self.subs, self.picks = subs, picks
    def __len__(self): return len(self.picks)
    def __getitem__(self, i):
        s_i, k = self.picks[i]
        return self.subs[s_i][k]

# -------------------- Model --------------------

class Coov_LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))
        conv_out_size = (input_size // 2) * (hidden_size // 2)
        self.lstm = nn.LSTM(conv_out_size, hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, feat = x.size()
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(batch_size, seq_len, -1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# -------------------- Helpers --------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        probs = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
        ys.extend(yb.numpy()); ps.extend(probs.cpu().numpy())
    if len(np.unique(ys)) < 2:
        return 0.5
    return float(roc_auc_score(ys, ps))

# -------------------- Main --------------------

def train_coov(cfg_path=None, seed=2028):
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "configs", "lstm_carter_script.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Config paths are relative to Code/Scripts/ (parent of this script's directory)
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data_path = os.path.normpath(os.path.join(scripts_dir, cfg["data_path"]))

    target_col = cfg["target"]
    train_sz   = int(cfg["train_sz"])
    seq_len    = int(cfg["seq_len"])
    batch_sz   = int(cfg["batch_sz"])
    hidden     = int(cfg["lstm_hd_sz"])
    layers     = int(cfg["lstm_num_lyrs"])
    dropout    = float(cfg["dropout"])
    lr         = float(cfg["learning_rate"])
    bloom_pen  = float(cfg["bloom_pen"])
    noisy_prop = float(cfg.get("noisy_prop", 0.5))
    noise_pct  = float(cfg.get("noise_pct", 0.03))
    val_len    = int(cfg.get("val_len", 100))
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    print(f"Device: {device}, seed: {seed}")

    # ---- Data (base features, no EDM) ----
    df = pd.read_csv(data_path)
    num_df = df.select_dtypes(include=[np.number]).copy()
    X_df = num_df.drop(columns=[target_col])
    y_sr = num_df[target_col]
    X_df, y_sr = X_df.ffill().bfill(), y_sr.ffill().bfill()
    X_np = X_df.to_numpy(dtype=np.float32)
    y_np = y_sr.to_numpy(dtype=np.float32)

    mask = ~y_sr.isna().to_numpy()
    mu = np.nanmean(X_np[:train_sz], axis=0)
    sigma = np.nanstd(X_np[:train_sz], axis=0)
    sigma[sigma == 0] = 1.0
    X_np = (X_np - mu) / sigma

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np).float()
    input_dim = X.shape[1]

    # ---- Labels (95th percentile on train-only portion) ----
    train_sz_adj = train_sz - val_len
    t2 = float(np.percentile(y[:train_sz_adj].numpy(), 95))
    print(f"Bloom threshold (95th pct): {t2:.4f}")
    bins = torch.tensor([t2], dtype=torch.float32)
    y = torch.bucketize(y, bins)

    # ---- Train segments (gap-aware) ----
    train_mask = mask[:train_sz_adj]
    good = np.where(train_mask)[0]
    breaks = np.where(np.diff(good) != 1)[0] + 1
    segments = np.split(good, breaks)

    vanilla_subs, noisy_subs = [], []
    for seg in segments:
        if len(seg) <= seq_len:
            continue
        Xi, yi = X[seg], y[seg]
        vanilla_subs.append(SequenceDataset(Xi, yi, seq_len))
        base_idxs = list(range(len(Xi) - seq_len))
        minority = [i for i in base_idxs if yi[i + seq_len] == 1]
        majority = [i for i in base_idxs if yi[i + seq_len] == 0]
        num_maj = int(len(majority) * noisy_prop)
        maj_sampled = random.sample(majority, k=max(0, num_maj))
        noisy_subs.append(NoisySequenceDataset(Xi, yi, seq_len=seq_len,
                          noise_pct=noise_pct, indices=minority + maj_sampled))

    aug_ds = ConcatDataset(vanilla_subs + noisy_subs)
    labels = torch.tensor([int(aug_ds[i][1]) for i in range(len(aug_ds))])
    counts = torch.bincount(labels, minlength=2).float()
    wts = (1.0 / counts)[labels].double()
    sampler = WeightedRandomSampler(wts, len(aug_ds), replacement=True)
    train_loader = DataLoader(aug_ds, batch_size=batch_sz, sampler=sampler)

    # ---- Val/Test interleave ----
    post_start = train_sz_adj + seq_len
    mask_seg = mask[post_start:len(X)]
    good_post = np.where(mask_seg)[0]
    cuts = np.where(np.diff(good_post) != 1)[0] + 1
    runs_post = [r for r in np.split(good_post, cuts) if len(r) > seq_len]

    all_subs = [SequenceDataset(X[r + post_start], y[r + post_start], seq_len)
                for r in runs_post if len(r) > seq_len]
    all_picks = [(s_i, k) for s_i, sub in enumerate(all_subs) for k in range(len(sub))]
    val_picks = [all_picks[i] for i in range(0, len(all_picks), 2)]
    test_picks = [all_picks[i] for i in range(1, len(all_picks), 2)]

    val_ds = _IndexedSlice(all_subs, val_picks)
    test_ds = _IndexedSlice(all_subs, test_picks)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

    print(f"Input dim: {input_dim}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # ---- Model ----
    model = Coov_LSTMClassifier(
        input_size=input_dim, hidden_size=hidden,
        num_layers=layers, num_classes=2, dropout=dropout
    ).to(device)

    with torch.no_grad():
        p = float((y[:train_sz_adj] == 1).float().mean().clamp(1e-6, 1 - 1e-6))
        b = math.log(p / (1 - p))
        model.fc.bias[1].fill_(b)
        model.fc.bias[0].fill_(-b)

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, bloom_pen], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=60, min_lr=1e-7)

    # ---- Training (val AUC selection) ----
    best_val, best_test, best_state, wait = 0.0, 0.0, None, 0
    patience, epochs = 100, 1000

    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = sum(loss_fn(model(Xb.to(device)), yb.to(device)).item()
                     for Xb, yb in val_loader)
            sched.step(vl / max(1, len(val_loader)))

        val_auc = eval_auc(model, val_loader, device)
        if val_auc > best_val:
            best_val = val_auc
            best_test = eval_auc(model, test_loader, device)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if ep % 25 == 0:
            test_auc = eval_auc(model, test_loader, device)
            print(f"Epoch {ep:4d} | val_auc={val_auc:.4f} | test_auc={test_auc:.4f} "
                  f"| best_val={best_val:.4f} | test@bestVal={best_test:.4f}")

        if wait >= patience:
            print(f"Early stopping at epoch {ep}")
            break

    if best_state:
        model.load_state_dict(best_state)

    print(f"\nFinal: best Val AUC={best_val:.4f} | Test AUC @ best Val={best_test:.4f}")
    return model, val_loader, test_loader, best_val, best_test


def plot_roc_and_save(model, test_loader, device, out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            probs = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
            ys.extend(yb.numpy()); ps.extend(probs.cpu().numpy())

    y_true, y_score = np.array(ys), np.array(ps)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], '--', lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Coov-LSTM — Test ROC"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coov_test_roc.png"), dpi=200)
    plt.close()
    print(f"Saved ROC plot to {out_dir}/coov_test_roc.png")

    # Save FPR/TPR CSV
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    csv_path = os.path.join(out_dir, "best_test_roc.csv")
    roc_df.to_csv(csv_path, index=False)
    print(f"Saved ROC CSV to {csv_path}")

    return fpr, tpr, auc_val


if __name__ == "__main__":
    N_SEEDS = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.dirname(os.path.abspath(__file__))

    best_overall_val = 0.0
    best_overall_test = 0.0
    best_seed = 0
    best_model_state = None

    for i, seed in enumerate(range(2000, 2000 + N_SEEDS)):
        print(f"\n[{i+1}/{N_SEEDS}] seed={seed} ...", end=" ", flush=True)
        model, val_loader, test_loader, bv, bt = train_coov(seed=seed)
        print(f"val={bv:.4f} test={bt:.4f}", end="")

        if bv > best_overall_val:
            best_overall_val = bv
            best_overall_test = bt
            best_seed = seed
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  *** NEW BEST VAL ***", end="")
        print()

    print(f"\n{'='*50}")
    print(f"BEST: seed={best_seed} val={best_overall_val:.4f} test={best_overall_test:.4f}")
    print(f"{'='*50}")

    # Load best model and save everything
    model.load_state_dict(best_model_state)
    fpr, tpr, auc_val = plot_roc_and_save(model, test_loader, device)
    torch.save(best_model_state, os.path.join(out_dir, "best_model.pth"))
    print(f"Saved model weights to {out_dir}/best_model.pth")

    with open(os.path.join(out_dir, "best_config.yaml"), "w") as f:
        yaml.dump({
            "model": "Coov_LSTMClassifier",
            "seed": best_seed,
            "val_auc": round(best_overall_val, 4),
            "test_auc": round(best_overall_test, 4),
        }, f)
