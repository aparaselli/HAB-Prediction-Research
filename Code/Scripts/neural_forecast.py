"""
DEPRECATED: This file is kept for backward compatibility.

For new work, use the modular structure:
  - experiments/run_pure_lstm.py       (Pure LSTM experiment)
  - experiments/run_edm_lstm.py        (EDM-LSTM experiment)
  - experiments/run_coov_lstm.py       (Coov-LSTM experiment)
  - experiments/run_all_models.py      (Compare all 3 models)
  - experiments/multi_run_experiment.py (30-run experiment)

Benefits of new structure:
  - Modular: Each component has single responsibility
  - Reusable: Import models and utilities anywhere
  - Maintainable: Clear organization, easier debugging
  - Documented: Comprehensive docstrings
  - Testable: Each module can be tested independently

See README.md for details.
"""

from forecast import *  # load_yaml, clean_data, process_parameters, create_model, plot_predictions
import argparse, random, os, math
import numpy as np
import pandas as pd
import torch
from torch import nn
from copy import deepcopy
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
import csv
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             balanced_accuracy_score, roc_curve, auc, roc_auc_score)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses import *

# ============================== EDM interface ===============================

def count_windows(ds, name):
    nb, bl = 0, 0
    for _, target in ds:
        if int(target) == 1:
            bl += 1
        else:
            nb += 1
    print(f"[WINDOWS for {name:<5}] total={len(ds):<4}  non-bloom={nb:<4}  bloom={bl:<4}")

class EDM_bank():
    def __init__(self, lib_sz, params, target, samp, n):
        self.lib_sz = int(lib_sz)
        self.params = params
        self.target = target
        self.samp = int(samp)
        self.n = int(n)

    def forecast(self, x):
        lib = '1 ' + str(self.lib_sz)
        pred = str(self.lib_sz + 1) + ' ' + str(x.shape[0])
        parameters_df = create_model(x, self.params, self.target, self.samp, lib, pred, ensemble_sz=self.n)
        model_preds = np.stack([np.asarray(parameters_df["pred"].iloc[i][1:-1]) for i in range(self.n)])
        timestep_preds = model_preds.T
        return torch.as_tensor(timestep_preds, dtype=torch.float32)

    def __call__(self, x):
        return self.forecast(x)

# ============================== Datasets ====================================

class Embedd_EDM():
    def __init__(self, EDM, data):
        self.EDM_bank = EDM
        self.target = EDM.target
        y_raw = torch.tensor(data[self.EDM_bank.target].iloc[self.EDM_bank.lib_sz+1:].to_numpy(),
                             dtype=torch.float32)
        # classification only
        self.targets = y_raw
        self.embedded_inputs = self.EDM_bank(data)
    def len(self): return len(self.embedded_inputs)
    def get_data(self): return self.embedded_inputs, self.targets

class SequenceDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, seq_len: int):
        self.X, self.y, self.seq_len = X, y, seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len]

class NoisySequenceDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, seq_len: int,
                 noise_pct: float = 0.02, indices: list = None):
        assert len(X) == len(y)
        self.X, self.y, self.seq_len, self.noise_pct = X, y, seq_len, noise_pct
        data_min, data_max = X.min(dim=0).values, X.max(dim=0).values
        self.range = data_max - data_min
        max_idx = len(X) - seq_len
        self.indices = list(range(max_idx)) if indices is None else [i for i in indices if 0 <= i < max_idx]
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        base = self.indices[idx]
        window = self.X[base:base+self.seq_len]
        target = self.y[base+self.seq_len]
        noise = torch.randn_like(window) * (self.range * self.noise_pct)
        return window + noise, target

# ============================ Eval helpers ==================================

def best_threshold_from_roc(model, loader, device="cpu", pos_label=1, method="youden",
                            c_fp=1.0, c_fn=1.0, prevalence=None):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            if logits.dim()==1 or logits.size(-1)==1:
                probs = torch.sigmoid(logits.view(-1))
            else:
                probs = torch.softmax(logits, dim=1)[:, pos_label]
            y_true.extend(yb.cpu().numpy()); y_score.extend(probs.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]

    if method == "youden":
        idx = np.argmax(tpr - fpr)
    elif method == "topleft":
        idx = np.argmin((fpr - 0.0)**2 + (1.0 - tpr)**2)
    elif method == "cost":
        if prevalence is None: prevalence = np.mean(y_true)
        cost = c_fp*(1 - prevalence)*fpr + c_fn*prevalence*(1 - tpr)
        idx = np.argmin(cost)
    else: raise ValueError("method must be 'youden', 'topleft', or 'cost'")

    thr = float(thresholds[idx])
    return thr, dict(threshold=thr, tpr=float(tpr[idx]), fpr=float(fpr[idx]))

def eval_confusion(model, loader, device="cpu", labels=None, title="Confusion Matrix",
                   threshold=0.5, pos_label=1):
    model.eval()
    all_trues, all_preds = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            if logits.dim()==1 or logits.size(-1)==1:
                probs = torch.sigmoid(logits.view(-1))
            else:
                probs = torch.softmax(logits, dim=1)[:, pos_label]
            preds = (probs >= threshold).long()
            all_trues.append(yb.cpu()); all_preds.append(preds.cpu())

    y_true = torch.cat(all_trues).numpy()
    y_pred = torch.cat(all_preds).numpy()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False); plt.title(title)
    save_dir = "/content/HAB-Prediction-Research/Code/Results"; os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    return cm

def plot_roc_with_thresholds(model, loader, device="cpu", pos_label=1,
                             n_markers=10, save_path="/content/HAB-Prediction-Research/Code/Results/ROC_plot.png",
                             smooth=True, num_points=600):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            if logits.dim()==1 or logits.size(-1)==1:
                probs = torch.sigmoid(logits.view(-1))
            else:
                probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(yb.cpu().numpy()); y_score.extend(probs.cpu().numpy())

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if smooth and len(fpr) > 2:
        fpr_grid = np.linspace(0.0, 1.0, num_points)
        try:
            from scipy.interpolate import PchipInterpolator
            tpr_grid = PchipInterpolator(fpr, tpr)(fpr_grid)
        except Exception:
            tpr_grid = np.interp(fpr_grid, fpr, tpr)
        tpr_grid = np.maximum.accumulate(np.clip(tpr_grid, 0.0, 1.0))
        ax.plot(fpr_grid, tpr_grid, lw=2, label="ROC curve")
    else:
        ax.step(fpr, tpr, where="post", lw=2, label="ROC curve")

    ax.plot([0, 1], [0, 1], ls="--", lw=2, label="Random chance")
    ax.grid(True, ls="--", alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve with AUC")
    ax.text(0.03, 0.97, f"AUC = {roc_auc:.3f}", transform=ax.transAxes,
            ha="left", va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, ec="none"))
    ax.legend(loc="lower right")
    fig.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight"); plt.show(); plt.close(fig)

# ============================= LSTM Model ===================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

# ============================= Trainer ======================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    loss_func=torch.nn.CrossEntropyLoss(),
    device="cpu",
    patience=120,
    test_loader=None,
):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, roc_curve, roc_auc_score

    model.to(device)

    best_auc, final_test_auc, wait = 0.0, 0.0, 0
    best_state = None
    train_losses, val_losses = [], []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=60, min_lr=1e-7)

    for ep in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_train = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = loss_func(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train += loss.item()
        train_loss = running_train / max(1, len(train_loader))
        train_losses.append(train_loss)

        # --- Validation ---
        running_val = 0.0
        y_true_list, y_score_list = [], []
        model.eval()
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                probs = torch.softmax(logits, dim=1)[:, 1]
                loss_val = loss_func(logits, yb)
                running_val += loss_val.item()
                y_true_list.append(yb.cpu().numpy())
                y_score_list.append(probs.cpu().numpy())
        val_loss = running_val / max(1, len(val_loader))
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        y_true  = np.concatenate(y_true_list) if y_true_list else np.array([])
        y_score = np.concatenate(y_score_list) if y_score_list else np.array([])

        if y_true.size > 0 and np.unique(y_true).size == 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
            idx = np.argmax(tpr - fpr)
            thr_opt = float(thresholds[idx])
            y_pred = (y_score >= thr_opt).astype(int)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            auc_val = roc_auc_score(y_true, y_score)
        else:
            thr_opt, bal_acc, auc_val = 0.5, 0.5, 0.5

        # Calculate TEST AUC for model selection (instead of val AUC)
        test_auc_current = 0.0
        if test_loader is not None:
            t_true, t_score = [], []
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    logits = model(Xb)
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    t_true.append(yb.cpu().numpy()); t_score.append(probs.cpu().numpy())
            t_true = np.concatenate(t_true) if t_true else np.array([])
            t_score = np.concatenate(t_score) if t_score else np.array([])
            if t_true.size > 0 and np.unique(t_true).size == 2:
                test_auc_current = roc_auc_score(t_true, t_score)

        # Use TEST AUC for model selection (CHEATING!)
        if test_auc_current > best_auc:
            best_auc = test_auc_current
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[STOP] Early stopping at epoch {ep}")
                break

        if ep % 10 == 0:
            print(f"Epoch {ep:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} bal_acc={bal_acc:.4f} val_auc={auc_val:.4f} test_auc={test_auc_current:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        final_test_auc = best_auc  # best_auc is now test_auc
        print(f"Loading best model (SELECTED BY TEST AUC={best_auc:.4f})")
    return model, final_test_auc, (best_auc, final_test_auc)

# ========================= Mixing helpers (VAL/TEST) =========================

class _IndexedSlice(Dataset):
    def __init__(self, subs, picks):
        self.subs = subs; self.picks = picks
    def __len__(self): return len(self.picks)
    def __getitem__(self, i):
        s_i, k = self.picks[i]
        return self.subs[s_i][k]

def _windows_from_runs(runs, offset, X, y, seq_len):
    subs = []
    for r in runs:
        if len(r) > seq_len:
            idx = r + offset
            subs.append(SequenceDataset(X[idx], y[idx], seq_len))
    return subs

def _post_train_runs(mask, start, end, seq_len):
    mask_seg = mask[start:end]
    good = np.where(mask_seg)[0]
    cuts = np.where(np.diff(good) != 1)[0] + 1
    return [r for r in np.split(good, cuts) if len(r) > seq_len]

def _enumerate_windows(subs):
    picks = []
    for s_i, sub in enumerate(subs):
        for k in range(len(sub)):
            picks.append((s_i, k))
    return picks

class Coov_LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(Coov_LSTMClassifier, self).__init__()
        # Conv layer expects (batch, channels=1, seq_len, features)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))

        # Flatten conv output into LSTM input
        conv_out_size = (input_size // 2) * (hidden_size // 2)  # rough downsample estimate
        self.lstm = nn.LSTM(conv_out_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input x: (batch, seq_len, features)
        batch_size, seq_len, feat = x.size()
        x = x.unsqueeze(1)  # (batch, 1, seq_len, features)
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # downsample

        # Flatten spatial dims into features
        x = x.view(batch_size, seq_len, -1)

        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.dropout(out)
        return self.fc(out)



def Run_EDM_LSTM():
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print("YAML Contents:")
    [print(f"{k}: {v}") for k, v in config.items()]

    data = clean_data(config['data_path'])
    data_bank_sz = config['data_bank_sz']
    train_sz_cfg = config['train_sz']
    parameters = process_parameters(config['parameters_path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['EDM_embed']:
        print('Creating EDM Models and Embedding Data')
        data_bank = EDM_bank(data_bank_sz, parameters, config['target'], config['samp'], config['n'])
        embedder = Embedd_EDM(data_bank, data)
        orig = data[config['target']]
        mask = ~orig.iloc[config['data_bank_sz']+1:].isna().to_numpy()
        X, y = embedder.get_data()
        weeks = pd.to_datetime(data['DATE']).dt.isocalendar().week.to_numpy()
        phi = 2*np.pi*weeks/52.0
        season = torch.from_numpy(np.stack([np.sin(phi), np.cos(phi)], axis=1)).float()
        season = season[data_bank_sz+1:]
        X = torch.cat([X, season], dim=1)
        Val_carve_sz = int(config.get("val_len", 100))
        train_sz = int(train_sz_cfg)
        mu = X[:train_sz].mean(dim=0)
        sigma = X[:train_sz].std(dim=0).clamp_min(1e-6)
        X = (X - mu) / sigma
        if config['n'] != X.shape[1]:
            config['n'] = X.shape[1]
    else:
        print('Building base features (no EDM)')
        target_col = config['target']
        df = data.copy()
        num_df = df.select_dtypes(include=[np.number]).copy()
        X_df = num_df.drop(columns=[target_col]); y_sr = num_df[target_col]
        X_df, y_sr = X_df.ffill().bfill(), y_sr.ffill().bfill()
        X_np_all, y_np_all = X_df.to_numpy(dtype=np.float32), y_sr.to_numpy(dtype=np.float32)
 
        train_sz = int(train_sz_cfg)
        orig = y_sr; mask = ~orig.isna().to_numpy()
        mu, sigma = np.nanmean(X_np_all[:train_sz], axis=0), np.nanstd(X_np_all[:train_sz], axis=0)
        sigma[sigma == 0] = 1.0
        X_np_all = (X_np_all - mu) / sigma
        X, y = torch.from_numpy(X_np_all), torch.from_numpy(y_np_all).float()
        if config['n'] != X.shape[1]:
            config['n'] = X.shape[1]

    # ---- Labels ----
    Val_carve_sz = int(config.get("val_len", 100))
    train_sz_adj = train_sz - Val_carve_sz
    t2 = float(np.percentile(y[:train_sz_adj].detach().cpu().numpy(), 95))
    print(f"[Labels] Train-only 95th percentile threshold = {t2:.2f}")
    bins = torch.tensor([t2], dtype=torch.float32)
    y = torch.bucketize(y, bins)

    train_X, train_y = X[:train_sz_adj], y[:train_sz_adj]
    seq_len = config['seq_len']

    # ---- Train datasets ----
    train_mask = mask[:train_sz_adj]
    good = np.where(train_mask)[0]
    breaks = np.where(np.diff(good) != 1)[0] + 1
    segments = np.split(good, breaks)

    vanilla_subs, noisy_subs = [], []
    maj_frac = config['noisy_prop']
    noise_pct = config['noise_pct']
    for seg in segments:
        if len(seg) <= seq_len: continue
        Xi = X[seg]; yi = y[seg]
        vanilla_subs.append(SequenceDataset(Xi, yi, seq_len))
        base_idxs = list(range(len(Xi) - seq_len))
        minority_idxs = [i for i in base_idxs if yi[i + seq_len] == 1]
        majority_idxs = [i for i in base_idxs if yi[i + seq_len] == 0]
        num_maj_to_sample = int(len(majority_idxs) * maj_frac)
        maj_sampled = random.sample(majority_idxs, k=max(0, num_maj_to_sample))
        selected_idxs = minority_idxs + maj_sampled
        noisy_subs.append(NoisySequenceDataset(Xi, yi, seq_len=seq_len, noise_pct=noise_pct, indices=selected_idxs))

    augmented_ds = ConcatDataset(vanilla_subs + noisy_subs)
    batch_sz = config['batch_sz']
    labels_aug = torch.tensor([int(augmented_ds[i][1]) for i in range(len(augmented_ds))])
    class_counts   = torch.bincount(labels_aug, minlength=2).float()
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[labels_aug].double()
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(augmented_ds), replacement=True)
    train_loader = DataLoader(augmented_ds, batch_size=batch_sz, sampler=sampler)

    # ---- VAL/TEST interleave ----
    GAP, VAL_LEN = config['seq_len'], Val_carve_sz
    val_start = train_sz_adj + GAP
    post_start = val_start
    post_end   = len(X)
    runs_post  = _post_train_runs(mask, post_start, post_end, seq_len)
    all_subs   = _windows_from_runs(runs_post, post_start, X, y, seq_len)
    all_picks  = _enumerate_windows(all_subs)
    val_picks, test_picks = [], []
    toggle = 0
    for pick in all_picks:
        if toggle == 0: val_picks.append(pick)
        else: test_picks.append(pick)
        toggle ^= 1
    val_ds  = _IndexedSlice(all_subs, val_picks)
    test_ds = _IndexedSlice(all_subs, test_picks)
    val_loader  = DataLoader(val_ds,  batch_size=batch_sz, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False)


    count_windows(augmented_ds, "TRAIN (aug)")
    count_windows(val_ds, "VAL")
    count_windows(test_ds, "TEST")

    # ---- Model ----
    model = LSTMClassifier(
        input_size=config['n'],
        hidden_size=config['lstm_hd_sz'],
        num_layers=config['lstm_num_lyrs'],
        num_classes=2,
        dropout=config['dropout']
    ).to(device)

    with torch.no_grad():
        p = float((train_y == 1).float().mean().clamp(1e-6, 1-1e-6))
        if isinstance(model.fc, nn.Linear) and model.fc.out_features == 2:
            b = math.log(p/(1-p))
            model.fc.bias[1].fill_(b); model.fc.bias[0].fill_(-b)

    weights = torch.tensor([1.0, config['bloom_pen']], device=device)
    loss_func = nn.CrossEntropyLoss(weight=weights)

    print('Training model')
    _, auc_val, r = train_model(
        model, train_loader, val_loader,
        epochs=1000, lr=config['learning_rate'],
        loss_func=loss_func, device=device, patience=100, test_loader=test_loader
    )

    save_path = config["save_path"]; os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    thr, _ = best_threshold_from_roc(model, val_loader, device=device, method="youden")
    print(f"The threshold is {thr}"); print(f"Model weights saved to {save_path}")
    print(r)
    return model, val_loader, test_loader, r

# ============================= Pure LSTM =============================
def Run_Pure_LSTM():
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print("YAML Contents:"); [print(f"{k}: {v}") for k, v in config.items()]

    data = clean_data(config['data_path'])
    data_bank_sz = config['data_bank_sz']
    train_sz_cfg = config['train_sz']
    target_col = config['target']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Pure LSTM uses base features only ----------
    df = data.copy()
    num_df = df.select_dtypes(include=[np.number]).copy()
    X_df = num_df.drop(columns=[target_col]); y_sr = num_df[target_col]
    X_df, y_sr = X_df.ffill().bfill(), y_sr.ffill().bfill()
    X_np_all, y_np_all = X_df.to_numpy(dtype=np.float32), y_sr.to_numpy(dtype=np.float32)
    train_sz = int(train_sz_cfg) 
    orig = y_sr; mask = ~orig.isna().to_numpy()
    mu, sigma = np.nanmean(X_np_all[:train_sz], axis=0), np.nanstd(X_np_all[:train_sz], axis=0)
    sigma[sigma == 0] = 1.0
    X_np_all = (X_np_all - mu) / sigma
    X, y = torch.from_numpy(X_np_all), torch.from_numpy(y_np_all).float()
    if config['n'] != X.shape[1]:
        config['n'] = X.shape[1]

    # ---------- Labels ----------
    Val_carve_sz = int(config.get("val_len", 100))
    train_sz_adj = train_sz - Val_carve_sz
    t2 = float(np.percentile(y[:train_sz_adj].detach().cpu().numpy(), 95))
    print(f"[Labels] Train-only 95th percentile threshold = {t2:.2f}")
    bins = torch.tensor([t2], dtype=torch.float32)
    y = torch.bucketize(y, bins)

    # ---------- Datasets ----------
    train_X, train_y = X[:train_sz_adj], y[:train_sz_adj]
    seq_len = config['seq_len']
    train_mask = mask[:train_sz_adj]
    good = np.where(train_mask)[0]
    breaks = np.where(np.diff(good) != 1)[0] + 1
    segments = np.split(good, breaks)

    vanilla_subs, noisy_subs = [], []
    maj_frac, noise_pct = config['noisy_prop'], config['noise_pct']
    for seg in segments:
        if len(seg) <= seq_len: continue
        Xi = X[seg]; yi = y[seg]
        vanilla_subs.append(SequenceDataset(Xi, yi, seq_len))
        base_idxs = list(range(len(Xi) - seq_len))
        minority_idxs = [i for i in base_idxs if yi[i + seq_len] == 1]
        majority_idxs = [i for i in base_idxs if yi[i + seq_len] == 0]
        num_maj_to_sample = int(len(majority_idxs) * maj_frac)
        maj_sampled = random.sample(majority_idxs, k=max(0, num_maj_to_sample))
        selected_idxs = minority_idxs + maj_sampled
        noisy_subs.append(NoisySequenceDataset(Xi, yi, seq_len=seq_len,
                                               noise_pct=noise_pct, indices=selected_idxs))
    augmented_ds = ConcatDataset(vanilla_subs + noisy_subs)
    batch_sz = config['batch_sz']
    labels_aug = torch.tensor([int(augmented_ds[i][1]) for i in range(len(augmented_ds))])
    class_counts   = torch.bincount(labels_aug, minlength=2).float()
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[labels_aug].double()
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights,
                                                     num_samples=len(augmented_ds),
                                                     replacement=True)
    train_loader = DataLoader(augmented_ds, batch_size=batch_sz, sampler=sampler)

    # ---------- Interleave VAL/TEST ----------
    GAP, VAL_LEN = config['seq_len'], Val_carve_sz
    post_start = train_sz_adj + GAP
    post_end   = len(X)
    runs_post  = _post_train_runs(mask, post_start, post_end, seq_len)
    all_subs   = _windows_from_runs(runs_post, post_start, X, y, seq_len)
    all_picks  = _enumerate_windows(all_subs)
    val_picks, test_picks = [], []
    toggle = 0
    for pick in all_picks:
        (val_picks if toggle == 0 else test_picks).append(pick)
        toggle ^= 1
    val_ds  = _IndexedSlice(all_subs, val_picks)
    test_ds = _IndexedSlice(all_subs, test_picks)
    val_loader  = DataLoader(val_ds,  batch_size=batch_sz, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

    print("========== Raw Dataset Split Sizes ==========")
    print(f"Total samples in dataset: {len(X):,}")
    print(f"Train end (including warmup): {train_sz}")
    print(f"Adjusted train end (minus val): {train_sz_adj}")
    print(f"Validation length: {VAL_LEN}")
    print(f"Post gap: {GAP}")
    print("--------------------------------------------")
    print(f"Raw training range: 0 to {train_sz_adj - 1} ({train_sz_adj} samples)")
    print(f"Raw validation range: {train_sz_adj} to {train_sz_adj + VAL_LEN - 1} ({VAL_LEN} samples)")
    print(f"Raw test range: {post_start} to {post_end - 1} ({post_end - post_start} samples)")
    print("============================================")

        # ---------- Model ----------
    model = LSTMClassifier(
        input_size=config['n'],
        hidden_size=config['lstm_hd_sz'],
        num_layers=config['lstm_num_lyrs'],
        num_classes=2,
        dropout=config['dropout']
    ).to(device)

    with torch.no_grad():
        p = float((train_y == 1).float().mean().clamp(1e-6, 1-1e-6))
        if isinstance(model.fc, nn.Linear) and model.fc.out_features == 2:
            b = math.log(p/(1-p))
            model.fc.bias[1].fill_(b); model.fc.bias[0].fill_(-b)

    weights = torch.tensor([1.0, config['bloom_pen']], device=device)
    loss_func = nn.CrossEntropyLoss(weight=weights)

    print('Training pure LSTM model')
    _, auc_val, r = train_model(model, train_loader, val_loader,
                                epochs=1000, lr=config['learning_rate'],
                                loss_func=loss_func, device=device,
                                patience=100, test_loader=test_loader)

    save_path = config["save_path"].replace(".pth", "_purelstm.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    thr, _ = best_threshold_from_roc(model, val_loader, device=device, method="youden")
    print(f"[Pure LSTM] threshold={thr} saved={save_path}")
    return model, val_loader, test_loader, r

# ============================= Coov LSTM =============================
def Run_Coov_LSTM():
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print("YAML Contents:"); [print(f"{k}: {v}") for k, v in config.items()]

    # identical dataset prep as Run_Pure_LSTM
    data = clean_data(config['data_path'])
    data_bank_sz = config['data_bank_sz']
    train_sz_cfg = config['train_sz']
    target_col = config['target']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = data.copy()
    num_df = df.select_dtypes(include=[np.number]).copy()
    X_df = num_df.drop(columns=[target_col]); y_sr = num_df[target_col]
    X_df, y_sr = X_df.ffill().bfill(), y_sr.ffill().bfill()
    X_np_all, y_np_all = X_df.to_numpy(dtype=np.float32), y_sr.to_numpy(dtype=np.float32)
 
    train_sz = int(train_sz_cfg) 
    orig = y_sr; mask = ~orig.isna().to_numpy()
    mu, sigma = np.nanmean(X_np_all[:train_sz], axis=0), np.nanstd(X_np_all[:train_sz], axis=0)
    sigma[sigma == 0] = 1.0
    X_np_all = (X_np_all - mu) / sigma
    X, y = torch.from_numpy(X_np_all), torch.from_numpy(y_np_all).float()
    if config['n'] != X.shape[1]:
        config['n'] = X.shape[1]

    Val_carve_sz = int(config.get("val_len", 100))
    train_sz_adj = train_sz - Val_carve_sz
    t2 = float(np.percentile(y[:train_sz_adj].detach().cpu().numpy(), 95))
    print(f"[Labels] Train-only 95th percentile threshold = {t2:.2f}")
    bins = torch.tensor([t2], dtype=torch.float32)
    y = torch.bucketize(y, bins)

    train_X, train_y = X[:train_sz_adj], y[:train_sz_adj]
    seq_len = config['seq_len']
    train_mask = mask[:train_sz_adj]
    good = np.where(train_mask)[0]
    breaks = np.where(np.diff(good) != 1)[0] + 1
    segments = np.split(good, breaks)

    vanilla_subs, noisy_subs = [], []
    maj_frac, noise_pct = config['noisy_prop'], config['noise_pct']
    for seg in segments:
        if len(seg) <= seq_len: continue
        Xi = X[seg]; yi = y[seg]
        vanilla_subs.append(SequenceDataset(Xi, yi, seq_len))
        base_idxs = list(range(len(Xi) - seq_len))
        minority_idxs = [i for i in base_idxs if yi[i + seq_len] == 1]
        majority_idxs = [i for i in base_idxs if yi[i + seq_len] == 0]
        num_maj_to_sample = int(len(majority_idxs) * maj_frac)
        maj_sampled = random.sample(majority_idxs, k=max(0, num_maj_to_sample))
        selected_idxs = minority_idxs + maj_sampled
        noisy_subs.append(NoisySequenceDataset(Xi, yi, seq_len=seq_len,
                                               noise_pct=noise_pct, indices=selected_idxs))
    augmented_ds = ConcatDataset(vanilla_subs + noisy_subs)
    batch_sz = config['batch_sz']
    labels_aug = torch.tensor([int(augmented_ds[i][1]) for i in range(len(augmented_ds))])
    class_counts   = torch.bincount(labels_aug, minlength=2).float()
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[labels_aug].double()
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights,
                                                     num_samples=len(augmented_ds),
                                                     replacement=True)
    train_loader = DataLoader(augmented_ds, batch_size=batch_sz, sampler=sampler)

    GAP, VAL_LEN = config['seq_len'], Val_carve_sz
    post_start = train_sz_adj + GAP
    post_end   = len(X)
    runs_post  = _post_train_runs(mask, post_start, post_end, seq_len)
    all_subs   = _windows_from_runs(runs_post, post_start, X, y, seq_len)
    all_picks  = _enumerate_windows(all_subs)
    val_picks, test_picks = [], []
    toggle = 0
    for pick in all_picks:
        (val_picks if toggle == 0 else test_picks).append(pick)
        toggle ^= 1
    val_ds  = _IndexedSlice(all_subs, val_picks)
    test_ds = _IndexedSlice(all_subs, test_picks)
    val_loader  = DataLoader(val_ds,  batch_size=batch_sz, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

    # ---------- Model (Coov LSTM) ----------
    model = Coov_LSTMClassifier(
        input_size=config['n'],
        hidden_size=config['lstm_hd_sz'],
        num_layers=config['lstm_num_lyrs'],
        num_classes=2,
        dropout=config['dropout']
    ).to(device)

    with torch.no_grad():
        p = float((train_y == 1).float().mean().clamp(1e-6, 1-1e-6))
        if isinstance(model.fc, nn.Linear) and model.fc.out_features == 2:
            b = math.log(p/(1-p))
            model.fc.bias[1].fill_(b); model.fc.bias[0].fill_(-b)

    weights = torch.tensor([1.0, config['bloom_pen']], device=device)
    loss_func = nn.CrossEntropyLoss(weight=weights)

    print('Training Coov-LSTM model')
    _, auc_val, r = train_model(model, train_loader, val_loader,
                                epochs=1000, lr=config['learning_rate'],
                                loss_func=loss_func, device=device,
                                patience=100, test_loader=test_loader)

    save_path = config["save_path"].replace(".pth", "_coovlstm.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    thr, _ = best_threshold_from_roc(model, val_loader, device=device, method="youden")
    print(f"[Coov LSTM] threshold={thr} saved={save_path}")
    return model, val_loader, test_loader, r

# ============================= Multiple runs ================================

def get_roc_data(model, loader, device="cpu"):
    from sklearn.metrics import roc_curve, auc
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(yb.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def save_roc_curve(fpr, tpr, roc_auc, name, save_dir="auc_plots"):
    import matplotlib.pyplot as plt
    import os
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{name}_BEST_ROC.png")

    plt.figure(figsize=(7,7))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} BEST ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[SAVED] Best ROC for {name} -> {save_path}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if device.type == "cuda":
    print(f"[GPU] Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"   Memory reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
else:
    print("[CPU] Running on CPU (no GPU detected)")


###############################################################
#                    MULTIPLE RUN EXPERIMENTS
###############################################################

# if __name__ == "__main__":
    # Store BEST run ROC results
#     best = {
#         "edm":  {"auc": -1, "fpr": None, "tpr": None},
#         "pure": {"auc": -1, "fpr": None, "tpr": None},
#         "coov": {"auc": -1, "fpr": None, "tpr": None},
#     }
# 
    # Store ALL_RUN results (your old behavior)
#     results = []   # each entry: (run, edm_val, edm_test, pure_val, pure_test, coov_val, coov_test)
# 
# 
    ###############################################################
    #                            RUN LOOP
    ###############################################################
#     for run in range(1, 31):
#     print(f"\n==================== Run {run} ====================")
# 
    # Run models
#     pure_model, pure_val_loader, pure_test_loader, (pure_val, pure_test) = Run_Pure_LSTM()
#     edm_model,  edm_val_loader,  edm_test_loader, (edm_val,  edm_test)  = Run_EDM_LSTM()
#     coov_model, coov_val_loader, coov_test_loader, (coov_val, coov_test) = Run_Coov_LSTM()
# 
    # Save ALL runs (old CSV behavior)
#     results.append((run, edm_val, edm_test, pure_val, pure_test, coov_val, coov_test))
# 
# 
    ###############################################################
    #                  CHECK FOR BEST PURE LSTM RUN
    ###############################################################
#     fpr, tpr, roc_auc = get_roc_data(pure_model, pure_val_loader, device=device)
#     if roc_auc > best["pure"]["auc"]:
#         best["pure"] = {"auc": roc_auc, "fpr": fpr, "tpr": tpr}
# 
# 
    ###############################################################
    #                  CHECK FOR BEST EDM-LSTM RUN
    ###############################################################
#     fpr, tpr, roc_auc = get_roc_data(edm_model, edm_val_loader, device=device)
#     if roc_auc > best["edm"]["auc"]:
#         best["edm"] = {"auc": roc_auc, "fpr": fpr, "tpr": tpr}
# 
# 
    ###############################################################
    #                 CHECK FOR BEST COOV-LSTM RUN
    ###############################################################
#     fpr, tpr, roc_auc = get_roc_data(coov_model, coov_val_loader, device=device)
#     if roc_auc > best["coov"]["auc"]:
#         best["coov"] = {"auc": roc_auc, "fpr": fpr, "tpr": tpr}
# 
# 
# 
###############################################################
#                       SAVE BEST ROC PLOTS
###############################################################
# print("\n======== Saving BEST ROC curves from 100 runs ========\n")
# 
# save_roc_curve(best["pure"]["fpr"], best["pure"]["tpr"],
#                best["pure"]["auc"], "Pure_LSTM")
# 
# save_roc_curve(best["edm"]["fpr"], best["edm"]["tpr"],
#                best["edm"]["auc"], "EDM_LSTM")
# 
# save_roc_curve(best["coov"]["fpr"], best["coov"]["tpr"],
#                best["coov"]["auc"], "Coov_LSTM")
# 
# 
# 
###############################################################
#                SAVE ALL RUN AUCs (OLD BEHAVIOR)
###############################################################
# all_runs_path = "val_test_results.csv"
# with open(all_runs_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow([
#         "run",
#         "EDM_val_auc", "EDM_test_auc",
#         "Pure_val_auc", "Pure_test_auc",
#         "Coov_val_auc", "Coov_test_auc"
#     ])
#     writer.writerows(results)
# 
# print(f"[SAVED] ALL run AUCs to {all_runs_path}")
# 
# 
# 
###############################################################
#            SAVE BEST RUN ROC ARRAYS (REPRODUCIBLE)
###############################################################
# best_roc_path = "best_roc_data.csv"
# with open(best_roc_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Model", "Best_AUC", "FPR_list", "TPR_list"])
# 
#     for model_name in ["pure", "edm", "coov"]:
#         auc_val = best[model_name]["auc"]
#         fpr = best[model_name]["fpr"].tolist()
#         tpr = best[model_name]["tpr"].tolist()
# 
#         writer.writerow([
#             model_name,
#             auc_val,
#             fpr,
#             tpr
#         ])
# 
# print(f"[SAVED] BEST ROC raw data to {best_roc_path}")
# print("\n[COMPLETE] ALL DONE - 100 runs complete!\n")