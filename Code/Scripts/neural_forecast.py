from forecast import *  # load_yaml, clean_data, process_parameters, create_model, ...
import argparse, random, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from copy import deepcopy
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, roc_curve, auc, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tcn_model import *  # HybridTABWithGate, TrustGate, HybridTAB
from losses import *

# ============================== EDM interface ===============================

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

# ============================ Simple MLP baselines ===========================

class HybridNeuralNetwork(nn.Module):
    def __init__(self, n, hd_sz, drop=0.5):
        super().__init__()
        self.ffw1 = nn.Linear(n, hd_sz)
        self.bn1  = nn.BatchNorm1d(hd_sz)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.ffw2 = nn.Linear(hd_sz, 1)
    def forward(self, x):
        return self.ffw2(self.drop(self.relu(self.bn1(self.ffw1(x)))))

class HybridNNCLS(nn.Module):
    def __init__(self, n, hd_sz, drop=0.5):
        super().__init__()
        self.ffw1 = nn.Linear(n, hd_sz)
        self.relu = nn.ReLU()
        self.ffw2 = nn.Linear(hd_sz, 4)
    def forward(self, x):
        return self.ffw2(self.relu(self.ffw1(x)))

# ============================== Datasets ====================================

class Embedd_EDM():
    def __init__(self, EDM, data, type):
        self.EDM_bank = EDM
        self.target = EDM.target
        y_raw = torch.tensor(data[self.EDM_bank.target].iloc[self.EDM_bank.lib_sz+1:].to_numpy(),
                             dtype=torch.float32)
        self.targets = torch.log1p(y_raw) if type == 'regression' else y_raw
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
                diff = logits[:, pos_label] - logits[:, 1 - pos_label]
                probs = torch.sigmoid(diff)
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
                diff  = logits[:, pos_label] - logits[:, 1 - pos_label]
                probs = torch.sigmoid(diff)
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
            elif logits.size(-1) == 2:
                probs = torch.sigmoid(logits[:, pos_label] - logits[:, 1 - pos_label])
            else:
                probs = torch.softmax(logits, dim=1)[:, pos_label]
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

# ================= AUC-based gate initializers (pick one) ===================

def init_probs_auc_topk(X_flat: torch.Tensor, y_flat: torch.Tensor,
                        top_k: int = 100, hi=0.98, lo=0.02):
    from sklearn.metrics import roc_auc_score
    Xn = X_flat.detach().cpu().numpy()
    yn = y_flat.detach().cpu().numpy().astype(int)
    n  = Xn.shape[1]; aucs = np.zeros(n, dtype=float)
    for j in range(n):
        xj = Xn[:, j]
        try:
            a = roc_auc_score(yn, xj)
            aucs[j] = max(a, 1 - a)  # flip-invariant
        except Exception:
            aucs[j] = 0.5
    order = np.argsort(-aucs)
    probs = np.full(n, lo, dtype=np.float32)
    probs[order[:min(top_k, n)]] = hi
    return torch.from_numpy(probs), aucs

def init_probs_auc_continuous(X_flat: torch.Tensor, y_flat: torch.Tensor,
                              lo=0.05, hi=0.95, center=0.5, sharp=6.0):
    from sklearn.metrics import roc_auc_score
    Xn = X_flat.detach().cpu().numpy()
    yn = y_flat.detach().cpu().numpy().astype(int)
    n  = Xn.shape[1]; aucs = np.zeros(n, dtype=float)
    for j in range(n):
        xj = Xn[:, j]
        try:
            a = roc_auc_score(yn, xj)
            aucs[j] = max(a, 1 - a)
        except Exception:
            aucs[j] = 0.5
    a = np.clip(aucs, 0.5, 1.0)
    z = (a - center) * sharp
    probs = lo + (hi - lo) * (1 / (1 + np.exp(-z)))
    return torch.from_numpy(probs.astype(np.float32)), aucs

# ============================= Trainer ======================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    loss_func=torch.nn.CrossEntropyLoss(),
    device="cpu",
    patience=10,
    test_loader=None,
    gate_reg_l1: float = 1e-4,
    gate_reg_bin: float = 1e-4,
    freeze_gate_epochs: int = 0,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-7)

    def gate_regularizer(m):
        if not hasattr(m, "gate"): return 0.0
        g = torch.sigmoid(m.gate.gate_logits)
        eps = 1e-6
        l1_term  = g.mean()
        bin_term = -(g*torch.log(g+eps) + (1-g)*torch.log(1-g+eps)).mean()
        return gate_reg_l1 * l1_term + gate_reg_bin * bin_term

    # optional warm-up: freeze the gate first N epochs
    if hasattr(model, "gate") and freeze_gate_epochs > 0:
        for p in model.gate.parameters(): p.requires_grad_(False)

    best_auc, final_test_auc, wait = 0.0, 0.0, 0
    best_state, train_losses, val_losses = None, [], []

    for ep in range(1, epochs + 1):
        # unfreeze gate after warm-up
        if hasattr(model, "gate") and ep == freeze_gate_epochs + 1:
            for p in model.gate.parameters(): p.requires_grad_(True)

        # ---- TRAIN ----
        model.train(); running_train = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = (loss_func(logits, yb) if not isinstance(loss_func, nn.BCEWithLogitsLoss)
                    else loss_func(logits[:, 1], yb.float()))
            loss = loss + gate_regularizer(model)
            loss.backward(); optimizer.step()
            running_train += loss.item()
        train_loss = running_train / len(train_loader); train_losses.append(train_loss)

        # ---- VAL ----
        model.eval(); running_val = 0.0; all_trues, all_preds, all_scores = [], [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                if isinstance(loss_func, nn.BCEWithLogitsLoss):
                    probs = torch.sigmoid(logits[:, 1] if logits.size(-1) > 1 else logits.view(-1))
                    loss = loss_func(logits[:, 1], yb.float())
                    preds = (probs > 0.5).long()
                else:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    loss = loss_func(logits, yb)
                    preds = torch.argmax(logits, dim=1)
                running_val += loss.item()
                all_trues.append(yb.cpu()); all_preds.append(preds.cpu()); all_scores.append(probs.cpu().numpy())

        val_loss = running_val / len(val_loader); val_losses.append(val_loss)
        scheduler.step(val_loss)

        y_true  = np.concatenate(all_trues)
        y_score = np.concatenate(all_scores)
        y_pred  = torch.cat(all_preds).numpy()
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        auc_val = roc_auc_score(y_true, y_score)

        test_auc = None
        if test_loader is not None:
            t_trues, t_scores = [], []
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    logits = model(Xb)
                    probs = (torch.sigmoid(logits[:, 1]) if logits.size(-1) > 1
                             else torch.sigmoid(logits.view(-1)))
                    t_trues.append(yb.cpu().numpy()); t_scores.append(probs.cpu().numpy())
            y_true_t, y_score_t = np.concatenate(t_trues), np.concatenate(t_scores)
            test_auc = roc_auc_score(y_true_t, y_score_t)

        if ep % 10 == 0:
            if test_loader is None:
                print(f"Epoch {ep:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  bal_acc={bal_acc:.4f}  auc={auc_val:.4f}")
            else:
                print(f"Epoch {ep:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  bal_acc={bal_acc:.4f}  val_auc={auc_val:.4f} test_auc={test_auc:.4f}")

        if auc_val > best_auc:
            best_auc, best_state, final_test_auc, wait = auc_val, deepcopy(model.state_dict()), (test_auc or final_test_auc), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping at epoch {ep}")
                model.load_state_dict(best_state); break

    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss"); plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training & Validation Loss")
    plt.tight_layout(); plt.show()

    if best_state is not None:
        model.load_state_dict(best_state)
        if test_loader is not None:
            t_trues, t_scores = [], []
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    logits = model(Xb)
                    probs = (torch.sigmoid(logits[:, 1]) if logits.size(-1) > 1
                             else torch.sigmoid(logits.view(-1)))
                    t_trues.append(yb.cpu().numpy()); t_scores.append(probs.cpu().numpy())
            final_test_auc = roc_auc_score(np.concatenate(t_trues), np.concatenate(t_scores))
        print(f"Loading best model (val AUC={best_auc:.4f})  Test_AUC={final_test_auc:.4f}")
    else:
        print("Best model not loaded")
    return model, final_test_auc

# ================================== Main ====================================

def main():
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_yaml(args.config)
    print("YAML Contents:"); [print(f"{k}: {v}") for k, v in config.items()]

    data = clean_data(config['data_path'])
    data_bank_sz = config['data_bank_sz']
    train_sz = config['train_sz']
    parameters = process_parameters(config['parameters_path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['EDM_embed']:
        print('Creating EDM Models and Embedding Data')
        data_bank = EDM_bank(data_bank_sz, parameters, config['target'], config['samp'], config['n'])
        embedder = Embedd_EDM(data_bank, data, config['forecast_type'])
        orig = data[config['target']]
        mask = ~orig.iloc[config['data_bank_sz']+1:].isna().to_numpy()
        X, y = embedder.get_data()
        VAL_LEN = 100
        assert train_sz > VAL_LEN + config['seq_len'], "Increase train_sz or reduce seq_len."
    else:
        print('Building base features (no EDM)')
        target_col = config['target']
        df = data.copy()
        num_df = df.select_dtypes(include=[np.number]).copy()
        if target_col not in num_df.columns:
            raise ValueError(f"Target '{target_col}' must be numeric and present in the dataframe.")
        X_df = num_df.drop(columns=[target_col]); y_sr = num_df[target_col]
        X_df, y_sr = X_df.ffill().bfill(), y_sr.ffill().bfill()
        X_np_all, y_np_all = X_df.to_numpy(dtype=np.float32), y_sr.to_numpy(dtype=np.float32)
        warmup = config['data_bank_sz'] + 1
        train_sz = config['train_sz'] + warmup
        orig = y_sr; mask = ~orig.isna().to_numpy()
        mu, sigma = np.nanmean(X_np_all[:train_sz], axis=0), np.nanstd(X_np_all[:train_sz], axis=0)
        sigma[sigma == 0] = 1.0
        X_np_all = (X_np_all - mu) / sigma
        X, y = torch.from_numpy(X_np_all), torch.from_numpy(y_np_all).float()
        if config['n'] != X.shape[1]:
            print(f"[WARN] config['n']={config['n']} ≠ n_features={X.shape[1]}. Using detected n_features.")
            config['n'] = X.shape[1]

    if config['forecast_type'] == 'regression':
        train_X, train_y = X[:train_sz], y[:train_sz]
        test_X,  test_y  = X[train_sz:], y[train_sz:]
        train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=16, shuffle=True)
        test_loader  = DataLoader(TensorDataset(test_X,  test_y),  batch_size=16, shuffle=False)
        model = HybridNeuralNetwork(config['n'], 1000)
        print('Training model')
        train_model(model, train_loader, test_loader=None, epochs=500, lr=1e-5)
        plot_predictions(model, train_X, train_y, config['target'], "Train", "plots/train_true_vs_pred.png")
        plot_predictions(model, test_X,  test_y,  config['target'], "Test",  "plots/test_true_vs_pred.png")

    elif config['forecast_type'] == 'lstm':
        # —— binarize labels
        t2 = np.percentile(data[config['target']], 95)
        print(f"Bloom threshold is {t2}")
        bins = torch.tensor([t2], dtype=torch.float32)
        y = torch.bucketize(y, bins)  # 0/1

        Val_carve_sz = 126
        train_sz = train_sz - Val_carve_sz
        train_X, train_y = X[:train_sz], y[:train_sz]
        val_X,   val_y   = X[train_sz:train_sz+Val_carve_sz], y[train_sz:train_sz+Val_carve_sz]
        test_X,  test_y  = X[train_sz+Val_carve_sz:],         y[train_sz+Val_carve_sz:]

        seq_len = config['seq_len']

        # —— Build TRAIN windows + augmented noisy subset
        train_mask = mask[:train_sz]
        good = np.where(train_mask)[0]
        breaks = np.where(np.diff(good) != 1)[0] + 1
        segments = np.split(good, breaks)

        vanilla_subs, noisy_subs = [], []
        maj_frac, noise_pct = config['noisy_prop'], config['noise_pct']

        for seg in segments:
            if len(seg) <= seq_len: continue
            Xi, yi = X[seg], y[seg]
            vanilla_subs.append(SequenceDataset(Xi, yi, seq_len))
            base_idxs = list(range(len(Xi) - seq_len))
            minority = [i for i in base_idxs if yi[i + seq_len] == 1]
            majority = [i for i in base_idxs if yi[i + seq_len] == 0]
            kmaj = int(len(majority) * maj_frac)
            selected = minority + random.sample(majority, k=kmaj)
            noisy_subs.append(NoisySequenceDataset(Xi, yi, seq_len=seq_len, noise_pct=noise_pct, indices=selected))

        augmented_ds = ConcatDataset(vanilla_subs + noisy_subs)
        batch_sz = config['batch_sz']
        GAP, VAL_LEN = config['seq_len'], Val_carve_sz
        train_end = train_sz

        # —— VAL windows
        val_start, val_end = train_end + GAP, train_end + GAP + VAL_LEN
        mask_val = mask[val_start:val_end]
        good_val = np.where(mask_val)[0]
        cuts = np.where(np.diff(good_val) != 1)[0] + 1
        runs = [r for r in np.split(good_val, cuts) if len(r) > seq_len]
        val_subs = []
        for r in runs:
            idx = r + val_start
            val_subs.append(SequenceDataset(X[idx], y[idx], seq_len))
        val_ds = ConcatDataset(val_subs)

        # —— TEST windows
        test_start, test_end = val_end, len(X)
        mask_test = mask[test_start:test_end]
        good_test = np.where(mask_test)[0]
        cuts = np.where(np.diff(good_test) != 1)[0] + 1
        runs = [r for r in np.split(good_test, cuts) if len(r) > seq_len]
        test_subs = []
        for r in runs:
            idx = r + test_start
            test_subs.append(SequenceDataset(X[idx], y[idx], seq_len))
        test_ds = ConcatDataset(test_subs)

        # —— Loaders
        val_loader  = DataLoader(val_ds,  batch_size=batch_sz, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

        # —— Weighted sampling on augmented train
        labels_aug = torch.tensor([int(augmented_ds[i][1]) for i in range(len(augmented_ds))])
        class_counts   = torch.bincount(labels_aug, minlength=2).float()
        class_weights  = 1.0 / class_counts
        sample_weights = class_weights[labels_aug].double()
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights,
                                                         num_samples=len(augmented_ds),
                                                         replacement=True)
        train_loader = DataLoader(augmented_ds, batch_size=batch_sz, sampler=sampler)

        # =================== AUC-based gate init (choose via config) ===================
        # Required config keys with defaults if missing:
        gate_mode   = str(config.get('gate_init_mode', 'topk'))  # 'topk' or 'continuous'
        gate_hi     = float(config.get('gate_hi', 0.98))
        gate_lo     = float(config.get('gate_lo', 0.02))
        gate_top_k  = int(config.get('gate_top_k', min(100, config['n'])))
        gate_sharp  = float(config.get('gate_sharp', 6.0))
        gate_temp   = float(config.get('gate_temperature', 8.0))
        freeze_g    = int(config.get('gate_freeze_epochs', 0))

        X_flat, y_flat = X[:train_sz], y[:train_sz]
        if gate_mode.lower().startswith('cont'):
            init_probs, _ = init_probs_auc_continuous(X_flat, y_flat, lo=gate_lo, hi=gate_hi,
                                                      center=0.5, sharp=gate_sharp)
        else:
            init_probs, _ = init_probs_auc_topk(X_flat, y_flat, top_k=gate_top_k, hi=gate_hi, lo=gate_lo)

        # —— Model
        channel_sz = config['tcn_channel']
        model = HybridTABWithGate(
            input_size = config['n'],
            tcn_channels=[channel_sz, channel_sz],
            mha_heads=config['atn_hds'],
            lstm_hidden=config['lstm_hd_sz'],
            lstm_layers=config['lstm_num_lyrs'],
            num_classes=2,
            dropout=config['dropout'],
            gate_init_probs=init_probs,
            gate_temperature=gate_temp
        ).to(device)

        # —— Loss with class weighting for imbalance
        weights = torch.tensor([1.0, config['bloom_pen']], device=device)
        loss_func = nn.CrossEntropyLoss(weight=weights)

        # —— Train (with optional gate warm-up freeze)
        print('Training model')
        _, auc_val = train_model(
            model, train_loader, val_loader,
            epochs=100000, lr=config['learning_rate'],
            loss_func=loss_func, device=device, patience=400, test_loader=test_loader,
            gate_reg_l1=1e-4, gate_reg_bin=1e-4, freeze_gate_epochs=freeze_g
        )

        # —— Save best, threshold, confusion matrices, ROC
        save_path = config["save_path"]; os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        thr, _ = best_threshold_from_roc(model, val_loader, device=device, method="youden")
        print(f"The threshold is {thr}"); print(f"Model weights saved to {save_path}")

        class_names = ["no bloom", "bloom"]
        print("=== Train Confusion Matrix ===")
        _ = eval_confusion(model, train_loader, device, labels=class_names, title="Train: True vs Predicted", threshold=thr)
        print("=== Test Confusion Matrix ===")
        _ = eval_confusion(model, test_loader,  device, labels=class_names, title="Test: True vs Predicted",  threshold=thr)
        print("=== Test ROC ==="); plot_roc_with_thresholds(model, test_loader, device=device, pos_label=1)

if __name__ == "__main__":
    main()
