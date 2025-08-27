from forecast import *  # load_yaml, clean_data, process_parameters, create_model, plot_predictions
import argparse, random, os, math
import numpy as np
import pandas as pd
import torch
from torch import nn
from copy import deepcopy
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             balanced_accuracy_score, roc_curve, auc, roc_auc_score)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tcn_model import *  # HybridTABWithGate, TrustGate, HybridTAB
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
                probs = torch.sigmoid(logits[:, 1] - logits[:, 0])
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
    patience=120,
    test_loader=None,
    gate_reg_l1: float = 1e-4,
    gate_reg_bin: float = 1e-4,
    freeze_gate_epochs: int = 0,
):
    import numpy as np
    import torch
    from torch import nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.metrics import balanced_accuracy_score, roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    model.to(device)

    # ---------- EMA as a state_dict buffer (no Module deepcopy) ----------
    @torch.no_grad()
    def _clone_state_dict(sd):
        return {k: v.detach().clone() for k, v in sd.items()}

    ema_decay = 0.995
    ema_state = _clone_state_dict(model.state_dict())

    @torch.no_grad()
    def ema_update():
        msd = model.state_dict()
        for k in ema_state.keys():
            ema_state[k].mul_(ema_decay).add_(msd[k].detach(), alpha=1.0 - ema_decay)

    # ---------- Optimizer / Scheduler ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=60, min_lr=1e-7)

    # ---------- Gate regularizer ----------
    def gate_regularizer(m):
        if not hasattr(m, "gate"):
            return 0.0
        g = torch.sigmoid(m.gate.gate_logits)
        eps = 1e-6
        l1_term  = g.mean()
        bin_term = -(g*torch.log(g+eps) + (1-g)*torch.log(1-g+eps)).mean()
        return gate_reg_l1 * l1_term + gate_reg_bin * bin_term

    # Optional warm-up: freeze the gate for first N epochs
    if hasattr(model, "gate") and freeze_gate_epochs > 0:
        for p in model.gate.parameters():
            p.requires_grad_(False)

    best_auc, final_test_auc, wait = 0.0, 0.0, 0
    best_state = None  # will hold EMA weights
    train_losses, val_losses = [], []

    for ep in range(1, epochs + 1):
        # Unfreeze gate after warm-up
        if hasattr(model, "gate") and ep == freeze_gate_epochs + 1:
            for p in model.gate.parameters():
                p.requires_grad_(True)

        # -------------------- TRAIN (live weights) --------------------
        model.train()
        running_train = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits = model(Xb)
            # Loss: handle CE (2 logits) or BCE-with-logits (single logit)
            if isinstance(loss_func, nn.BCEWithLogitsLoss):
                if logits.dim() > 1 and logits.size(-1) > 1:
                    pos_logit = logits[:, 1]  # use the positive-class logit
                else:
                    pos_logit = logits.view(-1)
                loss = loss_func(pos_logit, yb.float())
            else:
                loss = loss_func(logits, yb)

            loss = loss + gate_regularizer(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update EMA after each step
            ema_update()

            running_train += loss.item()

        train_loss = running_train / max(1, len(train_loader))
        train_losses.append(train_loss)

        # -------------------- VALIDATION (swap in EMA) --------------------
        running_val = 0.0
        y_true_list, y_score_list = [], []

        with torch.no_grad():
            live_state = _clone_state_dict(model.state_dict())   # save live
            model.load_state_dict(ema_state, strict=False)       # load EMA

            model.eval()
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)

                # probs for metrics
                if logits.dim() > 1 and logits.size(-1) == 2:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                else:
                    # single logit path
                    probs = torch.sigmoid(logits.view(-1))

                # val loss
                if isinstance(loss_func, nn.BCEWithLogitsLoss):
                    if logits.dim() > 1 and logits.size(-1) > 1:
                        pos_logit = logits[:, 1]
                    else:
                        pos_logit = logits.view(-1)
                    loss_val = loss_func(pos_logit, yb.float())
                else:
                    loss_val = loss_func(logits, yb)

                running_val += loss_val.item()
                y_true_list.append(yb.cpu().numpy())
                y_score_list.append(probs.cpu().numpy())

            model.load_state_dict(live_state, strict=False)      # restore live

        val_loss = running_val / max(1, len(val_loader))
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        y_true  = np.concatenate(y_true_list) if y_true_list else np.array([])
        y_score = np.concatenate(y_score_list) if y_score_list else np.array([])

        # Balanced accuracy at ROC-optimal threshold (Youden J)
        if y_true.size > 0 and np.unique(y_true).size == 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
            finite = np.isfinite(thresholds)
            fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
            if thresholds.size:
                idx = np.argmax(tpr - fpr)
                thr_opt = float(thresholds[idx])
            else:
                thr_opt = 0.5
            y_pred = (y_score >= thr_opt).astype(int)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            auc_val = roc_auc_score(y_true, y_score)
        else:
            thr_opt, bal_acc, auc_val = 0.5, 0.5, 0.5  # degenerate safeguard

        # -------------------- TEST (EMA) --------------------
        test_auc = None
        if test_loader is not None:
            with torch.no_grad():
                live_state = _clone_state_dict(model.state_dict())
                model.load_state_dict(ema_state, strict=False)

                model.eval()
                t_true, t_score = [], []
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    logits = model(Xb)
                    if logits.dim() > 1 and logits.size(-1) == 2:
                        probs = torch.softmax(logits, dim=1)[:, 1]
                    else:
                        probs = torch.sigmoid(logits.view(-1))
                    t_true.append(yb.cpu().numpy())
                    t_score.append(probs.cpu().numpy())

                t_true = np.concatenate(t_true) if t_true else np.array([])
                t_score = np.concatenate(t_score) if t_score else np.array([])
                if t_true.size > 0 and np.unique(t_true).size == 2:
                    test_auc = roc_auc_score(t_true, t_score)
                model.load_state_dict(live_state, strict=False)

        if ep % 5 == 0:
            msg = (f"Epoch {ep:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                   f"bal_acc={bal_acc:.4f}  val_auc={auc_val:.4f}")
            if test_loader is not None and test_auc is not None:
                msg += f" test_auc={test_auc:.4f}"
            msg += f" thr={thr_opt:.3f}"
            print(msg)

        # Early stop criterion on VAL AUC (using EMA snapshot)
        if auc_val > best_auc:
            best_auc = auc_val
            best_state = _clone_state_dict(ema_state)   # store EMA snapshot
            final_test_auc = (test_auc or final_test_auc)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping at epoch {ep}")
                break

    # ---------- Plot losses ----------
    try:
        plt.figure(figsize=(6,4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses,   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training & Validation Loss")
        plt.tight_layout(); plt.show()
    except Exception:
        pass  # plotting optional

    # ---------- Load best EMA snapshot and compute final TEST AUC ----------
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        if test_loader is not None:
            with torch.no_grad():
                model.eval()
                t_true, t_score = [], []
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    logits = model(Xb)
                    if logits.dim() > 1 and logits.size(-1) == 2:
                        probs = torch.softmax(logits, dim=1)[:, 1]
                    else:
                        probs = torch.sigmoid(logits.view(-1))
                    t_true.append(yb.cpu().numpy()); t_score.append(probs.cpu().numpy())
            t_true = np.concatenate(t_true) if t_true else np.array([])
            t_score = np.concatenate(t_score) if t_score else np.array([])
            if t_true.size > 0 and np.unique(t_true).size == 2:
                final_test_auc = roc_auc_score(t_true, t_score)
        print(f"Loading best EMA model (val AUC={best_auc:.4f})  Test_AUC={final_test_auc:.4f}")
    else:
        print("Best model not loaded")

    return model, final_test_auc


# ========================= Mixing helpers (VAL/TEST) =========================

class _IndexedSlice(Dataset):
    """Picks specific window indices (s_i, k) from a list of SequenceDatasets."""
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
        L = len(sub)
        for k in range(L):
            picks.append((s_i, k))
    return picks

# ================================== Main ====================================

def main():
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_yaml(args.config)
    print("YAML Contents:"); [print(f"{k}: {v}") for k, v in config.items()]

    data = clean_data(config['data_path'])
    data_bank_sz = config['data_bank_sz']
    train_sz_cfg = config['train_sz']

    parameters = process_parameters(config['parameters_path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['EDM_embed']:
        print('Creating EDM Models and Embedding Data')
        data_bank = EDM_bank(data_bank_sz, parameters, config['target'], config['samp'], config['n'])
        embedder = Embedd_EDM(data_bank, data, config['forecast_type'])
        orig = data[config['target']]
        mask = ~orig.iloc[config['data_bank_sz']+1:].isna().to_numpy()
        X, y = embedder.get_data()

        # ---------- Add seasonality features (sin/cos week-of-year) ----------
        weeks = pd.to_datetime(data['DATE']).dt.isocalendar().week.to_numpy()
        phi = 2*np.pi*weeks/52.0
        season = torch.from_numpy(np.stack([np.sin(phi), np.cos(phi)], axis=1)).float()
        season = season[data_bank_sz+1:]  # align to X,y
        X = torch.cat([X, season], dim=1)

        # ---------- Split sizes ----------
        Val_carve_sz = int(config.get("val_len", 100))
        train_sz = int(train_sz_cfg)
        assert train_sz > Val_carve_sz + config['seq_len'], "Increase train_sz or reduce seq_len."

        # ---------- Train-only scaling ----------
        mu = X[:train_sz].mean(dim=0)
        sigma = X[:train_sz].std(dim=0).clamp_min(1e-6)
        X = (X - mu) / sigma

        # ensure model input size matches
        if config['n'] != X.shape[1]:
            print(f"[INFO] Adjusting config['n'] from {config['n']} -> {X.shape[1]} after season+scaling.")
            config['n'] = X.shape[1]
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
        train_sz = int(train_sz_cfg) + warmup
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
        # ---- build 0/1 labels with TRAIN-ONLY threshold (no leakage) ----
        Val_carve_sz = int(config.get("val_len", 100))
        train_sz_adj = train_sz - Val_carve_sz
        assert train_sz_adj > 0, "train_sz must exceed val_len."

        # percentile on TRAIN ONLY
        t2 = float(np.percentile(y[:train_sz_adj].detach().cpu().numpy(), 95))
        print(f"[Labels] Train-only 95th percentile threshold = {t2:.2f}")
        bins = torch.tensor([t2], dtype=torch.float32)  # for 2 classes
        y = torch.bucketize(y, bins)

        # ---- split into train/val/test base tensors ----
        train_X, train_y = X[:train_sz_adj], y[:train_sz_adj]

        seq_len = config['seq_len']  # window length

        # 1) Segment-by-segment windowing to respect gaps (TRAIN)
        train_mask = mask[:train_sz_adj]
        good = np.where(train_mask)[0]
        breaks = np.where(np.diff(good) != 1)[0] + 1
        segments = np.split(good, breaks)

        vanilla_subs, noisy_subs = [], []
        maj_frac = config['noisy_prop']
        noise_pct = config['noise_pct']

        for seg in segments:
            if len(seg) <= seq_len:
                continue
            Xi = X[seg]; yi = y[seg]
            vanilla_subs.append(SequenceDataset(Xi, yi, seq_len))

            base_idxs = list(range(len(Xi) - seq_len))
            minority_idxs = [i for i in base_idxs if yi[i + seq_len] == 1]
            majority_idxs = [i for i in base_idxs if yi[i + seq_len] == 0]
            num_maj_to_sample = int(len(majority_idxs) * maj_frac)
            maj_sampled = random.sample(majority_idxs, k=max(0, num_maj_to_sample))
            selected_idxs = minority_idxs + maj_sampled

            noisy_subs.append(
                NoisySequenceDataset(Xi, yi, seq_len=seq_len, noise_pct=noise_pct, indices=selected_idxs)
            )

        augmented_ds = ConcatDataset(vanilla_subs + noisy_subs)

        batch_sz = config['batch_sz']
        # —— TRAIN loader (weighted sampling on augmented)
        labels_aug = torch.tensor([int(augmented_ds[i][1]) for i in range(len(augmented_ds))])
        class_counts   = torch.bincount(labels_aug, minlength=2).float()
        class_weights  = 1.0 / class_counts
        sample_weights = class_weights[labels_aug].double()
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights,
                                                         num_samples=len(augmented_ds),
                                                         replacement=True)
        train_loader = DataLoader(augmented_ds, batch_size=batch_sz, sampler=sampler)

        # ========================= VAL/TEST MIXING =========================
        mix_mode   = str(config.get("val_test_mix_mode", "block")).lower()     # "block" | "interleave" | "random"
        val_ratio  = float(config.get("val_mix_ratio", 0.4))                   # only for "random"
        mix_seed   = int(config.get("mix_seed", 1337))

        GAP, VAL_LEN = config['seq_len'], Val_carve_sz
        val_start, val_end = train_sz_adj + GAP, train_sz_adj + GAP + VAL_LEN
        test_start, test_end = val_end, len(X)

        if mix_mode == "block":
            # —— Original contiguous VAL, contiguous TEST
            # VAL windows
            mask_val = mask[val_start:val_end]
            good_val = np.where(mask_val)[0]
            cuts = np.where(np.diff(good_val) != 1)[0] + 1
            runs = [r for r in np.split(good_val, cuts) if len(r) > seq_len]
            val_subs = [SequenceDataset(X[r + val_start], y[r + val_start], seq_len) for r in runs]
            val_ds = ConcatDataset(val_subs)

            # TEST windows
            mask_test = mask[test_start:test_end]
            good_test = np.where(mask_test)[0]
            cuts = np.where(np.diff(good_test) != 1)[0] + 1
            runs = [r for r in np.split(good_test, cuts) if len(r) > seq_len]
            test_subs = [SequenceDataset(X[r + test_start], y[r + test_start], seq_len) for r in runs]
            test_ds = ConcatDataset(test_subs)

        else:
            # Build all post-train windows, then split by interleave or random
            post_start = val_start
            post_end   = len(X)
            runs_post  = _post_train_runs(mask, post_start, post_end, seq_len)
            all_subs   = _windows_from_runs(runs_post, post_start, X, y, seq_len)
            all_picks  = _enumerate_windows(all_subs)

            if mix_mode == "interleave":
                val_picks, test_picks = [], []
                toggle = 0
                for pick in all_picks:
                    if toggle == 0: val_picks.append(pick)
                    else:           test_picks.append(pick)
                    toggle ^= 1
            elif mix_mode == "random":
                rs = np.random.RandomState(mix_seed)
                perm = rs.permutation(len(all_picks))
                cut  = int(len(perm) * val_ratio)
                val_picks  = [all_picks[i] for i in perm[:cut]]
                test_picks = [all_picks[i] for i in perm[cut:]]
            else:
                raise ValueError("val_test_mix_mode must be 'block', 'interleave', or 'random'")

            val_ds  = _IndexedSlice(all_subs, val_picks)
            test_ds = _IndexedSlice(all_subs, test_picks)

        val_loader  = DataLoader(val_ds,  batch_size=batch_sz, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

        # Debug counts
        count_windows(augmented_ds, "TRAIN (aug)")
        count_windows(val_ds, "VAL")
        count_windows(test_ds, "TEST")

        # =================== AUC-based gate init (choose via config) ===================
        gate_mode   = str(config.get('gate_init_mode', 'continuous'))  # 'continuous' or 'topk'
        gate_hi     = float(config.get('gate_hi', 0.8))
        gate_lo     = float(config.get('gate_lo', 0.2))
        gate_top_k  = int(config.get('gate_top_k', min(100, config['n'])))
        gate_sharp  = float(config.get('gate_sharp', 3.0))
        gate_temp   = float(config.get('gate_temperature', 6.0))  # softer by default
        freeze_g    = int(config.get('gate_freeze_epochs', 10))

        X_flat, y_flat = X[:train_sz_adj], y[:train_sz_adj]
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

        # Prior-logit bias to match train prevalence
        with torch.no_grad():
            p = float((train_y == 1).float().mean().clamp(1e-6, 1-1e-6))
            last = model.core.classifier[-1]
            if isinstance(last, nn.Linear) and last.out_features == 2:
                b = math.log(p/(1-p))
                last.bias[1].fill_(b); last.bias[0].fill_(-b)

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

        # —— Save best, threshold, confusion matrices, ROC (EMA-best already loaded)
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
