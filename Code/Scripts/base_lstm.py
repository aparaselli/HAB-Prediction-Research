# base_lstm.py
import matplotlib
matplotlib.use("Agg")
import yaml, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# -------------------- Datasets --------------------
class SequenceDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, seq_len: int):
        assert len(X) == len(y)
        self.X, self.y, self.seq_len = X, y, seq_len
    def __len__(self): return max(0, len(self.X) - self.seq_len)
    def __getitem__(self, i):
        return self.X[i:i+self.seq_len], self.y[i+self.seq_len]

class IndexedDataset(Dataset):
    def __init__(self, base_ds: SequenceDataset, picks: list[int]):
        self.base = base_ds; self.picks = picks
    def __len__(self): return len(self.picks)
    def __getitem__(self, i): return self.base[self.picks[i]]

# -------------------- Model --------------------
class BaselineLSTM(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, num_layers=layers,
            batch_first=True, dropout=(dropout if layers > 1 else 0.0)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 2)
    def forward(self, x):
        _, (hT, _) = self.lstm(x)
        h = self.dropout(hT[-1])
        return self.fc(h)

# -------------------- AUC helper --------------------
@torch.no_grad()
def eval_auc(model, loader, device):
    if loader is None: return float("nan")
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.float().to(device)
        logits = model(xb)
        prob1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        ys.append(yb.numpy()); ps.append(prob1)
    if not ys: return float("nan")
    y = np.concatenate(ys); p = np.concatenate(ps)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size == 0 or len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, p))

# -------------------- Main --------------------
def run_baseline(cfg_path="configs/lstm_carter_script.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path   = cfg["data_path"]
    target_col  = cfg["target"]
    train_sz    = int(cfg["train_sz"])
    seq_len     = int(cfg["seq_len"])
    batch_sz    = int(cfg["batch_sz"])
    hidden      = int(cfg["lstm_hd_sz"])
    layers      = int(cfg["lstm_num_lyrs"])
    dropout     = float(cfg["dropout"])
    lr          = float(cfg["learning_rate"])
    bloom_pen   = float(cfg["bloom_pen"])
    bloom_thr   = float(cfg["bloom_thresh"])
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    df = pd.read_csv(data_path)
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan)
    mu_imp = num.iloc[:train_sz].mean(skipna=True)
    num = num.fillna(mu_imp)

    y_raw = num[target_col].to_numpy(np.float32)
    X_np  = num.drop(columns=[target_col]).to_numpy(np.float32)

    mu = X_np[:train_sz].mean(axis=0)
    sd = X_np[:train_sz].std(axis=0); sd[sd == 0] = 1.0
    X_np = (X_np - mu) / sd

    y_bin = (y_raw >= bloom_thr).astype(np.int64)

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_bin)

    train_ds = SequenceDataset(X[:train_sz], y[:train_sz], seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)

    post_start = train_sz + seq_len
    X_post, y_post = X[post_start:], y[post_start:]
    post_ds = SequenceDataset(X_post, y_post, seq_len)
    n_post = len(post_ds)

    val_idx  = list(range(0, n_post, 2))
    test_idx = list(range(1, n_post, 2))
    val_loader  = DataLoader(IndexedDataset(post_ds, val_idx),  batch_size=batch_sz, shuffle=False)
    test_loader = DataLoader(IndexedDataset(post_ds, test_idx), batch_size=batch_sz, shuffle=False)

    # ---- Model ----
    input_size = X.shape[1]
    model = BaselineLSTM(input_size, hidden, layers, dropout).to(device)
    class_w = torch.tensor([1.0, bloom_pen], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Training ----
    train_losses, val_losses = [], []
    best_val, best_test, wait, patience = -1.0, float("nan"), 0, 100
    max_epochs, print_every = 1000, 25

    for ep in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        running_train = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.float().to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_train += loss.item()
        train_loss = running_train / max(1, len(train_loader))
        train_losses.append(train_loss)

        # --- Validation loss ---
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.float().to(device), yb.to(device)
                logits = model(xb)
                loss_val = loss_fn(logits, yb.long())
                running_val += loss_val.item()
        val_loss = running_val / max(1, len(val_loader))
        val_losses.append(val_loss)

        # --- Metrics ---
        val_auc  = eval_auc(model, val_loader,  device)
        test_auc = eval_auc(model, test_loader, device)

        improved = (val_auc > best_val)
        if improved:
            best_val, best_test = val_auc, test_auc
            wait = 0
        else:
            wait += 1

        if ep % 25 == 0:
            print(f"Epoch {ep:4d} | train_loss={train_loss:.4f} "
                  f"| val_loss={val_loss:.4f} | val_auc={val_auc:.4f} "
                  f"| test_auc={test_auc:.4f} | best_val={best_val:.4f} "
                  f"| test@bestVal={best_test:.4f}")

        if wait >= patience:
            print(f"Early stopping at epoch {ep}")
            break

    print(f"\nFinal: best Val AUC={best_val:.4f} | Test AUC @ best Val={best_test:.4f}")
    return best_val, best_test, train_losses, val_losses


# ==== ONE-RUN ROC PLOTS (append below your existing code) =====================
def _collect_probs_labels(model, loader, device):
    import numpy as np, torch
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.float().to(device)
            logits = model(xb)
            prob1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            ys.append(yb.numpy()); ps.append(prob1)
    if not ys:
        return np.array([]), np.array([])
    y = np.concatenate(ys); p = np.concatenate(ps)
    m = np.isfinite(y) & np.isfinite(p)
    return y[m], p[m]

def plot_roc_curves_one_run(cfg_path="configs/lstm_carter_script.yaml", seed=1337,
                            val_png="baseline_val_roc.png", test_png="baseline_test_roc.png"):
    import yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
    from sklearn.metrics import roc_curve, roc_auc_score

    # ---- rebuild the same data/model pipeline as run_baseline, but keep artifacts ----
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path   = cfg["data_path"]
    target_col  = cfg["target"]
    train_sz    = int(cfg["train_sz"])
    seq_len     = int(cfg["seq_len"])
    batch_sz    = int(cfg["batch_sz"])
    hidden      = int(cfg["lstm_hd_sz"])
    layers      = int(cfg["lstm_num_lyrs"])
    dropout     = float(cfg["dropout"])
    lr          = float(cfg["learning_rate"])
    bloom_pen   = float(cfg["bloom_pen"])
    bloom_thr   = float(cfg["bloom_thresh"])
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seeds
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # data (impute with train-only means; standardize with train-only stats)
    df = pd.read_csv(data_path)
    num = df.select_dtypes(include=[np.number]).copy().replace([np.inf, -np.inf], np.nan)
    mu_imp = num.iloc[:train_sz].mean(skipna=True)
    num = num.fillna(mu_imp)

    y_raw = num[target_col].to_numpy(np.float32)
    X_np  = num.drop(columns=[target_col]).to_numpy(np.float32)
    mu = X_np[:train_sz].mean(axis=0)
    sd = X_np[:train_sz].std(axis=0); sd[sd == 0] = 1.0
    X_np = (X_np - mu) / sd
    y_bin = (y_raw >= bloom_thr).astype(np.int64)

    X = torch.from_numpy(X_np); y = torch.from_numpy(y_bin)

    # datasets/loaders (interleave Val/Test over post-train)
    train_ds = SequenceDataset(X[:train_sz], y[:train_sz], seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    post_start = train_sz + seq_len
    post_ds = SequenceDataset(X[post_start:], y[post_start:], seq_len)
    n_post = len(post_ds)
    val_idx  = list(range(0, n_post, 2))
    test_idx = list(range(1, n_post, 2))
    val_loader  = DataLoader(IndexedDataset(post_ds, val_idx),  batch_size=batch_sz, shuffle=False)
    test_loader = DataLoader(IndexedDataset(post_ds, test_idx), batch_size=batch_sz, shuffle=False)

    # model
    input_size = X.shape[1]
    model = BaselineLSTM(input_size, hidden, layers, dropout).to(device)
    class_w = torch.tensor([1.0, bloom_pen], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # quick train (shorter epochs just for demo plots)
    best_val, patience, wait = -1.0, 50, 0
    for ep in range(1, 501):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.float().to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # early stop on val AUC
        from sklearn.metrics import roc_auc_score
        v_y, v_p = _collect_probs_labels(model, val_loader, device)
        if v_y.size > 0 and len(np.unique(v_y)) > 1:
            v_auc = float(roc_auc_score(v_y, v_p))
        else:
            v_auc = float("nan")

        if v_auc > best_val:
            best_val = v_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    # load best snapshot
    if 'best_state' in locals():
        model.load_state_dict(best_state, strict=False)

    # collect probs/labels
    v_y, v_p = _collect_probs_labels(model, val_loader, device)
    t_y, t_p = _collect_probs_labels(model, test_loader, device)

    # plot helper
    def _plot_and_save(y, p, out_path, title):
        import matplotlib.pyplot as plt
        if y.size == 0 or len(np.unique(y)) < 2:
            print(f"[warn] insufficient positives/negatives to plot ROC: {title}")
            return
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
        plt.plot([0,1], [0,1], '--', lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(title); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"saved {out_path}")

    # make plots
    _plot_and_save(v_y, v_p, val_png,  "Baseline LSTM — Validation ROC")
    _plot_and_save(t_y, t_p, test_png, "Baseline LSTM — Test ROC")

# Example: run one ROC plot generation (comment out if not needed)
# plot_roc_curves_one_run(seed=2025)



if __name__ == "__main__":
    plot_roc_curves_one_run(seed=2028)
    # import csv, os, random, numpy as np, torch
    #
    # N_RUNS = 100
    # OUT_CSV = "baseline_lstm_runs.csv"
    #
    # # make results list and write header
    # rows = []
    # header = ["run", "seed", "best_val_auc", "test_auc_at_best_val"]
    #
    # # choose distinct seeds (but reproducible)
    # base_seed = 1337
    # seeds = [base_seed + i for i in range(N_RUNS)]
    #
    # # run loop
    # for i, seed in enumerate(seeds, start=1):
    #     print(f"\n===== run {i}/{N_RUNS} | seed={seed} =====")
    #
    #     # set seeds for reproducibility
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)
    #
    #     # train once
    #     best_val, test_at_best, train_losses, val_losses = run_baseline()
    #
    #     # store
    #     rows.append([i, seed, best_val, test_at_best])
    #
    # # write CSV
    # with open(OUT_CSV, "w", newline="") as f:
    #     w = csv.writer(f)
    #     w.writerow(header)
    #     w.writerows(rows)
    #
    #     # summary row (mean ± std)
    #     vals = np.array([r[2] for r in rows], dtype=float)
    #     tests = np.array([r[3] for r in rows], dtype=float)
    #     w.writerow([])
    #     w.writerow(["summary_mean", "", np.nanmean(vals), np.nanmean(tests)])
    #     w.writerow(["summary_std",  "", np.nanstd(vals, ddof=1), np.nanstd(tests, ddof=1)])
    #
    # print(f"\n✅ saved results to {OUT_CSV}")
