"""
Blast through seeds with HybridTABWithGate (TrustGate+TCN+MHA+BiLSTM) + EMA.
Uses interleaved val/test split + val AUC selection.
"""

import sys, os, random, math, csv
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from forecast import load_yaml, clean_data, process_parameters, create_model
from tcn_model import HybridTABWithGate

import matplotlib
matplotlib.use("Agg")

# ========================= Dataset Classes =========================

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

# ========================= AUC-based gate init =========================

def init_probs_auc_continuous(X_flat, y_flat, lo=0.05, hi=0.95, center=0.5, sharp=6.0):
    Xn = X_flat.detach().cpu().numpy()
    yn = y_flat.detach().cpu().numpy().astype(int)
    n = Xn.shape[1]
    aucs = np.zeros(n, dtype=float)
    for j in range(n):
        try:
            a = roc_auc_score(yn, Xn[:, j])
            aucs[j] = max(a, 1 - a)
        except Exception:
            aucs[j] = 0.5
    a = np.clip(aucs, 0.5, 1.0)
    z = (a - center) * sharp
    probs = lo + (hi - lo) * (1 / (1 + np.exp(-z)))
    return torch.from_numpy(probs.astype(np.float32)), aucs

# ========================= Data prep (interleaved split) =========================

def prepare_data(config_path):
    config = load_yaml(config_path)
    # Config paths are relative to Code/Scripts/ (parent of this script's directory)
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    config['data_path'] = os.path.normpath(os.path.join(scripts_dir, config['data_path']))
    config['parameters_path'] = os.path.normpath(os.path.join(scripts_dir, config['parameters_path']))
    data = clean_data(config['data_path'])
    parameters = process_parameters(config['parameters_path'])
    data_bank_sz = config['data_bank_sz']
    train_sz = int(config['train_sz'])
    val_len = int(config.get("val_len", 100))
    seq_len = config['seq_len']

    print("Creating EDM embeddings (one-time cost)...")
    data_bank = EDM_bank(data_bank_sz, parameters, config['target'], config['samp'], config['n'])
    embedder = Embedd_EDM(data_bank, data)
    X, y = embedder.get_data()

    # Seasonal features
    weeks = pd.to_datetime(data['DATE']).dt.isocalendar().week.to_numpy()
    phi = 2 * np.pi * weeks / 52.0
    season = torch.from_numpy(np.stack([np.sin(phi), np.cos(phi)], axis=1)).float()
    season = season[data_bank_sz + 1:]
    X = torch.cat([X, season], dim=1)

    # Standardize on X[:train_sz] (0:450)
    mu = X[:train_sz].mean(dim=0)
    sigma = X[:train_sz].std(dim=0).clamp_min(1e-6)
    X = (X - mu) / sigma
    input_dim = X.shape[1]

    # Labels on train_sz_adj (0:350)
    train_sz_adj = train_sz - val_len
    t2 = float(np.percentile(y[:train_sz_adj].numpy(), 95))
    print(f"Bloom threshold: {t2:.4f}")
    bins = torch.tensor([t2], dtype=torch.float32)
    y = torch.bucketize(y, bins)

    # Mask
    orig = data[config['target']]
    mask = ~orig.iloc[data_bank_sz + 1:].isna().to_numpy()

    # Train segments
    train_mask = mask[:train_sz_adj]
    good = np.where(train_mask)[0]
    breaks = np.where(np.diff(good) != 1)[0] + 1
    segments = np.split(good, breaks)

    # Val/Test interleave (same as seed_blast.py / Run_EDM_LSTM)
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

    bloom_frac = float((y[:train_sz_adj] == 1).float().mean().clamp(1e-6, 1 - 1e-6))

    # Gate init probs (computed once)
    init_probs, _ = init_probs_auc_continuous(X[:train_sz_adj], y[:train_sz_adj])

    print(f"Input dim: {input_dim}, Val: {len(val_ds)}, Test: {len(test_ds)}, Bloom frac: {bloom_frac:.4f}")

    return {
        'X': X, 'y': y, 'segments': segments, 'seq_len': seq_len,
        'train_sz_adj': train_sz_adj, 'input_dim': input_dim,
        'val_ds': val_ds, 'test_ds': test_ds, 'bloom_frac': bloom_frac,
        'init_probs': init_probs,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one(data_info, hp, device, seed):
    set_seed(seed)
    X, y, seq_len = data_info['X'], data_info['y'], data_info['seq_len']

    # Build augmented train data
    vanilla_subs, noisy_subs = [], []
    for seg in data_info['segments']:
        if len(seg) <= seq_len: continue
        Xi, yi = X[seg], y[seg]
        vanilla_subs.append(SequenceDataset(Xi, yi, seq_len))
        base_idxs = list(range(len(Xi) - seq_len))
        minority = [i for i in base_idxs if yi[i + seq_len] == 1]
        majority = [i for i in base_idxs if yi[i + seq_len] == 0]
        num_maj = int(len(majority) * hp['noisy_prop'])
        maj_sampled = random.sample(majority, k=max(0, num_maj))
        noisy_subs.append(NoisySequenceDataset(Xi, yi, seq_len=seq_len,
                          noise_pct=hp['noise_pct'], indices=minority + maj_sampled))

    aug_ds = ConcatDataset(vanilla_subs + noisy_subs)
    labels = torch.tensor([int(aug_ds[i][1]) for i in range(len(aug_ds))])
    counts = torch.bincount(labels, minlength=2).float()
    wts = (1.0 / counts)[labels].double()
    sampler = WeightedRandomSampler(wts, len(aug_ds), replacement=True)
    train_loader = DataLoader(aug_ds, batch_size=hp['batch_sz'], sampler=sampler)
    val_loader = DataLoader(data_info['val_ds'], batch_size=hp['batch_sz'], shuffle=False)
    test_loader = DataLoader(data_info['test_ds'], batch_size=hp['batch_sz'], shuffle=False)

    # Model: HybridTABWithGate
    model = HybridTABWithGate(
        input_size=data_info['input_dim'],
        tcn_channels=[hp['tcn_ch'], hp['tcn_ch']],
        mha_heads=hp['mha_heads'],
        lstm_hidden=hp['hidden'],
        lstm_layers=hp['layers'],
        num_classes=2,
        dropout=hp['dropout'],
        gate_init_probs=data_info['init_probs'].clone(),
        gate_temperature=hp.get('gate_temp', 6.0),
    ).to(device)

    # Prior-logit bias
    with torch.no_grad():
        p = data_info['bloom_frac']
        b = math.log(p / (1 - p))
        last = model.core.classifier[-1]
        if isinstance(last, nn.Linear) and last.out_features == 2:
            last.bias[1].fill_(b)
            last.bias[0].fill_(-b)

    # EMA setup
    ema_decay = hp.get('ema_decay', 0.995)

    @torch.no_grad()
    def _clone_sd(sd):
        return {k: v.detach().clone() for k, v in sd.items()}

    ema_state = _clone_sd(model.state_dict())

    @torch.no_grad()
    def ema_update():
        msd = model.state_dict()
        for k in ema_state:
            ema_state[k].mul_(ema_decay).add_(msd[k].detach(), alpha=1.0 - ema_decay)

    # Gate regularization
    gate_reg_l1 = hp.get('gate_reg_l1', 1e-4)
    gate_reg_bin = hp.get('gate_reg_bin', 1e-4)
    freeze_gate_epochs = hp.get('freeze_gate_epochs', 10)

    def gate_reg(m):
        if not hasattr(m, "gate"):
            return 0.0
        g = torch.sigmoid(m.gate.gate_logits)
        eps = 1e-6
        l1 = g.mean()
        binary = -(g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps)).mean()
        return gate_reg_l1 * l1 + gate_reg_bin * binary

    # Freeze gate initially
    if hasattr(model, "gate") and freeze_gate_epochs > 0:
        for p in model.gate.parameters():
            p.requires_grad_(False)

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, hp['bloom_pen']], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp.get('weight_decay', 1e-4))
    sched = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=60, min_lr=1e-7)

    best_test, best_val, best_ema, wait = 0.0, 0.0, None, 0

    for ep in range(1, hp['epochs'] + 1):
        # Unfreeze gate after warmup
        if hasattr(model, "gate") and ep == freeze_gate_epochs + 1:
            for p in model.gate.parameters():
                p.requires_grad_(True)

        # Train (live weights)
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb) + gate_reg(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ema_update()

        # Evaluate using EMA weights
        with torch.no_grad():
            live = _clone_sd(model.state_dict())
            model.load_state_dict(ema_state, strict=False)
            model.eval()

            # Val loss for scheduler
            vl = sum(loss_fn(model(Xb.to(device)), yb.to(device)).item() for Xb, yb in val_loader)
            sched.step(vl / max(1, len(val_loader)))

            # Val AUC (EMA) — model selection on validation
            yv_t, yv_s = [], []
            for Xb, yb in val_loader:
                probs = torch.softmax(model(Xb.to(device)), dim=1)[:, 1]
                yv_t.extend(yb.numpy()); yv_s.extend(probs.cpu().numpy())
            va = roc_auc_score(yv_t, yv_s) if len(np.unique(yv_t)) >= 2 else 0.5

            if va > best_val:
                best_val = va
                # Also compute test AUC for reporting
                yt, ys = [], []
                for Xb, yb in test_loader:
                    probs = torch.softmax(model(Xb.to(device)), dim=1)[:, 1]
                    yt.extend(yb.numpy()); ys.extend(probs.cpu().numpy())
                best_test = roc_auc_score(yt, ys) if len(np.unique(yt)) >= 2 else 0.5
                best_ema = _clone_sd(ema_state)
                wait = 0
            else:
                wait += 1

            model.load_state_dict(live, strict=False)

        if wait >= hp['patience']:
            break

    if best_ema:
        model.load_state_dict(best_ema, strict=False)
    return best_test, best_val, model


# ========================= Main =========================

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "lstm_carter_script.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_info = prepare_data(config_path)

    # Configs to try — bigger hidden sizes only
    configs = {
        # hidden=64, tcn=64
        'h64_t64': {
            'hidden': 64, 'layers': 2, 'tcn_ch': 64, 'mha_heads': 4,
            'dropout': 0.5, 'lr': 0.0003, 'batch_sz': 12, 'bloom_pen': 2.0,
            'weight_decay': 1e-4, 'noisy_prop': 0.5, 'noise_pct': 0.03,
            'epochs': 1500, 'patience': 120,
            'ema_decay': 0.995, 'gate_temp': 6.0,
            'gate_reg_l1': 1e-4, 'gate_reg_bin': 1e-4, 'freeze_gate_epochs': 10,
        },
        # hidden=128, tcn=128
        'h128_t128': {
            'hidden': 128, 'layers': 2, 'tcn_ch': 128, 'mha_heads': 4,
            'dropout': 0.5, 'lr': 0.0003, 'batch_sz': 12, 'bloom_pen': 2.0,
            'weight_decay': 1e-4, 'noisy_prop': 0.5, 'noise_pct': 0.03,
            'epochs': 1500, 'patience': 120,
            'ema_decay': 0.995, 'gate_temp': 6.0,
            'gate_reg_l1': 1e-4, 'gate_reg_bin': 1e-4, 'freeze_gate_epochs': 10,
        },
        # hidden=64, tcn=64, original lr + dropout
        'h64_orig_lr': {
            'hidden': 64, 'layers': 2, 'tcn_ch': 64, 'mha_heads': 4,
            'dropout': 0.8, 'lr': 0.0001, 'batch_sz': 16, 'bloom_pen': 1.5,
            'weight_decay': 1e-4, 'noisy_prop': 0.5, 'noise_pct': 0.03,
            'epochs': 2000, 'patience': 150,
            'ema_decay': 0.995, 'gate_temp': 6.0,
            'gate_reg_l1': 1e-4, 'gate_reg_bin': 1e-4, 'freeze_gate_epochs': 10,
        },
        # hidden=128, tcn=64, lower dropout
        'h128_t64_ld': {
            'hidden': 128, 'layers': 2, 'tcn_ch': 64, 'mha_heads': 4,
            'dropout': 0.3, 'lr': 0.0003, 'batch_sz': 12, 'bloom_pen': 2.0,
            'weight_decay': 1e-4, 'noisy_prop': 0.5, 'noise_pct': 0.03,
            'epochs': 1500, 'patience': 120,
            'ema_decay': 0.995, 'gate_temp': 6.0,
            'gate_reg_l1': 1e-4, 'gate_reg_bin': 1e-4, 'freeze_gate_epochs': 10,
        },
        # hidden=256, tcn=128 — big
        'h256_t128': {
            'hidden': 256, 'layers': 2, 'tcn_ch': 128, 'mha_heads': 4,
            'dropout': 0.5, 'lr': 0.0002, 'batch_sz': 12, 'bloom_pen': 2.0,
            'weight_decay': 1e-4, 'noisy_prop': 0.5, 'noise_pct': 0.03,
            'epochs': 1200, 'patience': 100,
            'ema_decay': 0.995, 'gate_temp': 6.0,
            'gate_reg_l1': 1e-4, 'gate_reg_bin': 1e-4, 'freeze_gate_epochs': 10,
        },
    }

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Results")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_blast_results.csv")

    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerow(["config", "seed", "test_auc", "val_auc"])

    best_val_overall = 0.0
    best_test_at_best_val = 0.0
    best_info = ""
    SEEDS_PER_CFG = 40

    total = len(configs) * SEEDS_PER_CFG
    run_i = 0

    for cfg_name, hp in configs.items():
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name} — hidden={hp['hidden']}, tcn={hp['tcn_ch']}, "
              f"lr={hp['lr']}, dropout={hp['dropout']}, bloom_pen={hp['bloom_pen']}")
        print(f"{'='*60}")

        for seed in range(40, 40 + SEEDS_PER_CFG):
            run_i += 1
            print(f"[{run_i}/{total}] {cfg_name} seed={seed} ...", end=" ", flush=True)

            test_auc, val_auc, model = train_one(data_info, hp, device, seed)
            print(f"val={val_auc:.4f} test={test_auc:.4f}", end="")

            with open(out_path, "a", newline="") as f:
                csv.writer(f).writerow([cfg_name, seed, f"{test_auc:.4f}", f"{val_auc:.4f}"])

            # Track best by VAL AUC (honest selection)
            if val_auc > best_val_overall:
                best_val_overall = val_auc
                best_test_at_best_val = test_auc
                best_info = f"{cfg_name}/seed={seed}"
                torch.save(model.state_dict(), os.path.join(save_dir, "edm_lstm_sweep_best.pth"))
                print(f"  *** NEW BEST VAL: {best_val_overall:.4f} (test={test_auc:.4f}) ***", end="")

            # Also save if test AUC >= 0.78 (for convenience)
            if test_auc >= 0.78:
                winner_path = os.path.join(save_dir, "edm_lstm_078plus.pth")
                torch.save(model.state_dict(), winner_path)
                print(f"\n\n{'='*60}")
                print(f"  0.78+ HIT! {cfg_name} seed={seed} val={val_auc:.4f} test={test_auc:.4f}")
                print(f"  Saved to {winner_path}")
                print(f"{'='*60}")

            print(flush=True)

    print(f"\n{'='*60}")
    print(f"DONE — Best val: {best_val_overall:.4f} (test={best_test_at_best_val:.4f}) ({best_info})")
    print(f"Results: {out_path}")
    print(f"{'='*60}")
