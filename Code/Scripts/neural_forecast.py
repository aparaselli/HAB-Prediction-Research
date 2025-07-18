from forecast import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
from torch.utils.data import ConcatDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tcn_model import *
from losses import *
#from batchdilate import DTWShpTime


class EDM_bank():
    '''
    lib_sz - library size
    params - edm model embedding specifications
    target - target variable
    samp - sampling from n*samp top models
    n - number of models in the ensemble
    '''
    def __init__(self, lib_sz,params,target,samp,n):
        self.lib_sz = lib_sz
        self.params = params
        self.target = target
        self.samp = samp
        self.n = n

        
    # Must take in a dataframe
    # Returns tensor of embeddings
    def forecast(self, x):
        x
        lib = '1 ' + str(self.lib_sz) 
        pred = '' + str(self.lib_sz + 1) + ' ' + str(x.shape[0])
        parameters_df = create_model(x,self.params,self.target,self.samp,lib,pred,ensemble_sz=self.n)
        #print(parameters_df.shape[0])
        #parameters_df = parameters_df.iloc[0:self.n*self.samp:self.samp]#.sample(n) DEPRECIATED
        #print(parameters_df.shape[0])
        model_preds = np.stack(
            [np.asarray(parameters_df["pred"].iloc[i][1:-1]) for i in range(self.n)]
        )
        timestep_preds = model_preds.T                      

        return torch.as_tensor(timestep_preds, dtype=torch.float32)
    
    def __call__(self, x):
        return self.forecast(x)



class HybridNeuralNetwork(nn.Module):
    def __init__(self, n, hd_sz, drop=0.5):
        super().__init__()
        self.ffw1    = nn.Linear(n, hd_sz)
        self.bn1     = nn.BatchNorm1d(hd_sz)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.ffw2    = nn.Linear(hd_sz, 1)

    def forward(self, x):
        x = self.ffw1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ffw2(x)
        return x
    
class HybridNNCLS(nn.Module):
    def __init__(self, n, hd_sz, drop=0.5):
        super().__init__()
        self.ffw1    = nn.Linear(n, hd_sz)
        self.bn1     = nn.BatchNorm1d(hd_sz)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.ffw2    = nn.Linear(hd_sz, 4)

    def forward(self, x):
        x = self.ffw1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.ffw2(x)
        return x
    



class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,    # = n, your ensemble size
        hidden_size: int,    # e.g. 128
        num_layers: int,    # e.g. 1 or 2
        num_classes: int,   # 4 for no/slight/mod/strong
        dropout: float=0.5,
        batch_first: bool=True
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers>1 else 0.0,
            batch_first=batch_first
        )
        # you can do many‐to‐one: only look at the last hidden output
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        returns logits of shape (batch, num_classes)
        """
        # out: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x)
        # grab last time‐step
        last = out[:, -1, :]               # (batch, hidden_size)
        return self.classifier(last)       # (batch, num_classes)
    


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,      # = your ensemble size (n)
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        # Normalize the concatenated hidden states
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # A small classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: logits of shape (batch, num_classes)
        """
        # LSTM output: (batch, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        # Take the last timestep
        last_step = lstm_out[:, -1, :]           # (batch, hidden_size * 2)
        # Layer-norm then head
        out = self.layer_norm(last_step)
        logits = self.classifier(out)             # (batch, num_classes)
        return logits

    

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    loss_func=torch.nn.CrossEntropyLoss(),  # for classification or BCEWithLogitsLoss
    device="cpu",
    patience=10,
):
    """
    Train a PyTorch model with support for both CrossEntropyLoss and BCEWithLogitsLoss.

    If using BCEWithLogitsLoss, assumes a two-logit model and uses the positive-class logit.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=200,
        min_lr=1e-7,
        verbose=True
    )
    best_val = float("inf")
    best_auc = 0.0 
    wait = 0

    train_losses = []
    val_losses = []
    best_state = None

    for ep in range(1, epochs + 1):
        # ---- TRAIN STEP ----
        model.train()
        running_train = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            # choose loss behavior
            if isinstance(loss_func, torch.nn.BCEWithLogitsLoss):
                pos_logit = logits[:, 1]
                loss = loss_func(pos_logit, yb.float())
            else:
                loss = loss_func(logits, yb)
            loss.backward()
            optimizer.step()
            running_train += loss.item()
        train_loss = running_train / len(train_loader)
        train_losses.append(train_loss)

        # ---- VALIDATION STEP ----
        model.eval()
        running_val = 0.0
        all_trues, all_preds, all_scores = [], [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                if isinstance(loss_func, torch.nn.BCEWithLogitsLoss):
                    pos_logit = logits[:, 1]
                    loss = loss_func(pos_logit, yb.float())
                    running_val += loss.item()
                    probs = torch.sigmoid(pos_logit)
                    preds = (probs > 0.5).long()
                else:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    loss = loss_func(logits, yb)
                    running_val += loss.item()
                    preds = torch.argmax(logits, dim=1)
                all_trues.append(yb.cpu())
                all_preds.append(preds.cpu())
                all_scores.append(probs.cpu().numpy())

        val_loss = running_val / len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # compute balanced accuracy once per epoch
        y_true  = np.concatenate(all_trues)
        y_score = np.concatenate(all_scores)
        y_pred = torch.cat(all_preds).numpy()
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score)

        if ep % 50 == 0:
            print(f"Epoch {ep:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  bal_acc={bal_acc:.4f}  auc={auc:.4f}")

        # ---- EARLY STOPPING ----
        #if val_loss < best_val:
            #best_val = val_loss
        if auc > best_auc:
            best_auc   = auc
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping at epoch {ep}")
                model.load_state_dict(best_state)
                break

    # ---- PLOTTING & CLEANUP ----
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.tight_layout()
    plt.show()

    if best_state is not None:
        print(f"Loading best model (AUC={best_auc:.4f})")
        model.load_state_dict(best_state)
    else:
        print("Best model not loaded")
    return model

        


class Embedd_EDM():
    def __init__(self, EDM, data, type):
        self.EDM_bank = EDM
        self.target = EDM.target
        y_raw = torch.tensor(
            data[self.EDM_bank.target]
               .iloc[self.EDM_bank.lib_sz+1:]
               .to_numpy(),
            dtype=torch.float32
        )
        if type == 'regression':
            print('Logging target variable')
            self.targets = torch.log1p(y_raw)
        else:
            self.targets = y_raw
        self.embedded_inputs = self.EDM_bank(data)

    def len(self):
        return len(self.embedded_inputs)

    def get_data(self):
        return self.embedded_inputs, self.targets
    

class SequenceDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, seq_len: int):
        """
        X: (T, n)      full time‐series of EDM embeddings
        y: (T,)        corresponding labels
        seq_len: int   how many past steps to look at
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        # you can make a sample for each time you have a full window + next target
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # inputs: the window from idx → idx+seq_len (exclusive)
        seq = self.X[idx : idx + self.seq_len]           # (seq_len, n)
        target = self.y[idx + self.seq_len]              # scalar
        return seq, target



class NoisySequenceDataset(Dataset):
    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 seq_len: int,
                 noise_pct: float = 0.02,
                 indices: list = None):
        """
        X:         (T, F) tensor of features
        y:         (T,)  tensor of targets
        seq_len:   how many past steps to look at
        noise_pct: fraction of the data‐range to use as noise sigma
        indices:   optional list of starting positions to sample
        """
        assert len(X) == len(y), "X and y must be same length"
        self.X         = X
        self.y         = y
        self.seq_len   = seq_len
        self.noise_pct = noise_pct

        # Precompute per-feature ranges
        data_min = X.min(dim=0).values    # (F,)
        data_max = X.max(dim=0).values    # (F,)
        self.range = data_max - data_min  # (F,)

        # Determine which indices to use
        max_idx = len(X) - seq_len
        if indices is None:
            # default: all valid start positions
            self.indices = list(range(max_idx))
        else:
            # filter out-of-bounds
            self.indices = [i for i in indices if 0 <= i < max_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        window = self.X[base_idx : base_idx + self.seq_len]  # (seq_len, F)
        target = self.y[base_idx + self.seq_len]

        # generate noise: same shape as window
        noise_std = self.range * self.noise_pct
        noise = torch.randn_like(window) * noise_std
        noisy_window = window + noise

        return noisy_window, target

    



def eval_confusion(model, loader, device="cpu", labels=None, title="Confusion Matrix"):
    model.eval()
    all_trues = []
    all_preds = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            preds  = torch.argmax(logits, dim=1)
            all_trues.append(yb.cpu())
            all_preds.append(preds.cpu())

    y_true = torch.cat(all_trues).numpy()
    y_pred = torch.cat(all_preds).numpy()

    # Ensure the confusion matrix includes all labels even if unused
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(title)
    plt.show()
    return cm


def plot_predictions(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    target_name: str,
    split_name: str,
    save_path: str = None
):
    """
    Plots true vs. predicted values for a given dataset split
    and optionally saves the figure to disk.

    Args:
        model:        trained PyTorch model
        X:            input features (n_samples × …)
        y:            true targets (n_samples,)
        target_name:  name of the target variable (for axis label)
        split_name:   e.g. "Train" or "Test" (for title/legend)
        save_path:    path (including filename) to save the plot.
                      If None, the plot is shown instead.
    """
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().cpu().numpy()
    true = y.cpu().numpy()
    steps = range(len(true))

    plt.figure(figsize=(10, 4))
    plt.plot(steps, true,  label=f"{split_name} True")
    plt.plot(steps, preds, label=f"{split_name} Predicted")
    plt.xlabel("Timestep")
    plt.ylabel(target_name)
    plt.title(f"{split_name}: True vs Predicted")
    plt.legend()
    plt.tight_layout()

    if save_path:
        # ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {split_name} plot to {save_path}")
    else:
        plt.show()

def plot_roc_with_thresholds(model, loader, device="cpu", pos_label=1, n_labels=20):
    model.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            if logits.dim()==1 or logits.size(1)==1:
                probs = torch.sigmoid(logits.view(-1))
            else:
                probs = torch.softmax(logits, dim=1)[:, pos_label]
            y_true.extend(yb.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],"--", label="Chance")

    # pick a few threshold indices to annotate
    idxs = np.linspace(0, len(thresholds)-1, n_labels, dtype=int)
    for i in idxs:
        thr = thresholds[i]
        plt.scatter(fpr[i], tpr[i], s=30, edgecolor="k")
        plt.text(
            fpr[i], tpr[i],
            f"{thr:.2f}",
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right"
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve with Thresholds")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()




def main():
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_yaml(args.config)
    print("YAML Contents:")
    for key, value in config.items():
        print(f"{key}: {value}")

    #set up data and edm    
    data = clean_data(config['data_path'])
    data_bank_sz = config['data_bank_sz']
    train_sz = config['train_sz']

    parameters = process_parameters(config['parameters_path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Creating EDM Models and Embedding Data')
    data_bank = EDM_bank(data_bank_sz,parameters,config['target'],config['samp'],config['n'])
    embedder = Embedd_EDM(data_bank, data,config['forecast_type'])

    orig = data[config['target']]
    mask = ~orig.iloc[ config['data_bank_sz']+1 : ].isna().to_numpy() #Exlude NAN values from train and test data
    X,y = embedder.get_data()
    print(f'Pre NAN mask X size{X.shape}')
    print(f'Pre NAN mask X size{y.shape}')




    if config['forecast_type'] == 'regression':

        train_X, train_y = X[:train_sz], y[:train_sz]
        test_X,  test_y  = X[train_sz:], y[train_sz:]
        train_ds = TensorDataset(train_X, train_y)
        test_ds  = TensorDataset(test_X,  test_y)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

        model = HybridNeuralNetwork(config['n'],1000)
        print('Training model')
        train_model(model, train_loader, epochs=500,lr=1e-5)
        print("Saving plots…")
        plot_predictions(
            model, train_X, train_y,
            target_name=config['target'],
            split_name="Train",
            save_path="plots/train_true_vs_pred.png"
        )
        plot_predictions(
            model, test_X, test_y,
            target_name=config['target'],
            split_name="Test",
            save_path="plots/test_true_vs_pred.png"
        )
    elif config['forecast_type'] == 'lstm':
        t2 = np.percentile(data[config['target']].iloc[:data_bank_sz],95) #V
        print(f"Bloom threshold is {t2}")
        bins = torch.tensor([t2], dtype=torch.float32)# for 2 class
        y = torch.bucketize(y, bins) 

        train_X, train_y = X[:train_sz], y[:train_sz]
        test_X,  test_y  = X[train_sz:], y[train_sz:]

        seq_len = config['seq_len']  # How many steps before will the LSTM use to make a decision

        # 1. find your good (non‐NaN) indices
        train_mask = mask[:train_sz] 
        good = np.where(train_mask)[0]
        breaks = np.where(np.diff(good) != 1)[0] + 1
        segments = np.split(good, breaks)

        vanilla_subs = []
        noisy_subs   = []
        maj_frac     = config['noisy_prop']   # same 50% majority sampling
        noise_pct    = config['noise_pct']  # same noise level

        for seg in segments:
            # only consider segments long enough to form at least one seq_len window
            if len(seg) <= seq_len:
                continue

            # extract the X,y for this continuous run
            Xi = X[seg]  # shape (T_seg, F)
            yi = y[seg]  # shape (T_seg,)

            # ——— vanilla windows —————————————————————————————————————
            vanilla_subs.append( SequenceDataset(Xi, yi, seq_len) )

            # ——— minority / majority indices inside this segment —————————————————
            # valid start positions are 0 … len(Xi)-seq_len-1
            base_idxs = list(range(len(Xi) - seq_len))

            # pick by looking at the label at the end of each window
            minority_idxs = [i for i in base_idxs if yi[i + seq_len] == 1]
            majority_idxs = [i for i in base_idxs if yi[i + seq_len] == 0]

            # sample 50% of the majority windows
            num_maj_to_sample = int(len(majority_idxs) * maj_frac)
            maj_sampled = random.sample(majority_idxs, k=num_maj_to_sample)

            # combine minority + sampled majority
            selected_idxs = minority_idxs + maj_sampled

            # ——— noisy windows ——————————————————————————————————————
            noisy_subs.append(
                NoisySequenceDataset(
                    Xi,
                    yi,
                    seq_len=seq_len,
                    noise_pct=noise_pct,
                    indices=selected_idxs
                )
            )

        # 2. concatenate all of them
        augmented_ds = ConcatDataset(vanilla_subs + noisy_subs)
        print("Total vanilla training observations:", sum([len(ds) for ds in vanilla_subs]))
        print("Total noisy training observations:", sum([len(ds) for ds in noisy_subs]))
        print("Total training observations:", len(augmented_ds))

        # 1. carve off the test‐portion of the mask
        mask_test = mask[train_sz:]              # length = len(X) - train_sz

        # 2. find the “good” indices *within* that test slice
        #    these are offsets 0…len(mask_test)-1
        good_test_local = np.where(mask_test)[0]

        # 3. split into contiguous runs
        breaks = np.where(np.diff(good_test_local) != 1)[0] + 1
        segments_test = np.split(good_test_local, breaks)

        # 4. for each run, build a SequenceDataset
        val_subs = []
        for seg in segments_test:
            if len(seg) <= seq_len:
                continue

            # seg is local to the test slice, so convert back to global X/y indices
            seg_global = seg + train_sz

            Xi = X[seg_global]   # shape (T_seg, n)
            yi = y[seg_global]   # shape (T_seg,)

            val_subs.append( SequenceDataset(Xi, yi, seq_len) )

        # 5. concat them all
        val_ds = ConcatDataset(val_subs)

        # 6. make your DataLoader
        print("Total testing observations:", len(val_ds))
        
        flat_labels = []
        for ds in vanilla_subs:
            for i in range(len(ds)):
                _, lbl = ds[i]
                flat_labels.append(lbl)
        train_labels = torch.tensor(flat_labels)
        train_ds = ConcatDataset(vanilla_subs)
        assert len(train_labels) == len(train_ds)

        # NEW: define class_weights based on train_labels
        num_classes = 2
        counts = torch.bincount(train_labels, minlength=num_classes).float()
        inv_freq = 1.0 / counts
        class_weights = inv_freq / inv_freq.sum() * len(counts)

        # use class_weights to generate per-sample weights
        sample_weights = class_weights[train_labels]


        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(augmented_ds),  # total samples = vanilla + noisy
            replacement=True
        )
        batch_sz = config['batch_sz']
        orig_train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=False)
        train_loader = DataLoader(augmented_ds, batch_size=batch_sz, sampler=sampler)

        test_loader   = DataLoader(val_ds,   batch_size=batch_sz, shuffle=False)
        '''
        model = BiLSTMClassifier(
            input_size = config['n'],
            hidden_size = 16,
            num_layers  = 3,
            num_classes = 2,
            dropout     = 0.5
        ).to(device)
        '''
        channel_sz = config['tcn_channel']
        model = HybridTAB(
            input_size = config['n'],
            tcn_channels=[channel_sz,channel_sz],  #started with 64 x 64  
            mha_heads=config['atn_hds'],
            lstm_hidden=config['lstm_hd_sz'],
            lstm_layers=config['lstm_num_lyrs'],
            num_classes=2,           # or 4, depending on your task
            dropout=config['dropout']
        ).to(device)




        weights = torch.tensor([config['bloom_pen'], 1.0], device=device)
        loss_func = nn.CrossEntropyLoss(weight=weights)
        print('Training model')
        model.to(device)
        train_model(model, train_loader, test_loader, epochs=100000,lr=config['learning_rate'],loss_func=loss_func,device=device,patience=400)

        save_path = config["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # save only the model’s weights
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")


        #class_names = ["no bloom", "slight", "moderate", "strong"]
        class_names = ["no bloom", "bloom"]

        print("=== Train Confusion Matrix ===")
        cm_train = eval_confusion(model, train_loader, device, labels=class_names,
                                title="Train: True vs Predicted")

        print("=== Original Train Confusion Matrix ===")
        cm_orig = eval_confusion(
            model,
            orig_train_loader,
            device,
            labels=class_names,
            title="Original Train: True vs Predicted"
)


        print("=== Test Confusion Matrix ===")
        cm_test  = eval_confusion(model, test_loader,  device, labels=class_names,
                                title="Test: True vs Predicted")
        
        print("=== Test ROC ===")
        plot_roc_with_thresholds(model, test_loader, device=device, pos_label=1)
    
if __name__ == "__main__":
    main()