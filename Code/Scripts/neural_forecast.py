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
        parameters_df = parameters_df.iloc[0:self.n*self.samp:self.samp]#.sample(n)
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

    

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    loss_func=nn.MSELoss(),
    device="cpu",
    patience=10,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val = float("inf")
    wait = 0

    train_losses = []
    val_losses   = []
    best_state   = None

    for ep in range(1, epochs + 1):
        # ---- TRAIN STEP ----
        model.train()
        running_train = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_pred = model(Xb).squeeze()
            loss   = loss_func(y_pred, yb)
            loss.backward()
            optimizer.step()
            running_train += loss.item()
        train_loss = running_train / len(train_loader)
        train_losses.append(train_loss)

        # ---- VALIDATION STEP ----
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                y_pred = model(Xb).squeeze()
                loss   = loss_func(y_pred, yb)
                running_val += loss.item()
        val_loss = running_val / len(val_loader)
        val_losses.append(val_loss)
        if ep%50 == 0:
            print(f"Epoch {ep:3d}: train={train_loss:.4f}  val={val_loss:.4f}")

        # ---- EARLY STOPPING CHECK ----
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping at epoch {ep}")
                model.load_state_dict(best_state)
                break

    # ---- PLOTTING ----
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.tight_layout()
    plt.show()

    # ensure best weights are loaded
    if best_state is not None:
        model.load_state_dict(best_state)

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
                 noise_pct: float = 0.02):
        """
        X:         (T, F) tensor of features
        y:         (T,)  tensor of targets
        seq_len:   how many past steps to look at
        noise_pct: fraction of the data‐range to use as noise sigma
                   (e.g. 0.02 = 2% noise)
        """
        assert len(X) == len(y), "X and y must be same length"
        self.X        = X
        self.y        = y
        self.seq_len  = seq_len
        self.noise_pct = noise_pct

        # Precompute per-feature ranges so noise is scaled appropriately
        # (range = max – min for each feature across the whole series)
        data_min = X.min(dim=0).values    # (F,)
        data_max = X.max(dim=0).values    # (F,)
        self.range = data_max - data_min  # (F,)

    def __len__(self):
        # only indices where idx+seq_len < len(X) are valid
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # grab the clean window
        window = self.X[idx : idx + self.seq_len]       # (seq_len, F)
        target = self.y[idx + self.seq_len]             # scalar

        # generate noise: same shape as window
        # noise_std for each feature = noise_pct * range_of_that_feature
        noise_std = self.range * self.noise_pct         # (F,)
        # torch.randn gives (seq_len, F); multiply by std and add
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
            logits = model(Xb)                       # [batch, 4]
            preds  = torch.argmax(logits, dim=1)     # [batch]
            all_trues.append(yb.cpu())
            all_preds.append(preds.cpu())
    y_true = torch.cat(all_trues).numpy()
    y_pred = torch.cat(all_preds).numpy()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
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




def main():
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_yaml(args.config)
    print("YAML Contents:")
    for key, value in config.items():
        print(f"{key}: {value}")
    data = clean_data(config['data_path'])
    data_bank_sz = 800
    parameters = process_parameters(config['parameters_path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Creating EDM Models and Embedding Data')
    data_bank = EDM_bank(data_bank_sz,parameters,config['target'],config['samp'],config['n'])
    embedder = Embedd_EDM(data_bank, data,config['forecast_type'])
    X,y = embedder.get_data()
    print(X.shape)
    print(y.shape)
    if config['forecast_type'] == 'regression':

        train_X, train_y = X[:500], y[:500]
        test_X,  test_y  = X[500:], y[500:]
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
        t1 = np.percentile(data[config['target']].iloc[:data_bank_sz],92)
        t2 = np.percentile(data[config['target']].iloc[:data_bank_sz],95)
        t3 = np.percentile(data[config['target']].iloc[:data_bank_sz],98)
        bins = torch.tensor([t1, t2, t3], dtype=torch.float32)
        y = torch.bucketize(y, bins) 

        train_X, train_y = X[:600], y[:600]
        test_X,  test_y  = X[600:], y[600:]

        seq_len = 6  # for example, look at the last 20 timesteps
        train_ds = SequenceDataset(train_X, train_y, seq_len)
        noisy_ds = NoisySequenceDataset(X[:-1*test_X.size(0):8], y[:-1*test_y.size(0):8], seq_len=seq_len, noise_pct=0.03)
        augmented_ds = ConcatDataset([train_ds, noisy_ds])
        print("Total vanilla training observations:", len(train_ds))
        print("Total noisy training observations:", len(noisy_ds))
        print("Total training observations:", len(augmented_ds))
        val_ds   = SequenceDataset(test_X,  test_y,  seq_len)

        train_loader = DataLoader(augmented_ds, batch_size=16, shuffle=True)
        test_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)
        model = LSTMClassifier(
            input_size = config['n'],
            hidden_size = 16,
            num_layers  = 3,
            num_classes = 4,
            dropout     = 0.2
        ).to(device)

        num_classes = 4
        counts = torch.bincount(train_y, minlength=num_classes).float()
        inv_freq = 1.0 / counts
        class_weights = inv_freq / inv_freq.sum() * len(counts)          # normalize if you like

        loss_func = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print('Training model')
        model.to(device)
        train_model(model, train_loader, test_loader, epochs=10000,lr=1e-6,loss_func=loss_func,device=device,patience=400)


        class_names = ["no bloom", "slight", "moderate", "strong"]

        print("=== Train Confusion Matrix ===")
        cm_train = eval_confusion(model, train_loader, device, labels=class_names,
                                title="Train: True vs Predicted")

        print("=== Test Confusion Matrix ===")
        cm_test  = eval_confusion(model, test_loader,  device, labels=class_names,
                                title="Test: True vs Predicted")
    else:
        t1 = np.percentile(data[config['target']].iloc[:data_bank_sz],92)
        t2 = np.percentile(data[config['target']].iloc[:data_bank_sz],95)
        t3 = np.percentile(data[config['target']].iloc[:data_bank_sz],98)
        bins = torch.tensor([t1, t2, t3], dtype=torch.float32)
        y = torch.bucketize(y, bins) 
        train_X, train_y = X[:500], y[:500]
        test_X,  test_y  = X[500:], y[500:]
        train_ds = TensorDataset(train_X, train_y)
        test_ds  = TensorDataset(test_X,  test_y)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)
        model = HybridNNCLS(config['n'],128,0.5)

        num_classes = 4
        counts = torch.bincount(train_y, minlength=num_classes).float()
        inv_freq = 1.0 / counts
        class_weights = inv_freq / inv_freq.sum() * len(counts)          # normalize if you like

        loss_func = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print('Training model')
        model.to(device)
        train_model(model, train_loader, test_loader, epochs=1000,lr=1e-6,loss_func=loss_func,device=device,patience=1000)


        class_names = ["no bloom", "slight", "moderate", "strong"]

        print("=== Train Confusion Matrix ===")
        cm_train = eval_confusion(model, train_loader, device, labels=class_names,
                                title="Train: True vs Predicted")

        print("=== Test Confusion Matrix ===")
        cm_test  = eval_confusion(model, test_loader,  device, labels=class_names,
                                title="Test: True vs Predicted")

if __name__ == "__main__":
    main()