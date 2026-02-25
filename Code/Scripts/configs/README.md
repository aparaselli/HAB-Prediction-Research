# Configuration Parameters

This directory contains YAML configuration files for HAB prediction experiments.

## Example: lstm_carter_script.yaml

```yaml
# Data Configuration
data_path: "../Data/d1carter_data_w_gaps.csv"  # Path to CSV data file
target: "Lingulodinium_polyedra"               # Target species column name

# EDM Configuration
EDM_embed: true                                # Use EDM embeddings (false for Pure LSTM)
data_bank_sz: 200                             # EDM library size
parameters_path: "../Results/1000_sample_biolum.csv"  # Pre-computed EDM parameters
samp: 1                                        # EDM sample parameter
n: 300                                         # Number of EDM ensemble models

# Training Configuration
train_sz: 450                                  # Training set size
val_len: 100                                   # Validation set size (carved from train_sz)
seq_len: 10                                    # Sequence length (temporal window)
batch_sz: 16                                   # Batch size
epochs: 1000                                   # Max epochs (early stopping usually triggers first)

# Model Architecture
lstm_hd_sz: 16                                 # LSTM hidden size
lstm_num_lyrs: 2                               # Number of LSTM layers
dropout: 0.8                                   # Dropout probability

# Optimization
learning_rate: 0.0001                          # Learning rate for Adam optimizer
patience: 120                                  # Early stopping patience (epochs)

# Class Imbalance Handling
bloom_pen: 1.5                                 # Penalty for bloom class (not currently used)
noisy_prop: 0.5                               # Fraction of majority class to augment
noise_pct: 0.03                               # Noise level (% of feature range)

# Thresholding
bloom_thresh: 5.977                           # Manual threshold (not used in current code)
# Threshold is computed from training data as 95th percentile
```

## Parameter Descriptions

### Data Parameters

- **data_path**: Path to the CSV file containing time series data
- **target**: Column name of the target variable (HAB species concentration)

### EDM Parameters

- **EDM_embed**: Boolean flag to enable EDM embeddings
  - `true`: Use EDM-LSTM (with embeddings)
  - `false`: Use Pure LSTM (raw features only)
- **data_bank_sz**: Size of EDM library for training SMap models
- **parameters_path**: Path to CSV with pre-computed EDM parameters
- **samp**: Sampling parameter for EDM (typically 1)
- **n**: Number of models in EDM ensemble (typically 300)

### Training Parameters

- **train_sz**: Total number of samples for training + validation
- **val_len**: Validation set size (carved out from end of train_sz)
- **seq_len**: Length of temporal sequences (sliding window)
- **batch_sz**: Batch size for training
- **epochs**: Maximum training epochs (early stopping usually triggers before this)

### Model Architecture

- **lstm_hd_sz**: Number of hidden units in LSTM layers
- **lstm_num_lyrs**: Number of stacked LSTM layers (use > 1 for dropout)
- **dropout**: Dropout probability (applied between LSTM layers if num_lyrs > 1)

### Optimization

- **learning_rate**: Learning rate for Adam optimizer
- **patience**: Early stopping patience in epochs
  - Training stops if validation AUC doesn't improve for this many epochs

### Data Augmentation

- **noisy_prop**: Fraction of majority class to augment with noise
  - 0.5 means 50% of non-bloom samples get noisy versions
- **noise_pct**: Noise level as percentage of feature range
  - 0.03 = 3% of range

### Class Imbalance

- **bloom_pen**: Weight for bloom class in loss function (not currently used)
  - Current code uses WeightedRandomSampler instead

## Typical Values

### For Small Dataset (~1000 samples)
```yaml
train_sz: 450
seq_len: 10
batch_sz: 16
lstm_hd_sz: 16
lstm_num_lyrs: 2
```

### For Large Dataset (>10000 samples)
```yaml
train_sz: 5000
seq_len: 20
batch_sz: 64
lstm_hd_sz: 64
lstm_num_lyrs: 3
```

### For Quick Testing
```yaml
train_sz: 200
epochs: 50
patience: 20
```

## Notes

1. **train_sz and val_len**: Actual training size is `train_sz - val_len`
2. **Threshold**: Binary classification threshold is computed from training data only (95th percentile)
3. **EDM embedding**: When `EDM_embed=true`, feature dimension becomes `n + 2` (n models + 2 seasonal features)
4. **Memory**: Larger `batch_sz` or `lstm_hd_sz` requires more GPU memory

## Creating Custom Configs

1. Copy `lstm_carter_script.yaml`
2. Modify parameters for your dataset
3. Pass config path to experiment scripts:
   ```bash
   python experiments/run_pure_lstm.py configs/my_config.yaml
   ```
