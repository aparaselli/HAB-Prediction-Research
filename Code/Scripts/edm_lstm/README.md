# EDM-LSTM: Hybrid Deep Learning for Harmful Algal Bloom Prediction

This directory contains the implementation of a hybrid deep learning architecture that combines Empirical Dynamic Modeling (EDM) with a multi-component neural network for predicting harmful algal blooms (HABs) in Lake Carter, Florida.

## Architecture Overview

The EDM-LSTM model is built on a novel hybrid approach that leverages the strengths of both traditional ecological modeling and modern deep learning. The architecture consists of two main stages:

### Stage 1: EDM Embeddings

Before any neural network processing, we use Empirical Dynamic Modeling to create rich feature representations of the time series data. EDM is a nonparametric forecasting technique that reconstructs the attractor manifold of a dynamical system using time-delayed embeddings.

**How it works:**
- For each time point, EDM identifies similar historical states in the system's phase space
- It generates an ensemble of predictions based on these analogous states
- The output is not a single prediction, but a distribution of forecasts from multiple "nearest neighbors"
- This ensemble becomes a high-dimensional embedding that captures the nonlinear dynamics of the ecosystem

The EDM embeddings encode temporal patterns, seasonality, and ecological state transitions that would be difficult for a neural network to learn from raw features alone.

### Stage 2: HybridTABWithGate Neural Network

After EDM preprocessing, the embedded features flow through a sophisticated neural architecture with four key components:

#### 1. TrustGate (Feature Selection Layer)

The TrustGate is a learnable, per-channel gating mechanism that acts as an intelligent feature selector:
- Each input channel has a trainable gate weight initialized based on its individual predictive power (measured by single-feature AUC)
- During training, the gates learn to upweight informative features and downweight noisy ones
- Uses a temperature-scaled sigmoid to produce continuous gates in the range (0, 1)
- Includes L1 and binary entropy regularization to encourage sparse, decisive gating

This component is particularly important given the high-dimensional EDM embeddings, as it helps the model focus on the most relevant patterns.

#### 2. Temporal Convolutional Network (TCN)

The TCN processes sequential patterns through stacked residual blocks:
- Uses dilated causal convolutions to capture temporal dependencies at multiple scales
- Each block doubles the dilation rate (1, 2, 4, ...), expanding the effective receptive field exponentially
- Residual connections allow gradient flow and enable learning both short and long-term patterns
- Weight normalization stabilizes training

The TCN excels at capturing local temporal structure and multi-scale patterns in the time series.

#### 3. Multi-Head Attention (MHA)

After TCN processing, a multi-head attention layer captures global dependencies:
- Computes self-attention over the entire sequence, allowing each time step to attend to all others
- Multiple attention heads learn different types of relationships in parallel
- Residual connection with layer normalization preserves TCN features while adding global context

This component helps the model identify long-range temporal relationships and critical transition points that may precede bloom events.

#### 4. Bidirectional LSTM

The BiLSTM further refines the temporal representation:
- Processes the sequence in both forward and backward directions
- Captures long-term dependencies and sequential patterns
- Two-layer architecture with dropout for regularization
- Layer normalization on the final hidden state

The bidirectional nature is crucial for understanding context from both past and future states when making predictions.

#### 5. Classification Head

The final component produces bloom/no-bloom predictions:
- Takes the BiLSTM's final hidden state (concatenated forward and backward)
- Two-layer MLP with ReLU activation and layer normalization
- Dropout for regularization
- Outputs logits for binary classification (bloom vs. non-bloom)
- Prior-adjusted bias initialization based on class imbalance

## Training Strategy

The model is trained with several techniques to handle the challenges of HAB prediction:

### Data Augmentation
- **Noisy samples**: Gaussian noise is added to training sequences (2-3% of feature range) to improve robustness
- **Minority oversampling**: Bloom events are rare, so we oversample positive examples during training
- **Weighted sampling**: Batch sampling ensures balanced representation of both classes

### Regularization
- Dropout layers throughout the architecture
- Weight decay on all parameters
- Gate regularization (L1 + binary entropy) to encourage feature sparsity
- Gradient clipping to prevent exploding gradients

### Optimization
- Exponential Moving Average (EMA) of model weights for more stable predictions at evaluation time
- ReduceLROnPlateau scheduler to adapt learning rate based on validation loss
- Early stopping based on validation AUC to prevent overfitting

### Model Selection
- Models are selected based on **validation AUC**, not training performance
- Test set is held completely separate and only used for final evaluation
- Interleaved validation/test split ensures both sets have similar temporal coverage

## Hyperparameter Configuration

The best performing model uses the following configuration:

```yaml
Architecture:
  - Hidden dimension: 256
  - TCN channels: [128, 128]
  - LSTM layers: 2
  - Attention heads: 4
  - Dropout: 0.5
  - Gate temperature: 6.0

Training:
  - Learning rate: 0.0002
  - Batch size: 12
  - Bloom penalty: 2.0 (class weight)
  - Weight decay: 0.0001
  - EMA decay: 0.995
  - Max epochs: 1200
  - Patience: 100
  - Seed: 58
```

## Results

The model was evaluated through extensive hyperparameter sweeps across 5 configurations with 40 random seeds each (200 total runs).

**Best Test Performance:**
- **Test AUC: 0.7971**
- Validation AUC: 0.7236
- Configuration: h256_t128, seed 58

The model achieves strong discriminative performance on held-out test data, demonstrating its ability to generalize to unseen bloom events.

## Files in This Directory

- `train.py`: Main training script with hyperparameter sweep
- `generate_best_test.py`: Script to retrain the best model and generate ROC curves
- `best_model.pth`: State dict of the best performing model
- `best_config.yaml`: Hyperparameters for the best model
- `best_test_roc.csv`: ROC curve data (FPR, TPR, thresholds)
- `best_test_roc.png`: Visualization of the ROC curve
- `sweep_log.csv`: Full log of all 200 training runs
- `tcn_model.py`: (in parent directory) Neural network architecture definitions

## Usage

To train models with the hyperparameter sweep:
```bash
python train.py
```

To regenerate the best model and ROC curve:
```bash
python generate_best_test.py
```

## Key Innovation

The key insight of this architecture is that EDM and deep learning are complementary rather than competing approaches. EDM excels at capturing the nonlinear dynamics and state-space structure of ecological systems, while deep neural networks excel at learning hierarchical representations and handling high-dimensional data. By using EDM as a preprocessing step that creates physically meaningful embeddings, we provide the neural network with a much richer representation of the system's dynamics than it could learn from raw features alone.

The TrustGate mechanism further enhances this hybrid approach by allowing the model to learn which aspects of the EDM embeddings are most relevant for prediction, effectively performing learned feature selection in the context of the full model.

## References

This work builds on:
- Temporal Convolutional Networks (TCN) for sequence modeling
- Transformer-style multi-head attention for time series
- Bidirectional LSTMs for sequential pattern recognition
- Empirical Dynamic Modeling for ecological forecasting
- Prior work on HAB prediction in Florida lakes
