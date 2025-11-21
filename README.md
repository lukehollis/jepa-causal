# JEPA-DML Causal Inference Engine

A global causal inference engine using Spatiotemporal Joint Embedding Predictive Architecture (JEPA) world models and Double Machine Learning (DML) for robust causal analysis from high-dimensional data.

## Overview

This implementation combines:
- **V-JEPA-inspired self-supervised learning** for spatiotemporal representation learning
- **Double Machine Learning (DML)** with K-fold cross-fitting for causal effect estimation
- **Confounder proxy networks** to handle latent confounding in observational data

## Setup

### Using Conda (Recommended)

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate jepa_dml
```

### Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training (Demo)

Run the demo simulation to verify installation:

```bash
conda activate jepa_dml
python scripts/train.py
```

This will:
1. Generate simulated spatiotemporal data with latent confounders
2. Train the JEPA world model using K-fold cross-fitting
3. Estimate the Average Treatment Effect (ATE) using DML
4. Output results with confidence intervals

Expected output should show an estimated ATE close to the true value of 1.0.

### Inference (Apply to New Data)

Apply the framework to your own causal questions:

```bash
python scripts/inference.py \
  --data path/to/your/data.h5 \
  --treatment "policy_intervention" \
  --outcome "economic_outcome" \
  --checkpoint checkpoints/jepa_encoder.pt \
  --output results/my_analysis.json
```

### Evaluation (Benchmarking)

Run the full evaluation suite from the ICCV paper:

```bash
# Quick evaluation (100 runs)
python scripts/eval.py --n_runs 100

# Full paper benchmark (500 runs)
python scripts/eval.py --n_runs 500 --eval_sufficiency --eval_overlap
```

This generates:
- Bias, RMSE, and coverage statistics
- Sufficiency validation (R² for U prediction)
- Overlap diagnostics (propensity score distribution)
- Distribution plots in `results/`

### Baselines & Comparison

Compare the JEPA-based approach against standard baselines:

```bash
python scripts/run_baselines.py
```

This evaluates:
1.  **JEPA-DML** (Proposed Method)
2.  **VICReg-DML** (Self-supervised baseline using Variance-Invariance-Covariance Regularization)
3.  **DragonNet** (End-to-end causal inference neural network)
4.  **Oracle** (Supervised baseline with access to latent confounders)

Results including ATE estimates and errors are saved to `results/baselines/comparison_results.csv`.

## Project Structure

```
jepa_dml/
├── scripts/
│   ├── train.py                 # Training demo/simulation
│   ├── inference.py             # Apply to new causal questions
│   ├── eval.py                  # Benchmark suite
│   └── run_baselines.py         # Baseline comparison runner
├── src/
│   ├── baselines/               # Comparison methods (DragonNet, VICReg, Oracle)
│   ├── data/
│   │   └── dataset.py           # Spatiotemporal data handling
│   ├── models/
│   │   ├── encoder.py           # Vision Transformer (ViT) encoder
│   │   ├── jepa.py              # JEPA architecture and training
│   │   └── masking.py           # Tube masking strategies
│   └── causal/
│       ├── nuisance_models.py   # Proxy, outcome, propensity models
│       └── dml_engine.py        # DML cross-fitting workflow
├── results/                     # Evaluation outputs
├── checkpoints/                 # Saved model weights
├── environment.yml              # Conda environment
└── requirements.txt             # Pip requirements
```

## Configuration

Edit the `CONFIG` dictionary in `scripts/train.py` to adjust:

### Data Parameters
- `n_samples`: Number of data samples
- `data_dims`: Spatiotemporal dimensions (C, T, H, W)
- `k_folds`: Number of cross-fitting folds

### JEPA Parameters
- `vit_size`: ViT variant ('small', 'base', 'large')
- `patch_size`: Spatial patch size (8 or 16)
- `tubelet_size`: Temporal grouping size (2)
- `rep_dim`: JEPA representation dimension
- `jepa_epochs`: Training epochs per fold (20-100)
- `mask_ratio`: Fraction of patches to mask (0.75)
- `num_mask_blocks`: Number of tube mask regions (4)

### DML Parameters
- `proxy_dim`: Confounder proxy dimension
- `dml_epochs`: Nuisance model training epochs

## Key Features

### Rigorous DML Implementation
- Per-fold JEPA training from scratch to prevent data leakage
- Out-of-sample predictions for valid causal inference
- Neyman-orthogonal score functions (AIPW estimator)

### Spatiotemporal World Model
- Self-supervised learning via masked prediction
- Momentum encoder for stable target representations
- Captures temporal dependencies in video/time-series data

### Confounder Proxy Network
- TarNet-style architecture for outcome prediction
- Separate propensity model for treatment probability
- Handles high-dimensional latent confounding

## Citation

Based on research from:
- V-JEPA 2 (Meta FAIR)
- Self-Supervised Predictive Representations for Causal Inference
- GenAI-Powered Inference (GPI) framework

## License

MIT
