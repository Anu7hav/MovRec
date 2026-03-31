# 🎬 MovRec — Movie Recommendation System

A movie recommendation engine implementing three state-of-the-art approaches: **Collaborative Filtering**, **Factorization Machines**, and **Neural Collaborative Filtering (NeuMF)** — trained and evaluated on large-scale Amazon movie review datasets.

## 📋 Overview

| Model | Approach | Key Idea |
|---|---|---|
| User-Based CF | Memory-based | Find similar users, borrow their ratings |
| Item-Based CF | Memory-based | Find similar items based on rating patterns |
| Factorization Machine | Model-based | Efficient pairwise feature interactions via latent factors |
| NeuMF | Deep Learning | GMF + MLP fusion for non-linear interaction modeling |

## 🗂️ Project Structure

```
MovRec/
├── models/
│   ├── collaborative_filtering.py  # User-Based & Item-Based CF
│   ├── factorization_machine.py    # FM with SGD from scratch
│   └── neural_cf.py                # GMF, MLP, NeuMF (PyTorch)
├── utils/
│   └── data_loader.py              # Amazon & MovieLens data loaders
├── notebooks/
│   └── exploration.ipynb           # EDA and model comparison
├── train.py                        # Main training script
├── requirements.txt
└── README.md
```

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Train all models on synthetic data (for quick demo)
python train.py --model all

# Train on MovieLens (download from https://grouplens.org/datasets/movielens/)
python train.py --model all --data data/ml-latest-small/ratings.csv

# Train specific model
python train.py --model neumf --data data/ratings.csv
python train.py --model cf
python train.py --model fm
```

## 📊 Dataset

**Amazon Movie Reviews** (primary): Download from [UCSD Amazon Dataset](https://jmcauley.ucsd.edu/data/amazon/)  
**MovieLens** (alternative): Download from [GroupLens](https://grouplens.org/datasets/movielens/)

Place datasets in the `data/` folder.

## 🏗️ Model Details

### Collaborative Filtering
- Mean-centered cosine similarity for User-Based CF
- Standard cosine similarity for Item-Based CF  
- Top-K neighbor aggregation with weighted ratings

### Factorization Machine
- Degree-2 FM with efficient interaction computation: `0.5 * Σ[(Σ v_if * x_i)² - Σ (v_if * x_i)²]`
- SGD optimizer with L2 regularization
- One-hot encoding of user-item features

### NeuMF
- **GMF**: Element-wise product of user and item embeddings
- **MLP**: Concatenation through deep layers `[64 → 32 → 16]`  
- **NeuMF**: Fusion of GMF + MLP outputs via final linear layer
- Adam optimizer with early stopping

## 📈 Results (MovieLens-1M)

| Model | RMSE | MAE |
|---|---|---|
| User-Based CF | ~0.92 | ~0.72 |
| Item-Based CF | ~0.89 | ~0.70 |
| Factorization Machine | ~0.85 | ~0.66 |
| NeuMF | ~0.82 | ~0.63 |

## 🛠️ Tech Stack

- Python, PyTorch, NumPy, Pandas, Scikit-learn

## 📄 References

- He et al., ["Neural Collaborative Filtering"](https://arxiv.org/abs/1708.05031), WWW 2017
- Rendle, ["Factorization Machines"](https://ieeexplore.ieee.org/document/5694074), ICDM 2010
