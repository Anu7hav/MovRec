"""
MovRec — Main Training & Evaluation Script
Trains CF, FM, and NeuMF models and compares their performance.

Usage:
    python train.py --model all          # Train all models
    python train.py --model neumf        # Train NeuMF only
    python train.py --model cf           # Train CF only
    python train.py --data path/to/ratings.csv
"""

import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from utils.data_loader import (
    generate_sample_data, preprocess, build_ratings_matrix,
    split_data, load_movielens
)
from models.collaborative_filtering import UserBasedCF, ItemBasedCF, evaluate_cf
from models.factorization_machine import FactorizationMachine, FMDataEncoder
from models.neural_cf import NeuMF, RatingsDataset, NCFTrainer


def train_cf(train_df, test_df):
    print("\n" + "="*50)
    print("Collaborative Filtering")
    print("="*50)

    ratings_matrix = build_ratings_matrix(train_df)

    print("\n[User-Based CF]")
    user_cf = UserBasedCF(n_neighbors=20)
    user_cf.fit(ratings_matrix)
    metrics = evaluate_cf(user_cf, test_df.sample(min(500, len(test_df))), ratings_matrix)
    print(f"  RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")

    print("\n[Item-Based CF]")
    item_cf = ItemBasedCF(n_neighbors=20)
    item_cf.fit(ratings_matrix)
    metrics = evaluate_cf(item_cf, test_df.sample(min(500, len(test_df))), ratings_matrix)
    print(f"  RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")

    return user_cf, item_cf


def train_fm(train_df, val_df, test_df):
    print("\n" + "="*50)
    print("Factorization Machine")
    print("="*50)

    encoder = FMDataEncoder()
    X_train = encoder.fit_transform(train_df)
    X_val = encoder.transform(val_df)
    X_test = encoder.transform(test_df)

    y_train = train_df['rating'].values
    y_val = val_df['rating'].values
    y_test = test_df['rating'].values

    fm = FactorizationMachine(n_factors=16, learning_rate=0.005, n_epochs=20)
    print("\nTraining FM...")
    fm.fit(X_train, y_train)

    val_metrics = fm.evaluate(X_val, y_val)
    test_metrics = fm.evaluate(X_test, y_test)
    print(f"\nVal  — RMSE: {val_metrics['RMSE']:.4f} | MAE: {val_metrics['MAE']:.4f}")
    print(f"Test — RMSE: {test_metrics['RMSE']:.4f} | MAE: {test_metrics['MAE']:.4f}")

    return fm


def train_neumf(train_df, val_df, test_df):
    print("\n" + "="*50)
    print("Neural Collaborative Filtering (NeuMF)")
    print("="*50)

    n_users = max(train_df['userId'].max(), val_df['userId'].max(), test_df['userId'].max()) + 1
    n_items = max(train_df['movieId'].max(), val_df['movieId'].max(), test_df['movieId'].max()) + 1

    train_ds = RatingsDataset(train_df)
    val_ds   = RatingsDataset(val_df)
    test_ds  = RatingsDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=512)
    test_loader  = DataLoader(test_ds, batch_size=512)

    model = NeuMF(n_users=n_users, n_items=n_items,
                  gmf_factors=16, mlp_factors=32,
                  hidden_dims=[64, 32, 16])

    trainer = NCFTrainer(model, lr=0.001)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training NeuMF...")
    trainer.fit(train_loader, val_loader, n_epochs=20, patience=5)

    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest — RMSE: {test_metrics['RMSE']:.4f} | MAE: {test_metrics['MAE']:.4f}")

    return trainer


def main(args):
    print("MovRec — Movie Recommendation System")
    print("Loading data...")

    if args.data:
        df = load_movielens(args.data)
    else:
        print("No data file specified. Using synthetic data (use --data path/to/ratings.csv for real data).")
        df = generate_sample_data(n_users=300, n_items=500, n_ratings=8000)

    df, user_enc, item_enc = preprocess(df, min_user_ratings=3, min_item_ratings=3)
    train_df, val_df, test_df = split_data(df)

    results = {}

    if args.model in ('all', 'cf'):
        train_cf(train_df, test_df)

    if args.model in ('all', 'fm'):
        train_fm(train_df, val_df, test_df)

    if args.model in ('all', 'neumf'):
        train_neumf(train_df, val_df, test_df)

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all', choices=['all', 'cf', 'fm', 'neumf'])
    parser.add_argument('--data', default=None, help='Path to ratings CSV file')
    args = parser.parse_args()
    main(args)
