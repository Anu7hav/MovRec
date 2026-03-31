"""
Data loading and preprocessing utilities for Amazon Movie Review datasets.
Supports both the 5-core and full Amazon review datasets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os


def load_amazon_reviews(filepath: str, nrows=None) -> pd.DataFrame:
    """
    Load Amazon movie reviews dataset (JSON format).
    Download from: https://jmcauley.ucsd.edu/data/amazon/

    Args:
        filepath: Path to .json or .json.gz file
        nrows: Limit rows for quick experiments (None = all)
    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp
    """
    records = []
    opener = open

    if filepath.endswith('.gz'):
        import gzip
        opener = gzip.open

    with opener(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if nrows and i >= nrows:
                break
            record = json.loads(line.strip())
            records.append({
                'userId': record.get('reviewerID', ''),
                'movieId': record.get('asin', ''),
                'rating': float(record.get('overall', 0)),
                'timestamp': record.get('unixReviewTime', 0),
                'title': record.get('summary', ''),
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} reviews | {df['userId'].nunique():,} users | {df['movieId'].nunique():,} items")
    return df


def load_movielens(filepath: str) -> pd.DataFrame:
    """
    Load MovieLens ratings CSV (ml-latest-small or ml-25m).
    Download from: https://grouplens.org/datasets/movielens/

    Returns DataFrame with: userId, movieId, rating, timestamp
    """
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]
    print(f"Loaded {len(df):,} ratings | {df['userid'].nunique():,} users | {df['movieid'].nunique():,} movies")
    df.rename(columns={'userid': 'userId', 'movieid': 'movieId'}, inplace=True)
    return df


def preprocess(df: pd.DataFrame, min_user_ratings=5, min_item_ratings=5,
               rating_col='rating') -> pd.DataFrame:
    """
    Filter cold-start users and items, encode IDs.

    Args:
        df: Raw ratings DataFrame
        min_user_ratings: Minimum ratings a user must have
        min_item_ratings: Minimum ratings an item must have
    Returns:
        Cleaned DataFrame with encoded integer IDs
    """
    print(f"Before filtering: {len(df):,} interactions")

    # Iterative filtering
    for _ in range(3):
        user_counts = df['userId'].value_counts()
        item_counts = df['movieId'].value_counts()
        df = df[df['userId'].isin(user_counts[user_counts >= min_user_ratings].index)]
        df = df[df['movieId'].isin(item_counts[item_counts >= min_item_ratings].index)]

    print(f"After filtering: {len(df):,} interactions | "
          f"{df['userId'].nunique():,} users | {df['movieId'].nunique():,} items")

    # Encode to contiguous integers
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df = df.copy()
    df['userId'] = user_enc.fit_transform(df['userId'])
    df['movieId'] = item_enc.fit_transform(df['movieId'])

    return df, user_enc, item_enc


def build_ratings_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build user-item ratings matrix for CF models."""
    return df.pivot_table(index='userId', columns='movieId', values='rating')


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.1, seed=42):
    """
    Temporal or random train/val/test split.
    """
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')

    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=seed)

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


def generate_sample_data(n_users=200, n_items=500, n_ratings=5000, seed=42):
    """
    Generate synthetic rating data for testing (when Amazon data not downloaded).
    Ratings follow a biased normal distribution around item quality.
    """
    rng = np.random.RandomState(seed)
    item_quality = rng.uniform(2.5, 5.0, n_items)

    users = rng.randint(0, n_users, n_ratings)
    items = rng.randint(0, n_items, n_ratings)
    ratings = np.clip(
        item_quality[items] + rng.normal(0, 0.5, n_ratings),
        1.0, 5.0
    ).round(1)

    df = pd.DataFrame({'userId': users, 'movieId': items, 'rating': ratings})
    df = df.drop_duplicates(['userId', 'movieId'])
    print(f"Generated {len(df):,} synthetic ratings")
    return df
