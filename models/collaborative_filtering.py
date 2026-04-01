"""
Collaborative Filtering for Movie Recommendations
User-based and Item-based CF using cosine similarity
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class UserBasedCF:
    """User-Based Collaborative Filtering"""

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.user_similarity = None
        self.ratings_matrix = None
        self.user_mean = None

    def fit(self, ratings_matrix: pd.DataFrame):
        """
        Fit the model on a user-item ratings matrix.
        Args:
            ratings_matrix: DataFrame with users as rows, items as columns
        """
        self.ratings_matrix = ratings_matrix
        self.user_mean = ratings_matrix.mean(axis=1)

        # Mean-center the ratings
        normalized = ratings_matrix.sub(self.user_mean, axis=0).fillna(0)
        self.user_similarity = cosine_similarity(normalized)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=ratings_matrix.index,
            columns=ratings_matrix.index
        )
        return self

    def predict(self, user_id, item_id):
        """Predict rating of a user for an item."""
        if user_id not in self.ratings_matrix.index:
            return self.user_mean.mean()
        if item_id not in self.ratings_matrix.columns:
            return self.user_mean[user_id]

        # Get similar users who rated this item
        sim_scores = self.user_similarity[user_id].drop(user_id)
        item_ratings = self.ratings_matrix[item_id].dropna()
        common_users = sim_scores.index.intersection(item_ratings.index)

        if len(common_users) == 0:
            return self.user_mean[user_id]

        sim_scores = sim_scores[common_users]
        item_ratings = item_ratings[common_users]

        # Top-K neighbors
        top_k = sim_scores.abs().nlargest(self.n_neighbors).index
        sim_scores = sim_scores[top_k]
        item_ratings = item_ratings[top_k]

        # Weighted average
        numerator = np.dot(sim_scores, item_ratings - self.user_mean[top_k])
        denominator = sim_scores.abs().sum()

        if denominator == 0:
            return self.user_mean[user_id]

        return self.user_mean[user_id] + numerator / denominator

    def recommend(self, user_id, n=10, exclude_rated=True):
        """
        Recommend top-N movies for a given user.
        Args:
            user_id: Target user ID
            n: Number of recommendations
            exclude_rated: If True, exclude already-rated movies
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if user_id not in self.ratings_matrix.index:
            raise ValueError(f"User {user_id} not found.")

        all_items = self.ratings_matrix.columns
        if exclude_rated:
            rated = self.ratings_matrix.loc[user_id].dropna().index
            candidates = all_items.difference(rated)
        else:
            candidates = all_items

        predictions = {item: self.predict(user_id, item) for item in candidates}
        top_n = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
        return top_n


class ItemBasedCF:
    """Item-Based Collaborative Filtering"""

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.item_similarity = None
        self.ratings_matrix = None

    def fit(self, ratings_matrix: pd.DataFrame):
        self.ratings_matrix = ratings_matrix
        filled = ratings_matrix.fillna(0)
        item_sim = cosine_similarity(filled.T)
        self.item_similarity = pd.DataFrame(
            item_sim,
            index=ratings_matrix.columns,
            columns=ratings_matrix.columns
        )
        return self

    def predict(self, user_id, item_id):
        if user_id not in self.ratings_matrix.index:
            return self.ratings_matrix.mean().mean()
        if item_id not in self.ratings_matrix.columns:
            return self.ratings_matrix.loc[user_id].mean()

        user_ratings = self.ratings_matrix.loc[user_id].dropna()
        rated_items = user_ratings.index.intersection(self.item_similarity.index)

        if len(rated_items) == 0:
            return self.ratings_matrix.loc[user_id].mean()

        sim_scores = self.item_similarity[item_id][rated_items]
        top_k = sim_scores.abs().nlargest(self.n_neighbors).index
        sim_scores = sim_scores[top_k]
        ratings = user_ratings[top_k]

        denominator = sim_scores.abs().sum()
        if denominator == 0:
            return self.ratings_matrix.loc[user_id].mean()

        return np.dot(sim_scores, ratings) / denominator

    def recommend(self, user_id, n=10, exclude_rated=True):
        if user_id not in self.ratings_matrix.index:
            raise ValueError(f"User {user_id} not found.")

        all_items = self.ratings_matrix.columns
        if exclude_rated:
            rated = self.ratings_matrix.loc[user_id].dropna().index
            candidates = all_items.difference(rated)
        else:
            candidates = all_items

        predictions = {item: self.predict(user_id, item) for item in candidates}
        top_n = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
        return top_n


def evaluate_cf(model, test_data: pd.DataFrame, ratings_matrix: pd.DataFrame):
    """Evaluate RMSE and MAE on test set."""
    predictions, actuals = [], []
    for _, row in test_data.iterrows():
        pred = model.predict(row['userId'], row['movieId'])
        predictions.append(pred)
        actuals.append(row['rating'])

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
    return {'RMSE': rmse, 'MAE': mae}
