"""
Factorization Machines for Movie Recommendation
Captures pairwise feature interactions efficiently via latent factor decomposition.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class FactorizationMachine:
    """
    Factorization Machine (degree-2) implemented from scratch with SGD.

    Model: y(x) = w0 + <w, x> + sum_i sum_j <vi, vj> xi xj
    Efficient computation via: 0.5 * sum_f [(sum_i vi_f * xi)^2 - sum_i (vi_f * xi)^2]
    """

    def __init__(self, n_factors=10, learning_rate=0.01, reg_w=0.01,
                 reg_v=0.01, n_epochs=20, seed=42):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.n_epochs = n_epochs
        self.seed = seed

        self.w0 = 0.0
        self.w = None
        self.V = None
        self.loss_history = []

    def _init_params(self, n_features):
        rng = np.random.RandomState(self.seed)
        self.w = np.zeros(n_features)
        self.V = rng.normal(0, 0.01, (n_features, self.n_factors))

    def _predict_sample(self, x):
        """Predict for a single sample x."""
        linear = self.w0 + np.dot(self.w, x)

        # Efficient pairwise interaction
        xV = x[:, None] * self.V          # (n_features, n_factors)
        interaction = 0.5 * np.sum(
            np.sum(xV, axis=0) ** 2 - np.sum(xV ** 2, axis=0)
        )
        return linear + interaction

    def predict(self, X):
        """Predict for a batch of samples."""
        return np.array([self._predict_sample(x) for x in X])

    def fit(self, X, y):
        """
        Train using SGD.
        Args:
            X: Feature matrix (n_samples, n_features) — sparse-friendly
            y: Target ratings
        """
        n_samples, n_features = X.shape
        self._init_params(n_features)
        y = np.array(y, dtype=np.float32)

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for i in indices:
                x = X[i]
                y_hat = self._predict_sample(x)
                error = y_hat - y[i]

                # Gradients
                grad_w0 = error
                grad_w = error * x + self.reg_w * self.w

                xV = x[:, None] * self.V  # (n_features, n_factors)
                sum_xV = np.sum(xV, axis=0)  # (n_factors,)
                grad_V = error * (x[:, None] * sum_xV[None, :] - xV) + self.reg_v * self.V

                # Updates
                self.w0 -= self.lr * grad_w0
                self.w -= self.lr * grad_w
                self.V -= self.lr * grad_V

                epoch_loss += error ** 2

            rmse = np.sqrt(epoch_loss / n_samples)
            self.loss_history.append(rmse)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.n_epochs} — RMSE: {rmse:.4f}")

        return self

    def evaluate(self, X, y):
        preds = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        mae = np.mean(np.abs(np.array(y) - preds))
        return {'RMSE': rmse, 'MAE': mae}


class FMDataEncoder:
    """
    Encode user-item interactions into one-hot feature vectors for FM.
    Supports additional context features (e.g., genre, year).
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
        self.n_users = 0
        self.n_items = 0

    def fit(self, df: pd.DataFrame, user_col='userId', item_col='movieId'):
        self.user_enc.fit(df[user_col])
        self.item_enc.fit(df[item_col])
        self.n_users = len(self.user_enc.classes_)
        self.n_items = len(self.item_enc.classes_)
        return self

    def transform(self, df: pd.DataFrame, user_col='userId', item_col='movieId'):
        """Convert to one-hot encoded feature matrix."""
        n = len(df)
        n_features = self.n_users + self.n_items
        X = np.zeros((n, n_features))

        user_ids = self.user_enc.transform(df[user_col])
        item_ids = self.item_enc.transform(df[item_col])

        for i, (u, it) in enumerate(zip(user_ids, item_ids)):
            X[i, u] = 1.0
            X[i, self.n_users + it] = 1.0

        return X

    def fit_transform(self, df, user_col='userId', item_col='movieId'):
        return self.fit(df, user_col, item_col).transform(df, user_col, item_col)
