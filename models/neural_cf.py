"""
Neural Collaborative Filtering (NCF) — PyTorch Implementation
Based on: He et al., "Neural Collaborative Filtering" (WWW 2017)

Architecture:
  - GMF (Generalized Matrix Factorization): element-wise product of user/item embeddings
  - MLP: concatenation of user/item embeddings through deep layers
  - NeuMF: fusion of GMF + MLP outputs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class RatingsDataset(Dataset):
    """PyTorch Dataset for user-item rating pairs."""

    def __init__(self, df: pd.DataFrame, user_col='userId', item_col='movieId',
                 rating_col='rating'):
        self.users = torch.LongTensor(df[user_col].values)
        self.items = torch.LongTensor(df[item_col].values)
        self.ratings = torch.FloatTensor(df[rating_col].values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class GMF(nn.Module):
    """Generalized Matrix Factorization"""

    def __init__(self, n_users, n_items, n_factors=16):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.output = nn.Linear(n_factors, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        interaction = u * i  # element-wise
        return self.output(interaction).squeeze()


class MLP(nn.Module):
    """Multi-Layer Perceptron for NCF"""

    def __init__(self, n_users, n_items, n_factors=32, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

        layers = []
        in_dim = n_factors * 2
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        x = torch.cat([u, i], dim=-1)
        x = self.mlp(x)
        return self.output(x).squeeze()


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization — combines GMF + MLP.
    Best performing NCF model.
    """

    def __init__(self, n_users, n_items, gmf_factors=16, mlp_factors=32,
                 hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        # GMF embeddings
        self.gmf_user = nn.Embedding(n_users, gmf_factors)
        self.gmf_item = nn.Embedding(n_items, gmf_factors)

        # MLP embeddings
        self.mlp_user = nn.Embedding(n_users, mlp_factors)
        self.mlp_item = nn.Embedding(n_items, mlp_factors)

        # MLP layers
        mlp_layers = []
        in_dim = mlp_factors * 2
        for h in hidden_dims:
            mlp_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction
        self.output = nn.Linear(gmf_factors + in_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user, item):
        # GMF path
        gmf_u = self.gmf_user(user)
        gmf_i = self.gmf_item(item)
        gmf_out = gmf_u * gmf_i

        # MLP path
        mlp_u = self.mlp_user(user)
        mlp_i = self.mlp_item(item)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Fusion
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output(fused).squeeze()


class NCFTrainer:
    """Trainer for NCF models with early stopping."""

    def __init__(self, model, lr=0.001, weight_decay=1e-5, device=None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for users, items, ratings in loader:
            users, items, ratings = (
                users.to(self.device),
                items.to(self.device),
                ratings.to(self.device)
            )
            self.optimizer.zero_grad()
            preds = self.model(users, items)
            loss = self.criterion(preds, ratings)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(ratings)
        return np.sqrt(total_loss / len(loader.dataset))

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_targets = [], []
        for users, items, ratings in loader:
            users, items = users.to(self.device), items.to(self.device)
            preds = self.model(users, items).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(ratings.numpy())
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = np.mean(np.abs(np.array(all_targets) - np.array(all_preds)))
        return {'RMSE': rmse, 'MAE': mae}

    def fit(self, train_loader, val_loader=None, n_epochs=20, patience=5):
        best_val_rmse = float('inf')
        patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            train_rmse = self.train_epoch(train_loader)
            self.train_losses.append(train_rmse)

            log = f"Epoch {epoch:>3}/{n_epochs} | Train RMSE: {train_rmse:.4f}"

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                val_rmse = val_metrics['RMSE']
                self.val_losses.append(val_rmse)
                log += f" | Val RMSE: {val_rmse:.4f} | Val MAE: {val_metrics['MAE']:.4f}"

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_neumf.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break

            print(log)

        return self

    @torch.no_grad()
    def recommend(self, user_id: int, all_item_ids: list, top_n=10):
        """Recommend top-N items for a user."""
        self.model.eval()
        users = torch.LongTensor([user_id] * len(all_item_ids)).to(self.device)
        items = torch.LongTensor(all_item_ids).to(self.device)
        scores = self.model(users, items).cpu().numpy()
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(all_item_ids[i], scores[i]) for i in top_idx]
