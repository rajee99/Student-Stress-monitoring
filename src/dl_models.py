"""
PyTorch Deep Learning Classifiers for Tabular Stress Data.
Implements:
1. SETabularClassifier (Squeeze-and-Excitation Feature Attention Network)
2. TabularLSTMClassifier (Bidirectional LSTM with Attention Pooling)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class SqueezeAndExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel/Feature Attention Block for Tabular Embeddings.
    Adaptively recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between feature dimensions.
    """
    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_channels = max(channels // reduction_ratio, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global excitation weights
        weights = self.fc(x)
        return x * weights


class SE_Tabular_Architecture(nn.Module):
    """Deep Squeeze-and-Excitation Residual Architecture for Tabular Classification."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        self.se_block1 = SqueezeAndExcitationBlock(hidden_dim, reduction_ratio=4)
        
        # Residual Dense Block 1
        self.dense_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        self.se_block2 = SqueezeAndExcitationBlock(hidden_dim, reduction_ratio=4)

        # Residual Dense Block 2
        self.dense_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2)
        )

        # Classification Head
        self.classifier_head = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        h = self.se_block1(h)
        
        # Residual connection 1
        residual = h
        h = self.dense_block1(h)
        h = self.se_block2(h)
        h = h + residual

        # Dense reduction
        h = self.dense_block2(h)
        logits = self.classifier_head(h)
        return logits


class LSTM_Tabular_Architecture(nn.Module):
    """Bidirectional LSTM with Self-Attention Readout for Tabular Signals."""
    def __init__(self, input_dim: int, num_classes: int, embedding_dim: int = 32, hidden_dim: int = 64, dropout_rate: float = 0.25):
        super().__init__()
        self.feature_embedder = nn.Linear(1, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, num_features] -> [batch_size, num_features, 1]
        x_seq = x.unsqueeze(-1)
        # Project each feature into an embedding: [batch_size, num_features, embedding_dim]
        embedded = self.feature_embedder(x_seq)
        
        lstm_out, _ = self.lstm(embedded)  # [batch_size, num_features, hidden_dim * 2]
        
        # Self-attention pooling
        attn_scores = self.attention_weights(lstm_out)  # [batch_size, num_features, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, hidden_dim * 2]
        
        logits = self.classifier(pooled)
        return logits


class SETabularClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-Learn compatible Wrapper for Squeeze-and-Excitation Tabular Network."""
    def __init__(self, hidden_dim: int = 128, epochs: int = 120, batch_size: int = 32,
                 lr: float = 0.002, weight_decay: float = 1e-4, dropout: float = 0.3, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y_arr)
        num_classes = len(self.classes_)
        input_dim = X_arr.shape[1]

        # Internal validation split for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y_arr, test_size=0.15, random_state=self.random_state, stratify=y_arr
        )

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model_ = SE_Tabular_Architecture(
            input_dim=input_dim, num_classes=num_classes,
            hidden_dim=self.hidden_dim, dropout_rate=self.dropout
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        best_state = None
        patience, patience_counter = 20, 0

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation step
            self.model_.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_x, v_y in val_loader:
                    v_out = self.model_(v_x)
                    val_loss += criterion(v_out, v_y).item() * len(v_y)
            val_loss /= len(val_dataset)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            self.model_.train()

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            tensor_x = torch.tensor(X_arr)
            logits = self.model_(tensor_x)
            probabilities = torch.softmax(logits, dim=1).numpy()
        return probabilities

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        return self.classes_[pred_indices]


class TabularLSTMClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-Learn compatible Wrapper for Tabular Bidirectional LSTM Classifier."""
    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64, epochs: int = 120,
                 batch_size: int = 32, lr: float = 0.003, weight_decay: float = 1e-4,
                 dropout: float = 0.25, random_state: int = 42):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y_arr)
        num_classes = len(self.classes_)
        input_dim = X_arr.shape[1]

        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y_arr, test_size=0.15, random_state=self.random_state, stratify=y_arr
        )

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model_ = LSTM_Tabular_Architecture(
            input_dim=input_dim, num_classes=num_classes,
            embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, dropout_rate=self.dropout
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        best_state = None
        patience, patience_counter = 20, 0

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_x, v_y in val_loader:
                    v_out = self.model_(v_x)
                    val_loss += criterion(v_out, v_y).item() * len(v_y)
            val_loss /= len(val_dataset)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            self.model_.train()

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            tensor_x = torch.tensor(X_arr)
            logits = self.model_(tensor_x)
            probabilities = torch.softmax(logits, dim=1).numpy()
        return probabilities

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        return self.classes_[pred_indices]
