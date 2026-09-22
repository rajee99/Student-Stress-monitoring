"""
Custom Novel Machine Learning Classifiers Architecture Module.
Strictly Zero Tree-Based Algorithms & Zero Regression Algorithms.

Architectures:
1. KernelManifoldAttentionClassifier (KMAC):
   - Non-parametric metric prototype learning with learned Mahalanobis precision weights.
   - Temperature-scaled hybrid RBF-Laplacian kernel attention.
   - Closed-form and mini-batch gradient optimization for metric weights.

2. ResidualGatedFeatureClassifier (RGFN):
   - Tabular Deep Residual Network with Squeeze-and-Excitation channel gating.
   - Hyperspherical Cosine Similarity classification head (CosFace / Cosine Margin Softmax).
   - Layer Normalization, Dropout, SiLU activations, and Cosine Annealing optimization.
"""

from typing import Optional, Union, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# 1. KERNEL MANIFOLD ATTENTION CLASSIFIER (KMAC)
# =============================================================================

class KernelManifoldAttentionClassifier(BaseEstimator, ClassifierMixin):
    """
    Kernel Manifold Attention Classifier (KMAC).
    
    A geometric metric learning classifier that constructs class Riemannian manifolds
    using multi-prototype clustering, learns feature metric weights via gradient descent,
    and classifies via temperature-scaled hybrid kernel attention.
    
    Strictly NO Decision Trees. Strictly NO Linear/Logistic Regressions.
    """

    def __init__(
        self,
        prototypes_per_class: int = 4,
        temperature: float = 0.45,
        kernel_hybrid_alpha: float = 0.65,
        learning_rate: float = 0.04,
        max_iter: int = 250,
        l2_regularization: float = 1e-4,
        random_state: int = 42
    ):
        self.prototypes_per_class = prototypes_per_class
        self.temperature = temperature
        self.kernel_hybrid_alpha = kernel_hybrid_alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit class prototypes and learn feature metric weights."""
        X, y = check_X_y(X, y)
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        rng = np.random.RandomState(self.random_state)

        # Step 1: Compute sub-cluster prototypes for each class
        prototypes_list = []
        proto_class_list = []
        proto_counts_list = []

        for c_idx in range(self.n_classes_):
            c_mask = (y_indices == c_idx)
            X_c = X[c_mask]
            n_c = len(X_c)

            k = min(self.prototypes_per_class, n_c)
            if k <= 1:
                prototypes_list.append(np.mean(X_c, axis=0, keepdims=True))
                proto_class_list.append(c_idx)
                proto_counts_list.append(n_c)
            else:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
                km.fit(X_c)
                prototypes_list.append(km.cluster_centers_)
                proto_class_list.extend([c_idx] * k)
                _, counts = np.unique(km.labels_, return_counts=True)
                proto_counts_list.extend(counts.tolist())

        self.prototypes_ = np.vstack(prototypes_list)
        self.proto_classes_ = np.array(proto_class_list, dtype=int)
        
        total_counts = np.array(proto_counts_list, dtype=float)
        self.proto_weights_ = total_counts / np.sum(total_counts)

        # Step 2: Learn diagonal metric weights via Adam mini-batch optimization
        log_w = np.zeros(n_features, dtype=float)
        m_t = np.zeros(n_features, dtype=float)
        v_t = np.zeros(n_features, dtype=float)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        batch_size = min(64, n_samples)
        
        for it in range(1, self.max_iter + 1):
            perm = rng.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y_indices[perm]

            for b in range(0, n_samples, batch_size):
                xb = X_shuffled[b:b+batch_size]
                yb = y_shuffled[b:b+batch_size]
                w = np.exp(log_w)

                diff = xb[:, np.newaxis, :] - self.prototypes_[np.newaxis, :, :]
                
                # Hybrid RBF (L2) + Laplacian (L1) metric
                d_l2 = np.sum((diff ** 2) * w[np.newaxis, np.newaxis, :], axis=2)
                d_l1 = np.sum(np.abs(diff) * np.sqrt(w[np.newaxis, np.newaxis, :] + 1e-8), axis=2)
                dist = self.kernel_hybrid_alpha * d_l2 + (1.0 - self.kernel_hybrid_alpha) * d_l1

                proto_logits = -dist / max(self.temperature, 1e-4) + np.log(self.proto_weights_[np.newaxis, :] + 1e-8)
                
                # LogSumExp class pooling
                class_logits = np.zeros((len(xb), self.n_classes_))
                for c in range(self.n_classes_):
                    c_mask = (self.proto_classes_ == c)
                    max_logit = np.max(proto_logits[:, c_mask], axis=1, keepdims=True)
                    class_logits[:, c] = np.squeeze(
                        max_logit + np.log(np.sum(np.exp(proto_logits[:, c_mask] - max_logit), axis=1, keepdims=True) + 1e-8)
                    )

                max_cl = np.max(class_logits, axis=1, keepdims=True)
                exp_logits = np.exp(class_logits - max_cl)
                probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-8)

                y_one_hot = np.zeros_like(probs)
                y_one_hot[np.arange(len(yb)), yb] = 1.0
                grad_logits = (probs - y_one_hot) / len(xb)

                grad_proto_logits = np.zeros_like(proto_logits)
                for c in range(self.n_classes_):
                    c_mask = (self.proto_classes_ == c)
                    sub_logits = proto_logits[:, c_mask]
                    sub_exp = np.exp(sub_logits - np.max(sub_logits, axis=1, keepdims=True))
                    sub_attn = sub_exp / (np.sum(sub_exp, axis=1, keepdims=True) + 1e-8)
                    grad_proto_logits[:, c_mask] = grad_logits[:, c:c+1] * sub_attn

                grad_dist = grad_proto_logits * (-1.0 / max(self.temperature, 1e-4))

                grad_w_l2 = np.sum(grad_dist[:, :, np.newaxis] * (diff ** 2) * self.kernel_hybrid_alpha, axis=(0, 1))
                grad_w_l1 = np.sum(
                    grad_dist[:, :, np.newaxis] * np.abs(diff) * (1.0 - self.kernel_hybrid_alpha) * (0.5 / np.sqrt(w + 1e-8)),
                    axis=(0, 1)
                )
                grad_w = grad_w_l2 + grad_w_l1 + self.l2_regularization * w
                
                grad_log_w = np.clip(grad_w * w, -5.0, 5.0)

                m_t = beta1 * m_t + (1 - beta1) * grad_log_w
                v_t = beta2 * v_t + (1 - beta2) * (grad_log_w ** 2)
                m_hat = m_t / (1.0 - beta1 ** it)
                v_hat = v_t / (1.0 - beta2 ** it)
                log_w -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        self.feature_weights_ = np.exp(log_w)
        self.feature_weights_ = self.feature_weights_ / (np.mean(self.feature_weights_) + 1e-8)
        self.feature_importances_ = self.feature_weights_ / np.sum(self.feature_weights_)
        return self

    def _compute_class_logits(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['prototypes_', 'feature_weights_', 'proto_classes_'])
        X = check_array(X)
        w = self.feature_weights_
        
        diff = X[:, np.newaxis, :] - self.prototypes_[np.newaxis, :, :]
        d_l2 = np.sum((diff ** 2) * w[np.newaxis, np.newaxis, :], axis=2)
        d_l1 = np.sum(np.abs(diff) * np.sqrt(w[np.newaxis, np.newaxis, :] + 1e-8), axis=2)
        dist = self.kernel_hybrid_alpha * d_l2 + (1.0 - self.kernel_hybrid_alpha) * d_l1

        proto_logits = -dist / max(self.temperature, 1e-4) + np.log(self.proto_weights_[np.newaxis, :] + 1e-8)
        
        class_logits = np.zeros((len(X), self.n_classes_))
        for c in range(self.n_classes_):
            c_mask = (self.proto_classes_ == c)
            max_logit = np.max(proto_logits[:, c_mask], axis=1, keepdims=True)
            class_logits[:, c] = np.squeeze(
                max_logit + np.log(np.sum(np.exp(proto_logits[:, c_mask] - max_logit), axis=1, keepdims=True) + 1e-8)
            )
        return class_logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        class_logits = self._compute_class_logits(X)
        max_cl = np.max(class_logits, axis=1, keepdims=True)
        exp_logits = np.exp(class_logits - max_cl)
        return exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-8)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


# =============================================================================
# 2. RESIDUAL GATED FEATURE CLASSIFIER (RGFN)
# =============================================================================

class _GatedFeatureBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_signal = nn.Linear(hidden_dim, hidden_dim)
        self.fc_gate = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Squeeze-and-Excitation
        self.se_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.se_fc2 = nn.Linear(hidden_dim // 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        normed = self.norm(x)
        signal = F.silu(self.fc_signal(normed))
        gate = torch.sigmoid(self.fc_gate(normed))
        gated = signal * gate
        
        se = F.relu(self.se_fc1(gated))
        se_weights = torch.sigmoid(self.se_fc2(se))
        recalibrated = gated * se_weights
        
        out = self.fc_out(self.dropout(recalibrated))
        return res + out


class _CosineSimilarityClassifierHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int, scale: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x_norm, w_norm)


class _ResidualGatedNetModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.15,
        cos_scale: float = 18.0
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([
            _GatedFeatureBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = _CosineSimilarityClassifierHead(hidden_dim, n_classes, scale=cos_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.head(h)


class ResidualGatedFeatureClassifier(BaseEstimator, ClassifierMixin):
    """
    Residual Gated Feature Classifier (RGFN).
    
    A deep tabular classification network combining Feature-Gated Residual Blocks,
    Squeeze-and-Excitation channel gating, and Hyperspherical Cosine Angle Classification.
    
    Strictly NO Decision Trees. Strictly NO Linear/Logistic Regressions.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.15,
        cos_scale: float = 18.0,
        learning_rate: float = 0.003,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 140,
        device: Optional[str] = None,
        random_state: int = 42
    ):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.cos_scale = cos_scale
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y)
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if self.device is None:
            self._target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._target_device = torch.device(self.device)

        self.network_ = _ResidualGatedNetModule(
            input_dim=n_features,
            n_classes=self.n_classes_,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            cos_scale=self.cos_scale
        ).to(self._target_device)

        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y_indices, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        self.network_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._target_device)
                batch_y = batch_y.to(self._target_device)

                optimizer.zero_grad()
                logits = self.network_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), max_norm=2.0)
                optimizer.step()

            scheduler.step()

        self.feature_importances_ = self._compute_feature_attributions(X)
        return self

    def _compute_feature_attributions(self, X: np.ndarray) -> np.ndarray:
        self.network_.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(self._target_device)
        logits = self.network_(tensor_x)
        max_logits, _ = torch.max(logits, dim=1)
        grad_outputs = torch.ones_like(max_logits)
        
        grads = torch.autograd.grad(
            outputs=max_logits,
            inputs=tensor_x,
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False
        )[0]
        
        attributions = torch.mean(torch.abs(grads), dim=0).detach().cpu().numpy()
        attributions = attributions / (np.sum(attributions) + 1e-8)
        return attributions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['network_', 'classes_'])
        X = check_array(X)
        self.network_.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self._target_device)
            logits = self.network_(tensor_x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
