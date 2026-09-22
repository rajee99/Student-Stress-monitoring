"""
Novel Non-Tree, Non-Regression Machine Learning Classifiers.

Designed and developed from first principles:
1. KernelManifoldAttentionClassifier (KMAC):
   - Non-parametric metric prototype attention classifier.
   - Learns sub-manifold prototypes per class with adaptive feature metric weighting.
   - Employs hybrid RBF-Laplacian distance attention with temperature scaling.
   - Pure geometric and manifold attention (Zero Trees, Zero Regression).

2. ResidualGatedFeatureClassifier (RGFN):
   - Tabular Deep Residual Gated Neural Network with Squeeze-and-Excitation channel gating.
   - Uses Hyperspherical Cosine Similarity classification head (CosFace/ArcSoftmax).
   - Fully regularized with LayerNorm, Dropout, and AdamW + Cosine Annealing.
   - Pure neural representation and angular classification (Zero Trees, Zero Regression).
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
# MODEL 1: KERNEL MANIFOLD ATTENTION CLASSIFIER (KMAC)
# =============================================================================

class KernelManifoldAttentionClassifier(BaseEstimator, ClassifierMixin):
    """
    Kernel Manifold Attention Classifier (KMAC).
    
    A geometric metric learning classifier that maps continuous feature spaces
    into multi-prototype Riemannian manifolds with learned feature relevance weights
    and temperature-scaled kernel attention.
    
    Zero Trees. Zero Regression.
    """

    def __init__(
        self,
        prototypes_per_class: int = 3,
        temperature: float = 0.5,
        kernel_hybrid_alpha: float = 0.6,
        learning_rate: float = 0.05,
        max_iter: int = 200,
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
        """Fit class manifold prototypes and optimize metric relevance weights."""
        X, y = check_X_y(X, y)
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        rng = np.random.RandomState(self.random_state)

        # 1. Initialize sub-cluster prototypes for each class
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

        self.prototypes_ = np.vstack(prototypes_list)  # Shape: (total_protos, n_features)
        self.proto_classes_ = np.array(proto_class_list, dtype=int)  # Shape: (total_protos,)
        
        # Prototype prior weights
        total_counts = np.array(proto_counts_list, dtype=float)
        self.proto_weights_ = total_counts / np.sum(total_counts)

        # 2. Learn diagonal metric weights (feature attention) via Gradient Optimization
        # Metric weights w: initialized to 1.0 (uniform feature scale)
        log_w = np.zeros(n_features, dtype=float)
        m_t = np.zeros(n_features, dtype=float)
        v_t = np.zeros(n_features, dtype=float)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Mini-batch gradient descent for metric weights
        batch_size = min(64, n_samples)
        
        for it in range(1, self.max_iter + 1):
            perm = rng.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y_indices[perm]

            for b in range(0, n_samples, batch_size):
                xb = X_shuffled[b:b+batch_size]
                yb = y_shuffled[b:b+batch_size]
                w = np.exp(log_w)  # Ensure positivity

                # Forward pass: compute distances to prototypes
                # xb: (B, D), protos: (P, D)
                # Diff: (B, P, D)
                diff = xb[:, np.newaxis, :] - self.prototypes_[np.newaxis, :, :]
                
                # Weighted RBF distance (L2) & Laplacian distance (L1)
                d_l2 = np.sum((diff ** 2) * w[np.newaxis, np.newaxis, :], axis=2)  # (B, P)
                d_l1 = np.sum(np.abs(diff) * np.sqrt(w[np.newaxis, np.newaxis, :] + 1e-8), axis=2)  # (B, P)
                dist = self.kernel_hybrid_alpha * d_l2 + (1.0 - self.kernel_hybrid_alpha) * d_l1

                # Prototype attention logit
                proto_logits = -dist / max(self.temperature, 1e-4) + np.log(self.proto_weights_[np.newaxis, :] + 1e-8)
                
                # Class aggregation via LogSumExp
                class_logits = np.zeros((len(xb), self.n_classes_))
                for c in range(self.n_classes_):
                    c_mask = (self.proto_classes_ == c)
                    # LogSumExp over prototypes belonging to class c
                    max_logit = np.max(proto_logits[:, c_mask], axis=1, keepdims=True)
                    class_logits[:, c] = np.squeeze(
                        max_logit + np.log(np.sum(np.exp(proto_logits[:, c_mask] - max_logit), axis=1, keepdims=True) + 1e-8)
                    )

                # Softmax probabilities
                max_cl = np.max(class_logits, axis=1, keepdims=True)
                exp_logits = np.exp(class_logits - max_cl)
                probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-8)

                # Cross-entropy gradient w.r.t class logits: (probs - one_hot)
                y_one_hot = np.zeros_like(probs)
                y_one_hot[np.arange(len(yb)), yb] = 1.0
                grad_logits = (probs - y_one_hot) / len(xb)  # (B, C)

                # Backprop to proto_logits:
                grad_proto_logits = np.zeros_like(proto_logits)
                for c in range(self.n_classes_):
                    c_mask = (self.proto_classes_ == c)
                    # softmax within class prototypes
                    sub_logits = proto_logits[:, c_mask]
                    sub_exp = np.exp(sub_logits - np.max(sub_logits, axis=1, keepdims=True))
                    sub_attn = sub_exp / (np.sum(sub_exp, axis=1, keepdims=True) + 1e-8)
                    grad_proto_logits[:, c_mask] = grad_logits[:, c:c+1] * sub_attn

                # Backprop to distance: d(proto_logits)/d(dist) = -1 / temp
                grad_dist = grad_proto_logits * (-1.0 / max(self.temperature, 1e-4))  # (B, P)

                # Backprop to w:
                grad_w_l2 = np.sum(grad_dist[:, :, np.newaxis] * (diff ** 2) * self.kernel_hybrid_alpha, axis=(0, 1))
                grad_w_l1 = np.sum(
                    grad_dist[:, :, np.newaxis] * np.abs(diff) * (1.0 - self.kernel_hybrid_alpha) * (0.5 / np.sqrt(w + 1e-8)),
                    axis=(0, 1)
                )
                grad_w = grad_w_l2 + grad_w_l1 + self.l2_regularization * w
                
                # Chain rule for log_w: dL/d(log_w) = dL/dw * w
                grad_log_w = grad_w * w
                grad_log_w = np.clip(grad_log_w, -5.0, 5.0)

                # Adam optimizer step
                m_t = beta1 * m_t + (1 - beta1) * grad_log_w
                v_t = beta2 * v_t + (1 - beta2) * (grad_log_w ** 2)
                m_hat = m_t / (1.0 - beta1 ** it)
                v_hat = v_t / (1.0 - beta2 ** it)
                log_w -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        self.feature_weights_ = np.exp(log_w)
        # Normalize weights so mean is 1.0
        self.feature_weights_ = self.feature_weights_ / (np.mean(self.feature_weights_) + 1e-8)
        self.feature_importances_ = self.feature_weights_ / np.sum(self.feature_weights_)
        return self

    def _compute_class_logits(self, X: np.ndarray) -> np.ndarray:
        """Compute class manifold attention logits."""
        check_is_fitted(self, ['prototypes_', 'feature_weights_', 'proto_classes_'])
        X = check_array(X)
        w = self.feature_weights_
        
        # Diff: (N, P, D)
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
        """Calculate class predictive probabilities."""
        class_logits = self._compute_class_logits(X)
        max_cl = np.max(class_logits, axis=1, keepdims=True)
        exp_logits = np.exp(class_logits - max_cl)
        probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-8)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for given samples."""
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]


# =============================================================================
# MODEL 2: RESIDUAL GATED FEATURE NETWORK (RGFN)
# =============================================================================

class _GatedFeatureBlock(nn.Module):
    """Residual Gated Linear Unit with LayerNorm and Feature Channel Recalibration."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_signal = nn.Linear(hidden_dim, hidden_dim)
        self.fc_gate = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Squeeze-and-Excitation channel gating
        self.se_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.se_fc2 = nn.Linear(hidden_dim // 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        normed = self.norm(x)
        signal = F.silu(self.fc_signal(normed))
        gate = torch.sigmoid(self.fc_gate(normed))
        gated = signal * gate
        
        # SE channel recalibration
        se = F.relu(self.se_fc1(gated))
        se_weights = torch.sigmoid(self.se_fc2(se))
        recalibrated = gated * se_weights
        
        out = self.fc_out(self.dropout(recalibrated))
        return res + out


class _CosineSimilarityClassifierHead(nn.Module):
    """
    Hyperspherical Cosine Margin Classification Head.
    Computes angle between normalized embedding and normalized class prototypes.
    """
    def __init__(self, in_features: int, n_classes: int, scale: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize features & weights onto unit hypersphere
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity logits
        cos_sim = F.linear(x_norm, w_norm)
        return self.scale * cos_sim


class _ResidualGatedNetModule(nn.Module):
    """Complete Neural Network Architecture for RGFN."""
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
    
    Zero Trees. Zero Regression.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.15,
        cos_scale: float = 16.0,
        learning_rate: float = 0.003,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 120,
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
        """Fit the deep residual gated network."""
        X, y = check_X_y(X, y)
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Select execution device (GPU or CPU)
        if self.device is None:
            self._target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._target_device = torch.device(self.device)

        # Build PyTorch Module
        self.network_ = _ResidualGatedNetModule(
            input_dim=n_features,
            n_classes=self.n_classes_,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            cos_scale=self.cos_scale
        ).to(self._target_device)

        # DataLoader
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y_indices, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer & Cosine Annealing Scheduler
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        # Training loop
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

        # Compute gradient-based feature attribution for feature importance ranking
        self.feature_importances_ = self._compute_feature_attributions(X)
        return self

    def _compute_feature_attributions(self, X: np.ndarray) -> np.ndarray:
        """Compute mean absolute input gradients as feature importance measures."""
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
        """Calculate class posterior probabilities."""
        check_is_fitted(self, ['network_', 'classes_'])
        X = check_array(X)
        self.network_.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self._target_device)
            logits = self.network_(tensor_x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for given samples."""
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]
