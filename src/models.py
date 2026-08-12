"""
Neural Network Architectures for Predictive Uncertainty Estimation.
Includes Gaussian MLP Regressor with Dual Heads, CNN Classifier, and Deep Ensemble Container.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianMLPRegressor(nn.Module):
    """
    Multi-Layer Perceptron predicting both Mean and Heteroscedastic Variance.
    Output: (mu, sigma^2) where sigma^2 > 0 enforced via Softplus.
    """
    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.var_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mu_head(h)
        # Softplus ensures strictly positive variance + epsilon for numerical stability
        var = F.softplus(self.var_head(h)) + 1e-6
        return mu, var


class CNNClassifier(nn.Module):
    """
    Convolutional Neural Network for SVHN / CIFAR-10 Classification.
    Supports standard training and Monte Carlo (MC) Dropout.
    """
    def __init__(self, in_channels=3, num_classes=10, dropout_rate=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        f = self.features(x)
        f_flat = f.view(f.size(0), -1)
        logits = self.classifier(f_flat)
        return logits


class DeepEnsemble:
    """
    Deep Ensemble Container combining M randomly initialized neural networks.
    Aggregates individual predictions into Gaussian Mixture (Regression) or Mean Softmax (Classification).
    """
    def __init__(self, models):
        self.models = models
        self.ensemble_size = len(models)

    def predict_regression(self, x_tensor):
        """
        Computes mixture mean, aleatoric uncertainty, epistemic uncertainty, and total variance.
        Formula:
          mu_* = 1/M sum(mu_m)
          var_total = 1/M sum(sigma_m^2 + mu_m^2) - mu_*^2
          var_aleatoric = 1/M sum(sigma_m^2)
          var_epistemic = 1/M sum((mu_m - mu_*)^2)
        """
        self.eval()
        mus, vars_ = [], []
        with torch.no_grad():
            for model in self.models:
                m, v = model(x_tensor)
                mus.append(m)
                vars_.append(v)

        mus = torch.stack(mus, dim=0)   # [M, N, 1]
        vars_ = torch.stack(vars_, dim=0) # [M, N, 1]

        ensemble_mean = torch.mean(mus, dim=0)
        aleatoric_var = torch.mean(vars_, dim=0)
        epistemic_var = torch.var(mus, dim=0, unbiased=False)
        total_var = aleatoric_var + epistemic_var

        return ensemble_mean, total_var, aleatoric_var, epistemic_var

    def predict_classification(self, x_tensor):
        """
        Computes ensemble mean class probabilities and predictive entropy.
        """
        self.eval()
        probs_list = []
        with torch.no_grad():
            for model in self.models:
                logits = model(x_tensor)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs)

        probs_stack = torch.stack(probs_list, dim=0)  # [M, N, C]
        ensemble_probs = torch.mean(probs_stack, dim=0) # [N, C]
        
        # Predictive Entropy: H(p) = -sum(p * log(p))
        entropy = -torch.sum(ensemble_probs * torch.log(ensemble_probs + 1e-12), dim=-1)
        return ensemble_probs, entropy

    def eval(self):
        for m in self.models:
            m.eval()

    def train(self):
        for m in self.models:
            m.train()
