"""
Loss Functions and Adversarial Training Utilities for Uncertainty Estimation.
Includes Heteroscedastic Gaussian Negative Log-Likelihood (NLL) and Fast Gradient Sign Perturbations (FGSM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def heteroscedastic_nll_loss(y_true, mu, var):
    """
    Negative Log-Likelihood for Gaussian Distribution with predicted Mean and Variance.
    Formula: 0.5 * log(var) + 0.5 * (y - mu)^2 / var
    """
    loss = 0.5 * torch.log(var) + 0.5 * torch.square(y_true - mu) / var
    return torch.mean(loss)


def generate_adversarial_sample(model, x, y, epsilon=0.01, is_regression=True):
    """
    Generates adversarial samples using the Fast Gradient Sign Method (FGSM).
    x_adv = x + epsilon * sign(grad_x(Loss))
    Improves predictive uncertainty calibration by smoothing predictions in input neighborhood.
    """
    x_adv = x.clone().detach().requires_grad_(True)
    
    if is_regression:
        mu, var = model(x_adv)
        loss = heteroscedastic_nll_loss(y, mu, var)
    else:
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        
    loss.backward()
    
    with torch.no_grad():
        gradient_sign = x_adv.grad.sign()
        x_perturbed = x + epsilon * gradient_sign
        
    return x_perturbed.detach()


def brier_score(probs, targets, num_classes=10):
    """
    Computes multi-class Brier Score (Strictly proper scoring rule for probability calibration).
    BS = 1/N sum_i sum_c (p_ic - y_ic)^2
    """
    if targets.dim() == 1:
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    else:
        targets_one_hot = targets.float()
    return torch.mean(torch.sum(torch.square(probs - targets_one_hot), dim=-1)).item()
