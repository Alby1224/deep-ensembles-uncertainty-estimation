"""
Evaluation Metrics for Uncertainty Estimation and Calibration.
Includes Expected Calibration Error (ECE), OOD AUROC, Multi-Class Brier Score, and Prediction Interval Coverage.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error


def compute_ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error (ECE).
    Measures the difference between model confidence and empirical accuracy across confidence bins.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_ood_auroc(in_entropy, ood_entropy):
    """
    Computes Area Under ROC Curve (AUROC) for distinguishing In-Distribution vs Out-of-Distribution
    based on predictive entropy.
    """
    if isinstance(in_entropy, torch.Tensor):
        in_entropy = in_entropy.cpu().numpy()
    if isinstance(ood_entropy, torch.Tensor):
        ood_entropy = ood_entropy.cpu().numpy()

    y_true = np.concatenate([np.zeros_like(in_entropy), np.ones_like(ood_entropy)])
    scores = np.concatenate([in_entropy, ood_entropy])
    
    return float(roc_auc_score(y_true, scores))


def regression_metrics(y_true, mu, var):
    """
    Computes RMSE, Gaussian NLL, and 95% Confidence Interval Coverage (PICP).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.cpu().numpy()
    if isinstance(var, torch.Tensor):
        var = var.cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_true, mu))
    nll = 0.5 * np.mean(np.log(var) + ((y_true - mu) ** 2) / var)
    
    std = np.sqrt(var)
    lower_95 = mu - 1.96 * std
    upper_95 = mu + 1.96 * std
    picp_95 = np.mean((y_true >= lower_95) & (y_true <= upper_95))

    return {
        "RMSE": float(rmse),
        "NLL": float(nll),
        "PICP_95": float(picp_95),
    }
