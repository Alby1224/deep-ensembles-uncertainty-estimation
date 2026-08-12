"""
Publication-Quality Visualization Routines for Uncertainty Estimation.
Generates paper-style confidence bands, OOD entropy distributions, and calibration diagrams.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.config import VIZ_CONFIG


def apply_publication_style():
    sns.set_theme(style=VIZ_CONFIG["style"])
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["figure.dpi"] = VIZ_CONFIG["figure_dpi"]


def plot_toy_regression(x_train, y_train, x_test, y_true, mu, total_var, epistemic_var=None, output_path="assets/02_toy_ensemble_variance_bands.png"):
    """
    Plots Toy cubic regression with ±3 sigma confidence intervals showcasing epistemic explosion outside training domain.
    """
    apply_publication_style()
    
    if isinstance(x_train, torch.Tensor): x_train = x_train.cpu().numpy()
    if isinstance(y_train, torch.Tensor): y_train = y_train.cpu().numpy()
    if isinstance(x_test, torch.Tensor): x_test = x_test.cpu().numpy()
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().numpy()
    if isinstance(mu, torch.Tensor): mu = mu.cpu().numpy()
    if isinstance(total_var, torch.Tensor): total_var = total_var.cpu().numpy()

    std_total = np.sqrt(total_var).flatten()
    mu = mu.flatten()
    x_test = x_test.flatten()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ground truth
    ax.plot(x_test, y_true, "k--", label="Ground Truth $y = x^3$", alpha=0.7, linewidth=1.5)

    # Ensemble Mean
    ax.plot(x_test, mu, color="#d95f02", label="Deep Ensemble Mean $\mu_*(x)$", linewidth=2.2)

    # Uncertainty Bands: 1, 2, 3 sigma
    ax.fill_between(x_test, mu - 3 * std_total, mu + 3 * std_total, color="#fc8d62", alpha=0.2, label="$\pm 3\sigma$ Total Uncertainty")
    ax.fill_between(x_test, mu - 1.96 * std_total, mu + 1.96 * std_total, color="#fc8d62", alpha=0.35, label="$\pm 1.96\sigma$ (95% CI)")

    # Training points
    ax.scatter(x_train, y_train, color="#7570b3", s=45, zorder=5, label="Observed Training Points ($x \in [-4, 4]$)")

    # Highlight epistemic extrapolation region
    ax.axvspan(-6.0, -4.0, color="gray", alpha=0.12, hatch="//")
    ax.axvspan(4.0, 6.0, color="gray", alpha=0.12, hatch="//", label="Unseen Region (High Epistemic Uncertainty)")

    ax.set_title("Deep Ensemble Predictive Uncertainty on $y = x^3 + \epsilon$", fontweight="bold", pad=12)
    ax.set_xlabel("Input $x$")
    ax.set_ylabel("Target $y$")
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-120, 120)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=VIZ_CONFIG["figure_dpi"])
    plt.close(fig)
    print(f"✓ Figure saved to: {output_path}")


def plot_entropy_distributions(in_entropy, ood_entropy, in_label="SVHN (In-Dist)", ood_label="CIFAR-10 (OOD)", output_path="assets/04_svhn_cifar10_ood_entropy_distribution.png"):
    """
    Plots Kernel Density Estimates / Histograms of predictive entropy for In-Distribution vs OOD datasets.
    """
    apply_publication_style()

    if isinstance(in_entropy, torch.Tensor): in_entropy = in_entropy.cpu().numpy()
    if isinstance(ood_entropy, torch.Tensor): ood_entropy = ood_entropy.cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.kdeplot(in_entropy, color="#1b9e77", fill=True, alpha=0.4, label=f"{in_label} (Low Entropy / High Confidence)", ax=ax)
    sns.kdeplot(ood_entropy, color="#d95f02", fill=True, alpha=0.4, label=f"{ood_label} (High Entropy / Uncertainty)", ax=ax)

    ax.set_title("Predictive Entropy Distribution: In-Distribution vs Out-of-Distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Predictive Entropy $H(p(y|x))$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=VIZ_CONFIG["figure_dpi"])
    plt.close(fig)
    print(f"✓ Figure saved to: {output_path}")
