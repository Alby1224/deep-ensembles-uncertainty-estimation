# 📊 Dataset Specifications & Download Instructions

This project benchmarks **Deep Ensembles for Predictive Uncertainty Estimation** across synthetic and real-world benchmark datasets:

---

## 1. 1D Synthetic Toy Dataset (Heteroscedastic Regression)
- **Mathematical Definition:** $y = x^3 + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2(x))$ and $x \in [-4, 4]$.
- **Purpose:** Disentangling **aleatoric uncertainty** (inherent data noise) from **epistemic uncertainty** (model uncertainty in unseen regions $x < -4$ and $x > 4$).
- **Generation:** Generated programmatically via `src/models.py` and `run_experiments.py`.

---

## 2. California Housing Dataset
- **Domain:** Tabular socioeconomic and geographic housing metrics.
- **Source:** Scikit-Learn `fetch_california_housing()`.
- **Target:** Median house values (continuous regression).
- **Download:** Automatically fetched and cached by Scikit-Learn during pipeline execution.

---

## 3. Street View House Numbers (SVHN)
- **Domain:** 10-class real-world digit classification images ($32 \times 32 \times 3$).
- **Source:** `torchvision.datasets.SVHN(root='data/', split='train'/'test', download=True)`.
- **Role:** Primary in-distribution image benchmark for predictive entropy, calibration (Brier Score), and NLL.

---

## 4. CIFAR-10 (Out-of-Distribution Dataset)
- **Domain:** 10-class natural object images ($32 \times 32 \times 3$).
- **Source:** `torchvision.datasets.CIFAR10(root='data/', train=False, download=True)`.
- **Role:** Pure Out-of-Distribution (OOD) test set against models trained on SVHN to evaluate confidence degradation and entropy spikes.
