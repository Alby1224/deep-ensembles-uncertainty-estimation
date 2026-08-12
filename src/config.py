"""
Centralized Configuration and Hyperparameters.
Defines random seeds, dataset parameters, ensemble sizes, and training configs.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
DOCS_DIR = PROJECT_ROOT / "docs"

# Reproducibility
SEED = 42

# Toy Regression Config
TOY_CONFIG = {
    "num_train": 20,
    "x_range": (-4.0, 4.0),
    "test_x_range": (-6.0, 6.0),
    "num_test": 200,
    "noise_std": 3.0,
    "ensemble_size": 5,
    "hidden_dim": 100,
    "num_epochs": 400,
    "learning_rate": 0.05,
    "adversarial_epsilon": 0.01,
}

# California Housing Regression Config
HOUSING_CONFIG = {
    "test_size": 0.2,
    "ensemble_size": 5,
    "hidden_dim": 128,
    "num_epochs": 200,
    "batch_size": 64,
    "learning_rate": 0.001,
}

# SVHN / CIFAR-10 Classification Config
VISION_CONFIG = {
    "batch_size": 128,
    "num_classes": 10,
    "ensemble_size": 5,
    "num_epochs": 20,
    "learning_rate": 0.001,
    "adversarial_epsilon": 0.01,
    "mc_dropout_samples": 20,
    "dropout_rate": 0.2,
}

# Visualization Aesthetics
VIZ_CONFIG = {
    "style": "whitegrid",
    "palette": "deep",
    "figure_dpi": 300,
    "font_scale": 1.1,
}
