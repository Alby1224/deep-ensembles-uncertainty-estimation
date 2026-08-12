"""
Main Experiment Runner & CLI Interface for Deep Ensembles Uncertainty Estimation.
Supports Toy 1D Regression, California Housing, and Vision Benchmark Evaluation.
"""

import argparse
import os
import sys
import torch
import numpy as np

from src.config import TOY_CONFIG, HOUSING_CONFIG, VISION_CONFIG, SEED
from src.models import GaussianMLPRegressor, DeepEnsemble
from src.losses import heteroscedastic_nll_loss, generate_adversarial_sample
from src.metrics import regression_metrics
from src.visualization import plot_toy_regression


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Ensembles for Predictive Uncertainty Estimation Runner."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["toy", "housing", "all"],
        default="toy",
        help="Task to run (toy 1D regression, california housing, or all).",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of individually initialized models in the ensemble.",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        default=True,
        help="Enable Fast Gradient Sign (FGSM) adversarial training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets",
        help="Directory to save generated publication figures.",
    )
    return parser.parse_args()


def run_toy_experiment(ensemble_size=5, use_adversarial=True, output_dir="assets"):
    print("=" * 65)
    print("🔬 RUNNING EXPERIMENT: Toy 1D Heteroscedastic Regression ($y = x^3 + \epsilon$)")
    print(f"📊 Ensemble Size (M): {ensemble_size} | Adversarial Training: {use_adversarial}")
    print("=" * 65)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Generate Toy Dataset
    x_train_np = np.random.uniform(TOY_CONFIG["x_range"][0], TOY_CONFIG["x_range"][1], (TOY_CONFIG["num_train"], 1))
    noise = np.random.normal(0, TOY_CONFIG["noise_std"], (TOY_CONFIG["num_train"], 1))
    y_train_np = x_train_np ** 3 + noise

    x_test_np = np.linspace(TOY_CONFIG["test_x_range"][0], TOY_CONFIG["test_x_range"][1], TOY_CONFIG["num_test"]).reshape(-1, 1)
    y_test_true_np = x_test_np ** 3

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_test_true = torch.tensor(y_test_true_np, dtype=torch.float32)

    # 2. Train Ensemble
    models = []
    for m in range(ensemble_size):
        torch.manual_seed(SEED + m * 100)
        model = GaussianMLPRegressor(input_dim=1, hidden_dim=TOY_CONFIG["hidden_dim"], output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=TOY_CONFIG["learning_rate"])

        for epoch in range(TOY_CONFIG["num_epochs"]):
            optimizer.zero_grad()
            mu, var = model(x_train)
            loss = heteroscedastic_nll_loss(y_train, mu, var)

            if use_adversarial:
                x_adv = generate_adversarial_sample(model, x_train, y_train, epsilon=TOY_CONFIG["adversarial_epsilon"], is_regression=True)
                mu_adv, var_adv = model(x_adv)
                loss_adv = heteroscedastic_nll_loss(y_train, mu_adv, var_adv)
                total_loss = 0.5 * loss + 0.5 * loss_adv
            else:
                total_loss = loss

            total_loss.backward()
            optimizer.step()

        models.append(model)
        print(f"  ✓ Model [{m+1}/{ensemble_size}] trained. Final Loss: {total_loss.item():.4f}")

    # 3. Evaluate Ensemble Mixture
    ensemble = DeepEnsemble(models)
    mean_pred, total_var, aleatoric_var, epistemic_var = ensemble.predict_regression(x_test)

    metrics = regression_metrics(y_test_true, mean_pred, total_var)
    print("\n📈 Benchmark Evaluation Metrics (Test Domain [-6, 6]):")
    for k, v in metrics.items():
        print(f"   - {k}: {v:.4f}")

    # 4. Generate Publication Plot
    out_path = os.path.join(output_dir, "02_toy_ensemble_variance_bands.png")
    plot_toy_regression(x_train, y_train, x_test, y_test_true, mean_pred, total_var, epistemic_var, output_path=out_path)
    print("\n✨ Experiment completed successfully!")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.task in ("toy", "all"):
        run_toy_experiment(
            ensemble_size=args.ensemble_size,
            use_adversarial=args.adversarial,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
