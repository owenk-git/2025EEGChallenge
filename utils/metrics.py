"""
Evaluation metrics for EEG Challenge 2025

Competition Metric: Normalized RMSE (NRMSE)
Final Score: 0.3 * C1_NRMSE + 0.7 * C2_NRMSE
"""

import torch
import numpy as np
from typing import Union


def rmse(predictions: Union[torch.Tensor, np.ndarray],
         targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Root Mean Squared Error (RMSE)

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        RMSE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    mse = np.mean((predictions - targets) ** 2)
    return np.sqrt(mse)


def normalized_rmse(predictions: Union[torch.Tensor, np.ndarray],
                   targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Normalized RMSE (NRMSE)

    Formula: NRMSE = RMSE / std(targets)

    This is the competition metric.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        NRMSE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    rmse_val = rmse(predictions, targets)
    target_std = np.std(targets)

    # Avoid division by zero
    if target_std == 0:
        return float('inf')

    nrmse = rmse_val / target_std
    return nrmse


def mae(predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Mean Absolute Error (MAE)

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        MAE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    return np.mean(np.abs(predictions - targets))


def compute_all_metrics(predictions: Union[torch.Tensor, np.ndarray],
                       targets: Union[torch.Tensor, np.ndarray]) -> dict:
    """
    Compute all evaluation metrics

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Dictionary with all metrics
    """
    return {
        'nrmse': normalized_rmse(predictions, targets),
        'rmse': rmse(predictions, targets),
        'mae': mae(predictions, targets),
    }


def combined_challenge_score(c1_nrmse: float, c2_nrmse: float) -> float:
    """
    Calculate combined challenge score

    Formula: 0.3 * C1_NRMSE + 0.7 * C2_NRMSE

    Args:
        c1_nrmse: Challenge 1 NRMSE
        c2_nrmse: Challenge 2 NRMSE

    Returns:
        Combined score
    """
    return 0.3 * c1_nrmse + 0.7 * c2_nrmse


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    targets = np.random.randn(100)
    predictions = targets + np.random.randn(100) * 0.1

    metrics = compute_all_metrics(predictions, targets)
    print("Test Metrics:")
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value:.4f}")

    # Example competition scores
    c1_nrmse = 1.45
    c2_nrmse = 1.01
    combined = combined_challenge_score(c1_nrmse, c2_nrmse)
    print(f"\nExample Competition Score:")
    print(f"  C1 NRMSE: {c1_nrmse:.3f}")
    print(f"  C2 NRMSE: {c2_nrmse:.3f}")
    print(f"  Combined: {combined:.3f}")
