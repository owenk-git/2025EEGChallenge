"""
Evaluation metrics for EEG Challenge 2025

Competition Metric: Normalized RMSE (NRMSE)
Final Score: 0.3 * C1_NRMSE + 0.7 * C2_NRMSE

Additional metrics for robust evaluation:
- Pearson Correlation: Linear relationship strength
- R² Score: Proportion of variance explained
- CCC: Concordance Correlation Coefficient (agreement)
- MAPE: Mean Absolute Percentage Error
"""

import torch
import numpy as np
from typing import Union
from scipy.stats import pearsonr


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


def pearson_correlation(predictions: Union[torch.Tensor, np.ndarray],
                       targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Pearson Correlation Coefficient

    Measures linear relationship strength between predictions and targets.
    Common metric in neuroscience research.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Pearson r ∈ [-1, 1], higher is better
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    r, p_value = pearsonr(predictions, targets)
    return r


def r2_score(predictions: Union[torch.Tensor, np.ndarray],
            targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate R² Score (Coefficient of Determination)

    Measures proportion of variance explained by the model.
    R² = 1 - (SS_res / SS_tot)

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        R² ∈ (-∞, 1], higher is better
        1.0 = perfect prediction
        0.0 = predictions = mean(targets)
        < 0 = worse than predicting mean
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot == 0:
        return float('-inf')

    r2 = 1 - (ss_res / ss_tot)
    return r2


def concordance_correlation_coefficient(predictions: Union[torch.Tensor, np.ndarray],
                                        targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Lin's Concordance Correlation Coefficient (CCC)

    Gold standard for agreement between two methods.
    Combines precision (Pearson correlation) and accuracy (bias).

    Formula: CCC = (2 * cov) / (var_pred + var_true + (mean_pred - mean_true)²)

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        CCC ∈ [-1, 1], higher is better
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    mean_pred = np.mean(predictions)
    mean_true = np.mean(targets)
    var_pred = np.var(predictions)
    var_true = np.var(targets)
    covariance = np.mean((predictions - mean_pred) * (targets - mean_true))

    ccc = (2 * covariance) / (var_pred + var_true + (mean_pred - mean_true) ** 2)
    return ccc


def mape(predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE)

    Interpretable metric in percentage terms.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE in percentage (e.g., 5.0 means 5% error)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    # Avoid division by zero
    mape_val = np.mean(np.abs((targets - predictions) / (np.abs(targets) + epsilon))) * 100
    return mape_val


def compute_all_metrics(predictions: Union[torch.Tensor, np.ndarray],
                       targets: Union[torch.Tensor, np.ndarray]) -> dict:
    """
    Compute basic evaluation metrics (backward compatibility)

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Dictionary with basic metrics
    """
    return {
        'nrmse': normalized_rmse(predictions, targets),
        'rmse': rmse(predictions, targets),
        'mae': mae(predictions, targets),
    }


def compute_comprehensive_metrics(predictions: Union[torch.Tensor, np.ndarray],
                                  targets: Union[torch.Tensor, np.ndarray]) -> dict:
    """
    Compute comprehensive evaluation metrics

    Includes all metrics for robust model evaluation:
    - NRMSE: Competition metric
    - RMSE, MAE: Basic error metrics
    - Pearson r: Linear relationship
    - R²: Variance explained
    - CCC: Agreement (gold standard)
    - MAPE: Percentage error

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Dictionary with all metrics
    """
    return {
        # Competition metric
        'nrmse': normalized_rmse(predictions, targets),

        # Basic error metrics
        'rmse': rmse(predictions, targets),
        'mae': mae(predictions, targets),

        # Correlation and agreement
        'pearson_r': pearson_correlation(predictions, targets),
        'r2': r2_score(predictions, targets),
        'ccc': concordance_correlation_coefficient(predictions, targets),

        # Percentage error
        'mape': mape(predictions, targets),
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

    print("="*70)
    print("Basic Metrics:")
    print("="*70)
    metrics = compute_all_metrics(predictions, targets)
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value:.4f}")

    print("\n" + "="*70)
    print("Comprehensive Metrics:")
    print("="*70)
    comp_metrics = compute_comprehensive_metrics(predictions, targets)
    for name, value in comp_metrics.items():
        print(f"  {name.upper()}: {value:.4f}")

    print("\n" + "="*70)
    print("Example Competition Score:")
    print("="*70)
    c1_nrmse = 1.45
    c2_nrmse = 1.01
    combined = combined_challenge_score(c1_nrmse, c2_nrmse)
    print(f"  C1 NRMSE: {c1_nrmse:.3f}")
    print(f"  C2 NRMSE: {c2_nrmse:.3f}")
    print(f"  Combined: {combined:.3f} (0.3×C1 + 0.7×C2)")
    print(f"  SOTA Target: 0.978")
