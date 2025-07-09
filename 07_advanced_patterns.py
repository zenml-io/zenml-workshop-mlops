"""
Example 7: Configuration Validation Pattern
===========================================

This example demonstrates how to use a configuration step to validate and 
standardize parameters for downstream steps. This pattern is especially useful
for run templates where you want centralized parameter validation.
"""

import pandas as pd
import numpy as np
from typing import Annotated, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from zenml import step, pipeline, log_metadata


@step
def validate_config(
    n_samples: int = 1000,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    noise_level: float = 0.1,
    test_size: float = 0.2
) -> Tuple[
    Annotated[int, "n_samples"],
    Annotated[int, "n_estimators"], 
    Annotated[Optional[int], "max_depth"],
    Annotated[float, "noise_level"],
    Annotated[float, "test_size"]
]:
    """
    Configuration validation step that validates parameters and returns them individually.
    
    This pattern is ideal for run templates as it centralizes parameter validation
    and provides individual validated parameters to downstream steps.
    """
    
    # Simple validation
    if n_samples < 100 or n_samples > 5000:
        raise ValueError(f"n_samples must be between 100 and 5000, got {n_samples}")
    
    if n_estimators < 10 or n_estimators > 500:
        raise ValueError(f"n_estimators must be between 10 and 500, got {n_estimators}")
    
    if max_depth is not None and (max_depth < 1 or max_depth > 20):
        raise ValueError(f"max_depth must be between 1 and 20, got {max_depth}")
    
    if noise_level < 0 or noise_level > 1:
        raise ValueError(f"noise_level must be between 0 and 1, got {noise_level}")
    
    if test_size <= 0 or test_size >= 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    # Log validation metadata
    log_metadata(
        metadata={
            "validation_status": "passed",
            "validated_params": {
                "n_samples": n_samples,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "noise_level": noise_level,
                "test_size": test_size
            }
        }
    )
    
    print(f"✅ Config validated: {n_samples} samples, {n_estimators} trees, noise={noise_level}")
    
    # Return individual validated parameters
    return n_samples, n_estimators, max_depth, noise_level, test_size


@step
def generate_data(n_samples: int, noise_level: float) -> Annotated[pd.DataFrame, "dataset"]:
    """Generate synthetic data using validated parameters."""
    np.random.seed(42)
    
    # Generate features and target
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, noise_level, n_samples)
    
    data = pd.DataFrame({
        'feature1': X[:, 0],
        'feature2': X[:, 1], 
        'feature3': X[:, 2],
        'target': y
    })
    
    print(f"Generated {len(data)} samples with noise level {noise_level}")
    return data


@step
def train_model(
    data: pd.DataFrame,
    n_estimators: int,
    max_depth: Optional[int],
    test_size: float
) -> Annotated[dict, "results"]:
    """Train model using validated parameters."""
    
    # Prepare data
    X = data[['feature1', 'feature2', 'feature3']]
    y = data['target']
    
    # Split data
    split_idx = int(len(data) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'mse': float(mse),
        'r2_score': float(r2),
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }
    
    # Log training metadata
    log_metadata(
        metadata={
            "training_results": {
                "mse": float(mse),
                "r2_score": float(r2),
                "n_estimators": n_estimators,
                "max_depth": max_depth
            }
        },
        infer_artifact=True
    )
    
    print(f"Model trained - R2: {r2:.4f}, MSE: {mse:.4f}")
    return results


@pipeline
def config_validated_pipeline(
    n_samples: int = 1000,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    noise_level: float = 0.1,
    test_size: float = 0.2
):
    """
    Pipeline demonstrating configuration validation pattern.
    
    The first step validates all parameters and returns them as individual
    values that downstream steps can use directly.
    """
    
    # Step 1: Validate configuration and get individual parameters
    (
        validated_n_samples,
        validated_n_estimators,
        validated_max_depth,
        validated_noise_level,
        validated_test_size
    ) = validate_config(
        n_samples=n_samples,
        n_estimators=n_estimators,
        max_depth=max_depth,
        noise_level=noise_level,
        test_size=test_size
    )
    
    # Step 2: Generate data using validated parameters
    data = generate_data(
        n_samples=validated_n_samples,
        noise_level=validated_noise_level
    )
    
    # Step 3: Train model using validated parameters
    results = train_model(
        data=data,
        n_estimators=validated_n_estimators,
        max_depth=validated_max_depth,
        test_size=validated_test_size
    )
    
    return results


if __name__ == "__main__":
    """
    Configuration Validation Pattern:
    
    1. ✅ Centralized parameter validation in first step
    2. ✅ Returns individual validated parameters as tuple
    3. ✅ Downstream steps receive validated native types (int, float, etc.)
    4. ✅ Perfect for run templates with parameter validation
    5. ✅ Single point to enforce business rules
    """
    
    print("=== Configuration Validation Demo ===")
    
    # Valid configuration
    print("\n1. Valid configuration:")
    config_validated_pipeline(
        n_samples=1500,
        n_estimators=150,
        max_depth=8,
        noise_level=0.1
    )
    