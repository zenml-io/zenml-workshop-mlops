"""
Example 5: Metadata Tracking
============================

This example demonstrates how to track metadata in ZenML with two different
pipeline runs using different parameters to show comparative metrics.
"""

import time
import pandas as pd
import numpy as np
from typing import Annotated, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from zenml import step, pipeline, log_metadata
from zenml.client import Client


@step
def generate_data(
    n_samples: int = 1000,
    noise_level: float = 0.1,
    random_state: int = 42
) -> Annotated[pd.DataFrame, "training_data"]:
    """Generate synthetic regression data."""
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, 5)
    
    # Create target with varying noise
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, noise_level, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    # Log metadata about data generation
    log_metadata(
        metadata={
            "data_generation": {
                "n_samples": n_samples,
                "noise_level": noise_level,
                "random_state": random_state,
                "target_mean": float(df['target'].mean()),
                "target_std": float(df['target'].std())
            }
        },
        infer_artifact=True
    )
    
    print(f"Generated {n_samples} samples with noise level {noise_level}")
    return df


@step
def train_model(
    data: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10
) -> Annotated[RandomForestRegressor, "trained_model"]:
    """Train a Random Forest model."""
    
    # Prepare features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    start_time = time.time()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    
    # Log comprehensive metadata
    log_metadata(
        metadata={
            "model_performance": {
                "mse": float(mse),
                "r2_score": float(r2),
                "training_time": float(training_time),
                "test_samples": len(X_test)
            },
            "model_config": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "model_type": "RandomForestRegressor"
            },
            "feature_importance": {
                k: float(v) for k, v in feature_importance.items()
            }
        },
        infer_artifact=True
    )
    
    # Also log metadata to the step
    log_metadata(
        metadata={
            "training_summary": {
                "r2_score": float(r2),
                "mse": float(mse),
                "model_complexity": n_estimators * max_depth,
                "training_time": float(training_time)
            }
        }
    )
    
    print(f"Model trained - R2: {r2:.4f}, MSE: {mse:.4f}, Time: {training_time:.2f}s")
    return model


@pipeline
def metadata_tracking_pipeline(
    n_samples: int = 1000,
    noise_level: float = 0.1,
    n_estimators: int = 100,
    max_depth: int = 10
):
    """Pipeline for ML experiments with metadata tracking."""
    # Generate data
    data = generate_data(
        n_samples=n_samples,
        noise_level=noise_level
    )
    
    # Train model
    model = train_model(
        data=data,
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    
    return model


def compare_experiments():
    """Compare metadata from different experiment runs."""
    client = Client()
    
    print("\n=== Comparing Experiment Runs ===")
    
    # Get recent runs
    runs = client.get_pipeline(name_id_or_prefix="metadata_tracking_pipeline").runs
    
    if len(runs) < 2:
        print("Need at least 2 runs to compare. Run the pipeline with different parameters.")
        return
    
    print(f"Found {len(runs)} runs to compare:")
    
    for i, run in enumerate(runs[:3]):  # Show top 3 runs
        print(f"\nRun {i+1}: {run.id}")
        
        # Get training step metadata
        if "train_model" in run.steps:
            train_step = run.steps["train_model"]
            if hasattr(train_step, 'run_metadata') and train_step.run_metadata:
                summary = train_step.run_metadata.get("training_summary")
                print(summary)
                if summary:
                    print(f"  R2 Score: {summary.get('r2_score', 'N/A')}")
                    print(f"  MSE: {summary.get('mse', 'N/A')}")
                    print(f"  Training Time: {summary.get('training_time', 'N/A')}s")


if __name__ == "__main__":
    print("=== ML Experiment with Metadata Tracking ===")

    # Run 1: Baseline experiment
    print("\n--- Running Experiment 1: Baseline ---")
    result1 = metadata_tracking_pipeline(
        n_samples=1000,
        noise_level=0.1,
        n_estimators=50,
        max_depth=5
    )
    
    # Run 2: More complex model
    print("\n--- Running Experiment 2: Complex Model ---")
    result2 = metadata_tracking_pipeline(
        n_samples=1000,
        noise_level=0.1,
        n_estimators=200,
        max_depth=15
    )
    
    # Run 3: Noisy data experiment
    print("\n--- Running Experiment 3: Noisy Data ---")
    result3 = metadata_tracking_pipeline(
        n_samples=1000,
        noise_level=0.5,
        n_estimators=100,
        max_depth=10
    )
    
    # Compare experiments
    compare_experiments()
    
    print("Check the ZenML dashboard to see metadata comparison across runs.")