"""
ZenML Timeseries Forecasting Pipeline - Complete Solution
This pipeline trains models and immediately uses them for batch forecasting.
Designed for companies that forecast on many products (e.g., 100,000 products)
with batch processing for workshop demonstration (10 products in 2 batches).
"""

from zenml import ArtifactConfig, Model, log_metadata, pipeline, step
from zenml.types import HTMLString
from zenml.config import DockerSettings
from zenml.enums import ArtifactType

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Annotated
from datetime import timedelta

# Import utility functions for HTML report generation
from utils import generate_timeseries_exploration_report, generate_forecast_validation_report


@step
def generate_timeseries_data(n_products: int = 10) -> pd.DataFrame:
    """Generate synthetic timeseries data for forecasting demo.
    
    Creates data for multiple products with seasonal patterns and trend.
    
    Args:
        n_products: Number of products to generate data for
    
    Returns:
        DataFrame with timeseries data for multiple products
    """
    print(f" Generating timeseries data for {n_products} products...")
    
    # Generate 2 years of daily data
    date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    
    data = []
    np.random.seed(42)
    
    for product_id in range(1, n_products + 1):
        # Base demand with trend and seasonality
        base_demand = 100 + product_id * 10
        trend = np.linspace(0, 50, len(date_range))
        seasonality = 20 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365.25)
        weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(len(date_range)) / 7)
        noise = np.random.normal(0, 15, len(date_range))
        
        demand = base_demand + trend + seasonality + weekly_pattern + noise
        demand = np.maximum(demand, 0)  # Ensure non-negative demand
        
        for i, date in enumerate(date_range):
            data.append({
                'product_id': f'PROD_{product_id:03d}',
                'date': date,
                'demand': demand[i],
                'day_of_week': date.weekday(),
                'month': date.month,
                'quarter': date.quarter,
                'is_weekend': date.weekday() >= 5
            })
    
    df = pd.DataFrame(data)
    print(f" Generated {len(df)} records for {n_products} products")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    log_metadata({
        "n_products": n_products,
        "total_records": len(df),
        "date_range_start": str(df['date'].min()),
        "date_range_end": str(df['date'].max())
    }, infer_model=True)
    
    return df


@step
def explore_timeseries_data(df: pd.DataFrame) -> Annotated[HTMLString, "timeseries_exploration_report"]:
    """Create timeseries data exploration report with visualizations.

    Args:
        df: Timeseries DataFrame

    Returns:
        HTML report with timeseries visualizations
    """
    return generate_timeseries_exploration_report(df)


@step
def prepare_timeseries_features(
    df: pd.DataFrame,
    forecast_horizon: int = 30
) -> Tuple[
    Annotated[pd.DataFrame, "training_data"],
    Annotated[pd.DataFrame, "forecast_data"],
    Annotated[StandardScaler, "scaler"],
    Annotated[list, "feature_names"],
]:
    """Prepare timeseries data for training and forecasting.

    Args:
        df: Raw timeseries DataFrame
        forecast_horizon: Number of days to forecast

    Returns:
        Tuple of (training_data, forecast_data, scaler, feature_names)
    """
    print("🔧 Preparing timeseries features...")

    df_processed = df.copy()
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    
    # Create lag features for each product
    processed_data = []
    
    for product_id in df_processed['product_id'].unique():
        product_data = df_processed[df_processed['product_id'] == product_id].copy()
        product_data = product_data.sort_values('date')
        
        # Create lag features
        product_data['demand_lag_1'] = product_data['demand'].shift(1)
        product_data['demand_lag_7'] = product_data['demand'].shift(7)
        product_data['demand_lag_30'] = product_data['demand'].shift(30)
        
        # Rolling averages
        product_data['demand_rolling_7'] = product_data['demand'].rolling(7).mean()
        product_data['demand_rolling_30'] = product_data['demand'].rolling(30).mean()
        
        processed_data.append(product_data)
    
    df_processed = pd.concat(processed_data, ignore_index=True)
    
    # Remove rows with NaN values (due to lags)
    df_processed = df_processed.dropna()
    
    # Split into training and forecast periods
    cutoff_date = df_processed['date'].max() - timedelta(days=forecast_horizon)
    
    training_data = df_processed[df_processed['date'] <= cutoff_date].copy()
    forecast_data = df_processed[df_processed['date'] > cutoff_date].copy()
    
    # Define features for modeling
    feature_names = [
        'day_of_week', 'month', 'quarter', 'is_weekend',
        'demand_lag_1', 'demand_lag_7', 'demand_lag_30',
        'demand_rolling_7', 'demand_rolling_30'
    ]
    
    # Scale features
    scaler = StandardScaler()
    training_data[feature_names] = scaler.fit_transform(training_data[feature_names])
    forecast_data[feature_names] = scaler.transform(forecast_data[feature_names])
    
    print(f" Training data: {len(training_data)} records")
    print(f" Forecast data: {len(forecast_data)} records")
    print(f" Features: {len(feature_names)}")
    
    log_metadata({
        "training_records": len(training_data),
        "forecast_records": len(forecast_data),
        "n_features": len(feature_names),
        "forecast_horizon_days": forecast_horizon
    }, infer_model=True)
    
    return training_data, forecast_data, scaler, feature_names


@step
def train_forecast_model(
    training_data: pd.DataFrame, 
    feature_names: list,
    n_estimators: int = 100
) -> Annotated[RandomForestRegressor, ArtifactConfig(name="forecast_model", artifact_type=ArtifactType.MODEL,)]:
    """Train Random Forest model for demand forecasting.

    Args:
        training_data: Training dataset
        feature_names: List of feature column names
        n_estimators: Number of trees

    Returns:
        Trained Random Forest regression model
    """
    print(" Training demand forecasting model...")

    X_train = training_data[feature_names]
    y_train = training_data['demand']
    
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=42, 
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    print(" Demand forecasting model trained")
    
    # Log training metrics
    train_pred = rf_model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    log_metadata({
        "training_mae": float(train_mae),
        "training_r2": float(train_r2),
        "n_estimators": n_estimators,
        "model_type": "RandomForestRegressor"
    }, infer_model=True)
    
    print(f"Training MAE: {train_mae:.2f}, R²: {train_r2:.3f}")

    return rf_model


@step
def batch_predict(
    model: RandomForestRegressor,
    forecast_data: pd.DataFrame,
    feature_names: list,
    batch_size: int = 5
) -> Annotated[pd.DataFrame, "batch_predictions"]:
    """Generate batch predictions for multiple products.

    Args:
        model: Trained forecasting model
        forecast_data: Data to forecast on
        feature_names: List of feature column names
        batch_size: Number of products per batch

    Returns:
        DataFrame with predictions for all products
    """
    print(f"🔮 Generating batch predictions (batch size: {batch_size})...")
    
    products = forecast_data['product_id'].unique()
    n_batches = int(np.ceil(len(products) / batch_size))
    
    all_predictions = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(products))
        batch_products = products[start_idx:end_idx]
        
        print(f"  Processing batch {batch_idx + 1}/{n_batches}: {len(batch_products)} products")
        
        # Filter data for current batch
        batch_data = forecast_data[forecast_data['product_id'].isin(batch_products)].copy()
        
        # Make predictions
        X_batch = batch_data[feature_names]
        predictions = model.predict(X_batch)
        
        # Add predictions to data
        batch_data['predicted_demand'] = predictions
        batch_data['batch_id'] = batch_idx + 1
        
        all_predictions.append(batch_data)
    
    # Combine all batch predictions
    result_df = pd.concat(all_predictions, ignore_index=True)
    
    print(f" Completed {n_batches} batches, {len(result_df)} predictions generated")
    
    log_metadata({
        "total_predictions": len(result_df),
        "n_batches": n_batches,
        "batch_size": batch_size,
        "n_products": len(products)
    })
    
    return result_df


@step
def validate_predictions(
    predictions_df: pd.DataFrame
) -> Tuple[
    Annotated[dict, "validation_metrics"],
    Annotated[HTMLString, "validation_report"],
]:
    """Validate forecast predictions and create evaluation report.

    Args:
        predictions_df: DataFrame with actual and predicted demand

    Returns:
        Tuple of (validation_metrics, validation_report)
    """
    print(" Validating predictions...")
    
    # Calculate validation metrics
    actual = predictions_df['demand']
    predicted = predictions_df['predicted_demand']
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Per-product metrics
    product_metrics = []
    for product_id in predictions_df['product_id'].unique():
        product_data = predictions_df[predictions_df['product_id'] == product_id]
        product_mae = mean_absolute_error(product_data['demand'], product_data['predicted_demand'])
        product_r2 = r2_score(product_data['demand'], product_data['predicted_demand'])
        
        product_metrics.append({
            'product_id': product_id,
            'mae': product_mae,
            'r2': product_r2,
            'n_predictions': len(product_data)
        })
    
    validation_metrics = {
        'overall_mae': float(mae),
        'overall_mse': float(mse),
        'overall_rmse': float(rmse),
        'overall_r2': float(r2),
        'overall_mape': float(mape),
        'product_metrics': product_metrics,
        'n_predictions': len(predictions_df),
        'n_products': len(predictions_df['product_id'].unique())
    }
    
    print(f"  Validation Results:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.3f}")
    print(f"  MAPE: {mape:.1f}%")
    
    # Log validation metrics
    log_metadata(validation_metrics, infer_model=True)
    
    # Generate validation report
    validation_report = generate_forecast_validation_report(predictions_df, validation_metrics)
    
    return validation_metrics, validation_report


@pipeline(
    enable_cache=False,
    model=Model(
        name="TimeseriesForecastModel",
        description="End-to-end timeseries forecasting pipeline with batch prediction.",
    ),
    settings={
        "docker": DockerSettings(
            requirements="requirements.txt",
            python_package_installer="uv",
            apt_packages=["libgomp1"],
        ),
    },
)
def timeseries_forecast_pipeline(
    n_products: int = 10,
    n_estimators: int = 100,
    batch_size: int = 5,
    forecast_horizon: int = 30
) -> None:
    """Complete timeseries forecasting pipeline with batch prediction.
    
    This pipeline trains a model and immediately uses it for batch forecasting,
    simulating real-world scenarios where companies forecast demand for many products.

    Args:
        n_products: Number of products to forecast (workshop: 10, production: 100,000)
        n_estimators: Number of trees for Random Forest
        batch_size: Products per batch (workshop: 5, production: 1000+)
        forecast_horizon: Days to forecast ahead
    """
    print(f"Starting timeseries forecasting pipeline for {n_products} products...")

    # Generate synthetic timeseries data
    data = generate_timeseries_data(n_products=n_products)

    # Create data exploration report
    explore_timeseries_data(data)

    # Prepare features and split data
    training_data, forecast_data, _, feature_names = prepare_timeseries_features(
        data, forecast_horizon=forecast_horizon
    )

    # Train forecasting model
    model = train_forecast_model(training_data, feature_names, n_estimators)

    # Generate batch predictions (simulates immediate inference)
    predictions = batch_predict(model, forecast_data, feature_names, batch_size)

    # Validate predictions and create report
    validate_predictions(predictions)


if __name__ == "__main__":
    # Run the pipeline
    timeseries_forecast_pipeline(
        n_products=10,
        n_estimators=100,
        batch_size=5,
        forecast_horizon=30
    )
