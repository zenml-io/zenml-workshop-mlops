"""
ZenML Timeseries Forecasting Pipeline - Scaffold
Progressive implementation template for converting timeseries forecasting to ZenML pipeline.

This scaffold provides the complete ZenML structure for timeseries forecasting with:
- All @step decorators in place for timeseries workflow
- Proper type hints and annotations for forecasting
- Function signatures ready for batch prediction implementation
- Implementation left as TODOs for learning

Workshop participants can progressively implement each function to build their
timeseries forecasting pipeline with batch processing.
"""

from zenml import ArtifactConfig, Model, pipeline, step, log_metadata
from zenml.config import DockerSettings
from zenml.logger import get_logger
from zenml.enums import ArtifactType
from zenml.types import HTMLString

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Annotated, Dict, Any, List
from datetime import datetime, timedelta

logger = get_logger(__name__)

# =============================================================================
# STEP 1: Timeseries Data Generation (Start Here!)
# =============================================================================

@step
def generate_timeseries_data(n_products: int = 10) -> Annotated[pd.DataFrame, "timeseries_data"]:
    """Generate synthetic timeseries data for forecasting demo.
    
    TODO: Implement timeseries data generation
    - Create 2 years of daily data (2022-01-01 to 2023-12-31)
    - Generate data for multiple products with realistic patterns
    - Include seasonality, trend, and noise
    - Add temporal features (day_of_week, month, quarter, is_weekend)
    
    Args:
        n_products: Number of products to generate data for

    Returns:
        DataFrame with timeseries data for multiple products
    """
    logger.info(f"Generating timeseries data for {n_products} products")
    
    # TODO: Implement data generation
    # Hints:
    # - Use pd.date_range() for date range
    # - Create base_demand + trend + seasonality + noise
    # - Use np.sin() for seasonal patterns
    # - Ensure non-negative demand values
    # - Create product_id as 'PROD_001', 'PROD_002', etc.
    
    pass

# =============================================================================
# STEP 2: Timeseries Feature Engineering (Multi-output Step)
# =============================================================================

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
    
    TODO: Implement timeseries feature engineering
    - Create lag features (1, 7, 30 days) for each product
    - Add rolling averages (7, 30 days)
    - Split data into training and forecast periods
    - Scale features using StandardScaler
    - Handle NaN values from lag features
    
    Args:
        df: Raw timeseries DataFrame
        forecast_horizon: Number of days to forecast

    Returns:
        Tuple of (training_data, forecast_data, scaler, feature_names)
    """
    logger.info("Starting timeseries feature preparation")
    
    # TODO: Implement feature engineering steps
    # Hints:
    # - Process each product separately for lag features
    # - Use .shift() for lag features
    # - Use .rolling().mean() for rolling averages
    # - Split based on cutoff_date = max_date - forecast_horizon
    # - Fit scaler only on training data
    # - Return feature_names list for modeling
    
    pass


# =============================================================================
# STEP 3: Forecasting Model Training (ML Step)  
# =============================================================================

@step
def train_forecast_model(
    training_data: pd.DataFrame, 
    feature_names: list,
    n_estimators: int = 100
) -> Annotated[
    RandomForestRegressor, ArtifactConfig(name="forecast_model", artifact_type=ArtifactType.MODEL,)
]:
    """Train Random Forest model for demand forecasting.
    
    TODO: Implement forecasting model training
    - Extract features and target from training_data
    - Create RandomForestRegressor with appropriate parameters
    - Fit model on training data
    - Calculate and log training metrics (MAE, R²)
    
    Args:
        training_data: Training dataset with features and target
        feature_names: List of feature column names
        n_estimators: Number of trees in the forest

    Returns:
        Trained RandomForestRegressor model
    """
    logger.info(f"Training forecasting model with {n_estimators} estimators")
    
    # TODO: Implement model training
    # Hints:
    # - X_train = training_data[feature_names]
    # - y_train = training_data['demand']
    # - Use RandomForestRegressor with regression-specific parameters
    # - Set n_jobs=-1 for parallel processing
    # - Log training metrics with log_metadata()
    
    pass


# =============================================================================
# STEP 4: Batch Prediction (Core Forecasting Step)
# =============================================================================

@step
def batch_predict(
    model: RandomForestRegressor,
    forecast_data: pd.DataFrame,
    feature_names: list,
    batch_size: int = 5
) -> Annotated[pd.DataFrame, "batch_predictions"]:
    """Generate batch predictions for multiple products.
    
    TODO: Implement batch prediction logic
    - Split products into batches of specified size
    - Process each batch separately (simulates production scaling)
    - Make predictions for each batch
    - Combine all predictions into single DataFrame
    - Add batch_id for tracking
    
    Args:
        model: Trained forecasting model
        forecast_data: Data to forecast on
        feature_names: List of feature column names
        batch_size: Number of products per batch

    Returns:
        DataFrame with predictions for all products
    """
    logger.info(f"Generating batch predictions (batch size: {batch_size})")
    
    # TODO: Implement batch prediction
    # Hints:
    # - Get unique products: forecast_data['product_id'].unique()
    # - Calculate n_batches = ceil(n_products / batch_size)
    # - Loop through batches, filter data, make predictions
    # - Add 'predicted_demand' and 'batch_id' columns
    # - Use pd.concat() to combine all batch results
    
    pass

# =============================================================================
# STEP 5: Forecast Validation (Evaluation Step)
# =============================================================================

@step
def validate_predictions(
    predictions_df: pd.DataFrame
) -> Tuple[
    Annotated[Dict[str, Any], "validation_metrics"],
    Annotated[str, "validation_summary"]
]:
    """Validate forecast predictions and calculate metrics.
    
    TODO: Implement forecast validation
    - Calculate overall metrics: MAE, RMSE, R², MAPE
    - Calculate per-product metrics
    - Create validation summary report
    - Log all metrics for tracking
    
    Args:
        predictions_df: DataFrame with actual and predicted demand

    Returns:
        Tuple of validation metrics dict and summary string
    """
    logger.info("Validating forecast predictions")
    
    # TODO: Implement validation logic
    # Hints:
    # - Use sklearn.metrics for MAE, MSE, R²
    # - Calculate MAPE: mean(abs((actual - predicted) / actual)) * 100
    # - Loop through products for per-product metrics
    # - Use log_metadata() to track all metrics
    # - Create readable summary string
    
    pass

# =============================================================================
# OPTIONAL STEP 6: Advanced Reporting and Visualization
# =============================================================================

@step
def create_forecast_report(
    predictions_df: pd.DataFrame,
    validation_metrics: Dict[str, Any]
) -> Annotated[HTMLString, "forecast_report"]:
    """Create comprehensive forecast evaluation report with visualizations.
    
    TODO: Create interactive forecast visualization report
    - Plot actual vs predicted for each product
    - Show batch processing results
    - Include error distribution plots
    - Generate HTML report for dashboard
    
    Args:
        predictions_df: DataFrame with predictions and actuals
        validation_metrics: Dictionary with validation metrics
        
    Returns:
        HTML string containing comprehensive forecast report
    """
    logger.info("Creating forecast evaluation report")
    
    # TODO: Implement forecast report
    # Hints:
    # - Use plotly for interactive charts
    # - Create subplots for multiple products
    # - Include metrics summary table
    # - Return fig.to_html() + metrics_html
    
    pass


# =============================================================================
# MAIN PIPELINE: Connect All Steps
# =============================================================================

@pipeline(
    enable_cache=False,
    model=Model(
        name="TimeseriesForecastModel",
        description="End-to-end timeseries forecasting pipeline with batch prediction.",
    ),
    settings={
        "docker": DockerSettings(
            parent_image="europe-west3-docker.pkg.dev/zenml-workshop/zenml-436496/zenml@sha256:6fc236a9c95ca84033b06ee578d0c47db7a289404c640b96426c4641b63db576",
            requirements="requirements.txt",
            python_package_installer="uv",
        ),
    },
)
def timeseries_forecast_pipeline(
    n_products: int = 10,
    n_estimators: int = 100,
    batch_size: int = 5,
    forecast_horizon: int = 30,
    create_reports: bool = True
) -> None:
    """Complete timeseries forecasting pipeline with batch prediction.
    
    TODO: Connect all steps to create the forecasting pipeline
    - Call steps in correct order for timeseries workflow
    - Pass outputs as inputs to next steps
    - Use conditional logic for optional reports
    - Implement train-and-predict pattern (no separate inference)
    
    Args:
        n_products: Number of products to forecast (workshop: 10, production: 100,000)
        n_estimators: Number of trees for Random Forest
        batch_size: Products per batch (workshop: 5, production: 1000+)
        forecast_horizon: Days to forecast ahead
        create_reports: Whether to create detailed HTML reports
    """
    # TODO: Implement pipeline by connecting steps
    # Hints:
    # - Start with: data = generate_timeseries_data(n_products)
    # - Chain step outputs to inputs for timeseries workflow
    # - training_data, forecast_data, scaler, features = prepare_timeseries_features(data)
    # - model = train_forecast_model(training_data, features, n_estimators)
    # - predictions = batch_predict(model, forecast_data, features, batch_size)
    # - metrics, summary = validate_predictions(predictions)
    # - Use if create_reports: for optional reporting
    
    pass

if __name__ == "__main__":

    # Uncomment when implementation is complete:
    timeseries_forecast_pipeline(
    #     n_products=10,
    #     batch_size=5,
    #     n_estimators=100,
    #     forecast_horizon=30
    )