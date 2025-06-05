"""
ZenML Training Pipeline Scaffold
Progressive implementation template for converting notebook code to ZenML pipeline

This scaffold provides the complete ZenML structure with:
- All @step decorators in place
- Proper type hints and annotations
- Function signatures ready
- Implementation left as TODOs for learning

Users can progressively implement each function to build their ML pipeline.
"""

from zenml import ArtifactConfig, Model, pipeline, step
from zenml.config import DockerSettings
from zenml.logger import get_logger
from zenml.enums import ArtifactType

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Annotated, Dict, Any

logger = get_logger(__name__)

# =============================================================================
# STEP 1: Basic Data Loading (Start Here!)
# =============================================================================

@step
def load_data(url: str = "https://gist.githubusercontent.com/AlexejPenner/1dae173189a3e5f3671f178e3de97483/raw/customer_churn.csv") -> Annotated[pd.DataFrame, "raw_data"]:
    """Load customer churn data from CSV.
    
    TODO: Implement data loading logic
    - Load CSV
    
    Args:
        data_path: Path to the CSV file

    Returns:
        DataFrame with customer churn data
    """
    logger.info(f"Loading data from {url}")
    
    # TODO: Implement data loading
    # Hints:
    # - Use pd.read_csv() with try/except
    
    pass

# =============================================================================
# STEP 2: Data Preprocessing (Multi-output Step)
# =============================================================================

@step
def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"], 
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
    Annotated[Dict[str, Any], "preprocessing_info"]
]:
    """Prepare data for training with comprehensive preprocessing.
    
    TODO: Implement data preprocessing pipeline
    - Drop customer_id column
    - Encode categorical variables
    - Split features and target
    - Train/test split with stratification
    - Scale numerical features
    - Create preprocessing metadata
    
    Args:
        df: Raw customer churn DataFrame
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of processed training/test data, scaler, and preprocessing info
    """
    logger.info("Starting data preparation")
    
    # TODO: Implement preprocessing steps
    # Hints:
    # - Use LabelEncoder for categorical variables
    # - Use train_test_split with stratify parameter
    # - Fit StandardScaler on training data only
    # - Return detailed preprocessing_info dict
    
    pass


# =============================================================================
# STEP 3: Single Model Training (Basic ML Step)  
# =============================================================================

@step
def train_rf_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int = 100,
    random_state: int = 42
) -> Annotated[
    RandomForestClassifier, ArtifactConfig(name="lr_churn_model", artifact_type=ArtifactType.MODEL,)
]:
    """Train a Random Forest classifier.
    
    TODO: Implement model training
    - Create RandomForestClassifier with parameters
    - Fit model on training data
    - Add logging for training completion
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility

    Returns:
        Trained RandomForestClassifier model
    """
    logger.info(f"Training Random Forest with {n_estimators} estimators")
    
    # TODO: Implement model training
    # Hints:
    # - Use RandomForestClassifier with n_jobs=-1
    # - Set random_state for reproducibility
    # - Call .fit() method
    
    pass


# =============================================================================
# STEP 4: Model Evaluation (Metrics and Analysis)
# =============================================================================

@step
def evaluate_model(
    model: RandomForestClassifier, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    model_name: str = "RandomForest"
) -> Tuple[
    Annotated[float, "accuracy"], 
    Annotated[Dict[str, Any], "detailed_metrics"]
]:
    """Evaluate a model and return comprehensive metrics.
    
    TODO: Implement model evaluation
    - Make predictions on test data
    - Calculate accuracy, precision, recall, f1-score
    - Create confusion matrix
    - Add feature importance if available
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for tracking

    Returns:
        Tuple of accuracy score and detailed metrics dictionary
    """
    logger.info(f"Evaluating {model_name} model")
    
    # TODO: Implement model evaluation
    # Hints:
    # - Use model.predict() and model.predict_proba()
    # - Use sklearn metrics: accuracy_score, classification_report
    # - Include confusion_matrix and feature_importances
    # - Structure results in detailed_metrics dict
    
    pass

# =============================================================================
# OPTIONAL STEP 5: Advanced Model Comparison and Reporting
# =============================================================================
# * here we add a step to also train a Logistic Regression model
# * then we can compare the two models and create a report
# =============================================================================

@step
def train_lr_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int = 100,
    random_state: int = 42
) -> Annotated[
    LogisticRegression, ArtifactConfig(name="lr_churn_model", artifact_type=ArtifactType.MODEL,)
]:
    """Train a Logistic Regression classifier.
    
    TODO: Implement model training
    - Create Logistic Regression with parameters
    - Fit model on training data
    - Add logging for training completion
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility

    Returns:
        Trained RandomForestClassifier model
    """
    logger.info(f"Training Random Forest with {n_estimators} estimators")
    
    # TODO: Implement model training
    # Hints:
    # - Use RandomForestClassifier with n_jobs=-1
    # - Set random_state for reproducibility
    # - Call .fit() method
    
    pass

@step
def create_evaluation_report(
    rf_model: RandomForestClassifier,
    lr_model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessing_info: Dict[str, Any]
) -> Annotated[str, "evaluation_report"]:
    """Create comprehensive model evaluation report with visualizations.
    
    TODO: Create interactive model comparison report
    - Evaluate both models
    - Create comparison charts (accuracy, confusion matrices)
    - Show feature importance plots
    - Generate HTML report for dashboard
    
    Args:
        rf_model: Trained Random Forest model
        lr_model: Trained Logistic Regression model
        X_test: Test features
        y_test: Test labels
        preprocessing_info: Information about data preprocessing
        
    Returns:
        HTML string containing comprehensive evaluation report
    """
    logger.info("Creating comprehensive evaluation report")
    
    # TODO: Implement evaluation report
    # Hints:
    # - Evaluate both models and compare results
    # - Use plotly subplots for multiple charts
    # - Include model comparison table in HTML
    # - Return fig.to_html() + metrics_html
    
    pass


# =============================================================================
# MAIN PIPELINE: Connect All Steps
# =============================================================================

@pipeline(
    enable_cache=False,
    model=Model(
        name="ChurnPredictionModel",
        description="End-to-end pipeline for churn prediction.",
    ),
    settings={
        "docker": DockerSettings(
            parent_image="europe-west3-docker.pkg.dev/zenml-workshop/zenml-436496/zenml@sha256:6fc236a9c95ca84033b06ee578d0c47db7a289404c640b96426c4641b63db576",
            requirements="requirements.txt",
            python_package_installer="uv",
        ),
    },
)
def training_pipeline(
    n_estimators: int = 100,
    test_size: float = 0.2,
    create_reports: bool = True
) -> None:
    """Complete ML training pipeline demonstrating ZenML capabilities.
    
    TODO: Connect all steps to create the pipeline
    - Call steps in correct order
    - Pass outputs as inputs to next steps
    - Use conditional logic for optional reports
    
    Args:
        data_path: Path to the training data
        n_estimators: Number of trees for Random Forest
        test_size: Fraction of data for testing
        create_reports: Whether to create detailed HTML reports
    """
    # TODO: Implement pipeline by connecting steps
    # Hints:
    # - Start with: data = load_data(data_path)
    # - Chain step outputs to inputs: X_train, X_test, y_train, y_test, scaler, preprocessing_info = prepare_data(data)
    # - use optional steps if needed
    # - Each step output becomes input to next step
    
    pass

if __name__ == "__main__":
    print("🚀 ZenML Training Pipeline Scaffold")
    print("=" * 50)
    print("\n📚 Progressive Implementation Guide:")
    print("1. Start with simple_training_pipeline()")
    print("2. Implement each @step function one by one")
    print("3. Test individual steps before connecting them")
    print("4. Move to full training_pipeline() with reports")
    
    print("\n✨ Implementation Order:")
    print("📊 Step 1: load_data() - Basic data loading")
    print("🔍 Step 2: explore_data() - Data visualization") 
    print("⚙️  Step 3: prepare_data() - Data preprocessing")
    print("🤖 Step 4: train_model() - Model training")
    print("📋 Step 5: evaluate_model() - Model evaluation")
    
    print("\n🎯 Key Learning Points:")
    print("✅ Each function has @step decorator")
    print("✅ Type hints with Annotated for artifact naming")
    print("✅ Multi-output steps return Tuples")
    print("✅ Pipeline connects step outputs to inputs")
    print("✅ Automatic artifact versioning and caching")
    
    print("\n🚀 Next Steps:")
    print("1. Implement load_data() function first")
    print("2. Test with: simple_training_pipeline()")
    print("3. Add one step at a time and test")
    print("4. Check ZenML dashboard for artifacts")
    
    # Uncomment when implementation is complete:
    # training_pipeline()
