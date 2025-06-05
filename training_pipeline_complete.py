"""
ZenML Training Pipeline - Complete Solution
This comprehensive solution mirrors the notebook workflow with clean ZenML steps
and includes interactive visualizations and reports.
"""

from zenml import ArtifactConfig, Model, log_metadata, pipeline, step
from zenml.types import HTMLString
from zenml.config import DockerSettings
from zenml.enums import ArtifactType

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier
from typing import Tuple, Annotated
import os
import tempfile
import requests

# Import utility functions for HTML report generation
from utils import generate_data_exploration_report, generate_model_evaluation_report


@step
def load_data(url: str = "https://gist.githubusercontent.com/AlexejPenner/1dae173189a3e5f3671f178e3de97483/raw/customer_churn.csv") -> pd.DataFrame:
    """Load customer churn data from CSV.

    In this case we load it from a fictive productions data source represented by a GIST.

    Args:
        data_path: Path to the CSV file

    Returns:
        DataFrame with customer churn data
    """

    # Create temp dir and download file
    temp_dir = tempfile.mkdtemp()
    response = requests.get(url)
    temp_path = os.path.join(temp_dir, "customer_churn.csv")

    with open(temp_path, "wb") as f:
        f.write(response.content)

    # Load into DataFrame
    df = pd.read_csv(temp_path)
    print(f"📊 Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    return df


@step
def explore_data(df: pd.DataFrame) -> Annotated[HTMLString, "data_exploration_report"]:
    """Create comprehensive data exploration report with interactive visualizations.

    Args:
        df: Raw customer churn DataFrame

    Returns:
        HTML report with interactive visualizations
    """
    return generate_data_exploration_report(df)


@step
def prepare_data(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
    Annotated[LabelEncoder, "le_contract"],
    Annotated[LabelEncoder, "le_payment"],
    Annotated[LabelEncoder, "le_internet"],
    Annotated[StandardScaler, "scaler"],
    Annotated[list, "feature_names"],
]:
    """Prepare data for training with comprehensive preprocessing.

    Args:
        df: Raw customer churn DataFrame

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    print("🔧 Preparing data for training...")

    # Create a copy to avoid modifying original
    df_processed = df.copy()

    # Drop customer_id column (not useful for prediction)
    if "customer_id" in df_processed.columns:
        df_processed = df_processed.drop("customer_id", axis=1)

    # Encode categorical variables
    le_contract = LabelEncoder()
    df_processed["contract_type_encoded"] = le_contract.fit_transform(
        df_processed["contract_type"]
    )

    le_payment = LabelEncoder()
    df_processed["payment_method_encoded"] = le_payment.fit_transform(
        df_processed["payment_method"]
    )

    le_internet = LabelEncoder()
    df_processed["internet_service_encoded"] = le_internet.fit_transform(
        df_processed["internet_service"]
    )

    # Drop original categorical columns
    df_processed = df_processed.drop(
        ["contract_type", "payment_method", "internet_service"], axis=1
    )

    # Separate features and target
    X = df_processed.drop("churned", axis=1)
    y = df_processed["churned"]

    feature_names = list(X.columns)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ["age", "tenure_months", "monthly_charges", "total_charges"]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    print(f"✅ Training set: {X_train_scaled.shape[0]} samples")
    print(f"✅ Test set: {X_test_scaled.shape[0]} samples")
    print(f"✅ Features: {len(feature_names)}")

    return (
        X_train_scaled.values,
        X_test_scaled.values,
        y_train.values,
        y_test.values,
        le_contract,
        le_payment,
        le_internet,
        scaler,
        feature_names,
    )


@step
def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100
) -> Annotated[RandomForestClassifier, ArtifactConfig(name="rf_churn_model", artifact_type=ArtifactType.MODEL,)]:
    """Train Random Forest model.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees

    Returns:
        Trained Random Forest model
    """
    print("🌲 Training Random Forest model...")

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42, max_depth=10
    )
    rf_model.fit(X_train, y_train)
    print("✅ Random Forest trained")

    return rf_model


@step
def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100
) -> Annotated[XGBClassifier, ArtifactConfig(name="xgb_churn_model", artifact_type=ArtifactType.MODEL,)]:
    """Train XGBoost model.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees

    Returns:
        Trained XGBoost model
    """
    print("🚀 Training XGBoost model...")

    xgb_model = XGBClassifier(
        n_estimators=n_estimators, random_state=42, max_depth=6, learning_rate=0.1
    )
    xgb_model.fit(X_train, y_train)
    print("✅ XGBoost trained")

    return xgb_model

@step
def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> Annotated[LogisticRegression, ArtifactConfig(name="lr_churn_model", artifact_type=ArtifactType.MODEL,)]:
    """Train Logistic Regression model.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained Logistic Regression model
    """
    print("📊 Training Logistic Regression model...")

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    print("✅ Logistic Regression trained")

    return lr_model


@step
def evaluate_models(
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier,
    lr_model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> Tuple[
    Annotated[dict, "all_metrics"],
    Annotated[HTMLString, "evaluation_report"],
]:
    """Comprehensive model evaluation with interactive visualizations.

    Args:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        lr_model: Trained Logistic Regression model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names

    Returns:
        Tuple of (all_metrics, evaluation_report)
    """
    print("📈 Evaluating models and creating report...")

    models = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "Logistic Regression": lr_model,
    }

    all_metrics = {}

    # Evaluate each model
    for model_type, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )

        all_metrics[model_type] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "predictions": y_pred.tolist(),
            "probabilities": y_prob.tolist() if y_prob is not None else None,
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        log_metadata(
            metadata={
                model_type: {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1)
                }
            },
            infer_model=True
        )

        print(f"🎯 {model_type} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

    # Find best model
    best_model_type = max(all_metrics, key=lambda x: all_metrics[x]["accuracy"])
    best_model = models[best_model_type]
    best_accuracy = all_metrics[best_model_type]["accuracy"]

    print(f"🏆 Best Model: {best_model_type} with accuracy: {best_accuracy:.3f}")

    # Generate evaluation report using utility function
    evaluation_report = generate_model_evaluation_report(
        all_metrics, best_model_type, best_model, y_test, feature_names
    )

    log_metadata(
        metadata = {
            "best_model_type": best_model_type,
            "best_accuracy": float(best_accuracy)
        },
        infer_model=True
    )

    return all_metrics, evaluation_report


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
            apt_packages=["libgomp1"],
            skip_build=True
        ),
    },
)
def training_pipeline(
    n_estimators: int = 100
) -> None:
    """Complete ML training pipeline with comprehensive analysis and reporting.

    Args:
        n_estimators: Number of trees for ensemble methods
    """
    print("🚀 Starting comprehensive training pipeline...")

    # Load data
    data = load_data()

    # Create data exploration report
    explore_data(data)

    # Prepare data
    X_train, X_test, y_train, y_test, _, _, _, _, feature_names = prepare_data(data)

    # Train different types of models
    rf_model = train_random_forest(X_train, y_train, n_estimators)
    xgb_model = train_xgboost(X_train, y_train, n_estimators)
    lr_model = train_logistic_regression(X_train, y_train)

    # Evaluate and compare models
    evaluate_models(
        rf_model, xgb_model, lr_model, X_test, y_test, feature_names
    )


if __name__ == "__main__":
    # Run the pipeline
    training_pipeline(n_estimators=100)

    print("\n✅ ZenML Training Pipeline Complete!")
    print("\nKey Benefits Achieved:")
    print("✅ Comprehensive data exploration with interactive visualizations")
    print("✅ Multiple model comparison and evaluation")
    print("✅ Interactive HTML reports stored as ZenML artifacts")
    print("✅ Automatic artifact versioning and caching")
    print("✅ Pipeline lineage tracking")
    print("✅ Experiment tracking")
    print("✅ Reproducible ML workflows")
    print("✅ Clean separation of concerns")

    print("\nNext steps:")
    print("1. View pipeline runs: zenml pipeline runs list")
    print("2. Explore artifacts: zenml artifact list")
    print("3. Check interactive reports in ZenML dashboard")
    print("4. Run inference pipeline with trained model")
