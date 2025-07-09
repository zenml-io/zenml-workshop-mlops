"""
Example 1: How to Write a Basic ZenML Pipeline
=============================================

This example demonstrates the fundamentals of creating a ZenML pipeline:
- Defining steps with the @step decorator
- Defining pipelines with the @pipeline decorator
- Basic data flow between steps
- Using strong typing for data
"""

from typing import Annotated, Tuple
from zenml import pipeline, step
from zenml.config import DockerSettings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


@step
def load_data() -> Tuple[Annotated[pd.DataFrame, "X"], Annotated[pd.Series, "y"]]:
    """
    Load the iris dataset and return features and target.
    
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    
    print(f"Loaded dataset with {len(X)} samples and {len(X.columns)} features")
    return X, y


@step
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[Annotated[pd.DataFrame, "X_train"], Annotated[pd.DataFrame, "X_test"], Annotated[pd.Series, "y_train"], Annotated[pd.Series, "y_test"]]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Annotated[RandomForestClassifier, "model"]:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model training completed")
    return model


@step
def evaluate_model(
    model: RandomForestClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Annotated[float, "accuracy"]:
    """
    Evaluate the trained model and return accuracy.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        float: Model accuracy
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model accuracy: {accuracy:.4f}")
    return accuracy


@pipeline()
def basic_ml_pipeline() -> float:
    """
    A basic machine learning pipeline that demonstrates:
    - Data loading
    - Data splitting
    - Model training
    - Model evaluation
    
    Returns:
        float: Model accuracy
    """
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    return accuracy


if __name__ == "__main__":
    # Run the pipeline
    basic_ml_pipeline()

    # Run the pipeline on a remote stack

    # switch to remote stack like this from your terminal:
    # zenml stack set zenml-workshop-stack
    
    # basic_ml_pipeline.with_options( 
    #    settings={  
    #        "docker": DockerSettings(
    #             parent_image="europe-west3-docker.pkg.dev/zenml-workshop/zenml-436496/zenml@sha256:d4d0e1c128d1848fccfc3b884a803e4eaaa259ea31426799b5ed52ec87860ac4",
    #             skip_build=True
    #         )
    #     },
    #     enable_cache=False
    # )()
    