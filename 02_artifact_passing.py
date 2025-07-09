"""
Example 2: How to Pass Artifacts Between Steps
==============================================

This example demonstrates how to pass artifacts between pipelines in ZenML.
"""

from typing import Annotated
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from zenml import step, pipeline, Model, get_step_context
from zenml.client import Client
from zenml.config import DockerSettings
from zenml.enums import ModelStages


@step
def create_sample_data() -> Annotated[pd.DataFrame, "sample_data"]:
    """Create sample data for demonstration."""
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [10, 20, 30, 40, 50]
    })
    return data


@step
def train_simple_model(data: pd.DataFrame) -> Annotated[Pipeline, "simple_model"]:
    """Train a simple model on the data."""
    X = data[['feature1', 'feature2']]
    y = data['target']
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(n_estimators=10, random_state=42))
    ])
    
    model.fit(X, y)
    return model


@pipeline(
    enable_cache=False,
    model=Model(
        name="SimpleModel",
        description="Pipeline that creates and trains a simple model.",
    ),
    settings={
        "docker": DockerSettings(
            requirements=["pandas", "scikit-learn"],
        ),
    },
)
def training_pipeline():
    """Pipeline that creates data and trains a model."""
    data = create_sample_data()
    model = train_simple_model(data)
    return model


@step
def use_model_artifacts(model_artifact: str = "simple_model", data_artifact: str = "sample_data"):
    """Load and use artifacts from the model."""
    zenml_model = get_step_context().model
    
    # Load the model artifact
    model = zenml_model.get_model_artifact(model_artifact).load()
    
    # Load the data artifact
    data = zenml_model.get_artifact(data_artifact).load()
    
    # Use the model to make predictions
    X = data[['feature1', 'feature2']]
    predictions = model.predict(X)
    
    return predictions


@pipeline(
    enable_cache=False,
    model=Model(
        name="SimpleModel",
        description="Pipeline that loads artifacts from the model.",
        version=ModelStages.PRODUCTION
    ),
    settings={
        "docker": DockerSettings(
            requirements=["pandas", "scikit-learn"],
        ),
    },
)
def inference_pipeline():
    """Pipeline that loads and uses artifacts from the trained model."""
    predictions = use_model_artifacts()
    return predictions


if __name__ == "__main__":
    # First, run the training pipeline to create artifacts
    print("Running training pipeline...")
    training_pipeline()

    latest_model = Model(name="SimpleModel", version=ModelStages.LATEST)
    latest_model.set_stage(stage=ModelStages.PRODUCTION)

    # Then, run the inference pipeline to use the artifacts
    print("\nRunning inference pipeline...")
    inference_pipeline()