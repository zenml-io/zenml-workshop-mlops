"""
ZenML Inference Pipeline - Simplified with Model Control Plane
Uses ZenML's Model Control Plane to load the best model and preprocessing artifacts
from the training pipeline for batch inference.
"""

from zenml import pipeline, step, get_step_context, Model
from zenml.enums import ModelStages
from zenml.config import DockerSettings
import pandas as pd
from typing import Annotated


@step
def batch_inference(
    inference_data: pd.DataFrame,
) -> Annotated[pd.DataFrame, "predictions"]:
    """Run batch inference using the best model from training pipeline.
    
    Args:
        inference_data: New customer data to predict on
        preprocessing_artifacts: Name of preprocessing artifacts
        
    Returns:
        DataFrame with predictions and customer info
    """
    print("🔮 Running batch inference...")
    
    # Get the current model context
    zenml_model = get_step_context().model

    best_model_name = zenml_model.run_metadata.get("best_model", "")

    best_model = zenml_model.get_model_artifact(best_model_name).load()
    
    # TODO: Implement batch inference logic here


@pipeline(
    enable_cache=True,
    model=Model(
        name="ChurnPredictionModel",
        description="End-to-end pipeline for churn prediction.",
        version=ModelStages.LATEST  # This takes the latest model produced by the training pipeline
    ),
    settings={
        "docker": DockerSettings(
            parent_image="europe-west3-docker.pkg.dev/zenml-workshop/zenml-436496/zenml@sha256:6fc236a9c95ca84033b06ee578d0c47db7a289404c640b96426c4641b63db576",
            requirements=[
                "pandas",
                "numpy",
                "scikit-learn",
                "plotly",
                "xgboost",
                "lightgbm",
                "pyarrow",
                "fastparquet"
            ],
            python_package_installer="uv",
            apt_packages=["libgomp1"],
            skip_build=True
        ),
    },
)
def batch_inference_pipeline():
    """Simplified batch inference pipeline using Model Control Plane.
    
    Args:
        data_path: Path to new customer data for predictions
    """
    print("🚀 Starting simplified batch inference pipeline...")
    
    


if __name__ == "__main__":
    print("🔮 ZenML Simplified Batch Inference Pipeline")
    
    # Run the pipeline
    predictions = batch_inference_pipeline()
    
    print("\n🎉 Simplified inference pipeline completed!")
    print("\nKey benefits of this approach:")
    print("✅ Uses ZenML Model Control Plane for artifact management")
    print("✅ Automatically loads best model from training pipeline")
    print("✅ Consistent preprocessing between training and inference")
    print("✅ Much simpler code with same functionality")
    print("✅ Automatic model versioning and lineage tracking")
    
    print("\nNext steps:")
    print("1. Modify training pipeline to save artifacts to Model Control Plane")
    print("2. Set up model staging (STAGING -> PRODUCTION)")
    print("3. Add model monitoring and drift detection")
    print("4. Schedule for regular batch processing")
