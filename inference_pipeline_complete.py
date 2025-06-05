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

from training_pipeline_complete import load_data

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

    best_model_type = zenml_model.run_metadata.get("best_model", "")

    best_model = zenml_model.get_model_artifact(best_model_type).load()
    print("✅ Loaded best model from ZenML Model Control Plane")
        
    le_contract = zenml_model.get_artifact("le_contract").load()
    le_payment = zenml_model.get_artifact("le_payment").load()
    le_internet = zenml_model.get_artifact("le_internet").load()
    scaler = zenml_model.get_artifact("scaler").load()
    
    # Preprocess the inference data (same as training)
    df_processed = inference_data.copy()
    
    # Ensure customer_id exists for tracking
    if "customer_id" not in df_processed.columns:
        df_processed["customer_id"] = [f"CUST_{i:06d}" for i in range(len(df_processed))]
    
    # Store customer_id for results
    customer_ids = df_processed["customer_id"].copy()
    df_features = df_processed.drop("customer_id", axis=1)
    
    # Drop target column if it exists in inference data (shouldn't be used for prediction)
    if "churned" in df_features.columns:
        df_features = df_features.drop("churned", axis=1)
    
    # Apply categorical encoding
    df_features["contract_type_encoded"] = le_contract.transform(df_features["contract_type"])
    df_features["payment_method_encoded"] = le_payment.transform(df_features["payment_method"])
    df_features["internet_service_encoded"] = le_internet.transform(df_features["internet_service"])
    
    # Drop original categorical columns
    df_features = df_features.drop(["contract_type", "payment_method", "internet_service"], axis=1)
    
    # Scale numerical features
    numerical_cols = ["age", "tenure_months", "monthly_charges", "total_charges"]
    df_features[numerical_cols] = scaler.transform(df_features[numerical_cols])
    
    # Make predictions
    predictions = best_model.predict(df_features.values)
    probabilities = best_model.predict_proba(df_features.values)[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        "customer_id": customer_ids,
        "churn_prediction": predictions,
        "churn_probability": probabilities,
        "risk_level": ["High" if p > 0.7 else "Medium" if p > 0.5 else "Low" for p in probabilities]
    })
    
    # Add some customer context back
    results["monthly_charges"] = inference_data["monthly_charges"].values
    results["contract_type"] = inference_data["contract_type"].values
    
    print(f"🎯 Generated predictions for {len(results)} customers")
    print(f"📊 Predicted churn rate: {predictions.mean():.1%}")
    print(f"⚠️  High risk customers: {sum(1 for r in results['risk_level'] if r == 'High')}")
    
    return results


@pipeline(
    enable_cache=True,
    model=Model(
        name="ChurnPredictionModel",
        description="End-to-end pipeline for churn prediction.",
        version=ModelStages.LATEST
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
    
    # Load new data
    inference_data = load_data()
    
    # Run batch inference using Model Control Plane
    predictions = batch_inference(inference_data)
    
    print("✅ Batch inference completed!")
    return predictions


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
