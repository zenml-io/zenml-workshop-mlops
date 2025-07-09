"""
Example 6: Run Templates
========================

This example demonstrates how to create and use run templates in ZenML.
"""

import pandas as pd
import numpy as np
from typing import Annotated

from zenml import step, pipeline
from zenml.client import Client


@step
def generate_data(n_samples: int = 1000, noise: float = 0.1) -> Annotated[pd.DataFrame, "data"]:
    """Generate synthetic data."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, 3)
    
    # Generate target with noise
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, noise, n_samples)
    
    data = pd.DataFrame({
        'feature1': X[:, 0],
        'feature2': X[:, 1], 
        'feature3': X[:, 2],
        'target': y
    })
    
    print(f"Generated {n_samples} samples with noise level {noise}")
    return data


@step
def process_data(data: pd.DataFrame, scale_factor: float = 1.0) -> Annotated[dict, "results"]:
    """Process data and return statistics."""
    
    # Apply scaling
    scaled_data = data.copy()
    scaled_data[['feature1', 'feature2', 'feature3']] *= scale_factor
    
    # Calculate statistics
    stats = {
        'mean_target': float(scaled_data['target'].mean()),
        'std_target': float(scaled_data['target'].std()),
        'correlation': float(scaled_data['feature1'].corr(scaled_data['target'])),
        'scale_factor': scale_factor,
        'n_samples': len(scaled_data)
    }
    
    print(f"Processed {len(scaled_data)} samples with scale factor {scale_factor}")
    print(f"Target mean: {stats['mean_target']:.3f}, std: {stats['std_target']:.3f}")
    
    return stats


@pipeline
def simple_template_pipeline(
    n_samples: int = 1000,
    noise: float = 0.1,
    scale_factor: float = 1.0
):
    """Simple pipeline with parameters for template demonstration."""
    data = generate_data(n_samples=n_samples, noise=noise)
    results = process_data(data, scale_factor=scale_factor)
    return results


def create_template():
    """Create a run template from the pipeline."""
    print("=== Creating Run Template ===")
    
    # Create template from pipeline
    template = simple_template_pipeline.create_run_template(
        name="simple_data_processing_template"
    )
    
    print(f" Created template: {template.name}")
    print(f"   Template ID: {template.id}")
    
    return template


def demonstrate_template_usage():
    """Show how to use the created template."""
    print("\n=== How to Use the Template ===")
    
    print("\n= Option A: Trigger via Dashboard")
    print("   1. Go to your ZenML dashboard")
    print("   2. Navigate to 'Run Templates' section")
    print("   3. Find 'simple_data_processing_template'")
    print("   4. Click 'Run Template'")
    print("   5. Modify parameters if needed:")
    print("      - n_samples: 500 (or any number)")
    print("      - noise: 0.2 (or any float)")
    print("      - scale_factor: 2.0 (or any float)")
    print("   6. Click 'Run' to execute")
    
    print("\n= Option B: Trigger Programmatically")
    print("   Run the function below to trigger via Python:")


def trigger_template_programmatically():
    """Trigger the template programmatically."""
    print("\n=== Triggering Template Programmatically ===")
    
    client = Client()
    
    try:
        # Get the template
        template = client.get_run_template("simple_data_processing_template")
        
        # Get the template configuration
        config = template.config_template
        
        # Modify parameters if desired
        config.parameters["n_samples"] = 750
        config.parameters["noise"] = 0.15
        config.parameters["scale_factor"] = 1.5
        
        print("Modified parameters:")
        print(f"  n_samples: {config.parameters['n_samples']}")
        print(f"  noise: {config.parameters['noise']}")
        print(f"  scale_factor: {config.parameters['scale_factor']}")
        
        # Trigger the pipeline
        print("\n=� Triggering pipeline...")
        client.trigger_pipeline(
            template_id=template.id,
            run_configuration=config
        )
        
        print(" Pipeline triggered successfully!")
        print("Check the dashboard to see the running pipeline.")
        
    except Exception as e:
        print(f"L Error triggering template: {e}")
        print("Make sure the template was created first!")


if __name__ == "__main__":
    print("=== ZenML Run Templates Example ===")
    
    # Step 1: Create the template
    template = create_template()
    
    # Step 2: Show usage options
    demonstrate_template_usage()
    
    # Step 3: Demonstrate programmatic triggering
    # trigger_template_programmatically()