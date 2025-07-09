"""
Example 4: Configuration Files & Caching
========================================

This example demonstrates how to use YAML configuration files to configure
step parameters and control caching behavior.
"""

import time
import pandas as pd
from typing import Annotated

from zenml import step, pipeline


@step
def create_data(size: int = 100) -> Annotated[pd.DataFrame, "data"]:
    """Create sample data."""
    print(f"Creating {size} rows of data...")
    time.sleep(1)  # Simulate work
    
    data = pd.DataFrame({
        'values': range(size),
        'squared': [x**2 for x in range(size)]
    })
    
    return data


@step
def process_data(
    data: pd.DataFrame, 
    multiply_by: float = 2.0,
    add_column: bool = True
) -> Annotated[pd.DataFrame, "processed_data"]:
    """Process the data."""
    print(f"Processing data: multiply_by={multiply_by}, add_column={add_column}")
    time.sleep(1)  # Simulate work
    
    result = data.copy()
    result['values'] = result['values'] * multiply_by
    
    if add_column:
        result['extra'] = result['values'] + result['squared']
    
    return result


@pipeline
def config_pipeline():
    """Simple 2-step pipeline."""
    data = create_data()
    processed = process_data(data)
    return processed


if __name__ == "__main__":
    # Run with default parameters
    print("=== Running with defaults ===")
    result1 = config_pipeline()
    
    # Run with config file
    print("=== Running with config.yaml ===")
    result2 = config_pipeline.with_options(config_path="config.yaml")()
    
    # Run again to show caching
    print("=== Running again (caching demo) ===")
    result3 = config_pipeline.with_options(config_path="config.yaml")()