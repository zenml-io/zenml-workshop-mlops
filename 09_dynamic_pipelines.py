"""
Example 9: Dynamic Pipeline Patterns
====================================

This example demonstrates correct patterns for creating dynamic pipelines in ZenML.
Shows how to work within the constraints of client-side pipeline definition.
"""

import pandas as pd
import numpy as np
from typing import Annotated, List

from zenml import step, pipeline, get_step_context
from zenml.client import Client


@step
def generate_data(size: int = 100) -> Annotated[pd.DataFrame, "data"]:
    """Generate synthetic data."""
    np.random.seed(42)
    data = pd.DataFrame({
        'values': np.random.randn(size),
        'category': np.random.choice(['A', 'B', 'C'], size)
    })
    return data


@step
def process_type_a(data: pd.DataFrame) -> Annotated[dict, "results"]:
    """Process data using method A."""
    return {
        'method': 'A',
        'mean': float(data['values'].mean()),
        'count': len(data)
    }


@step
def process_type_b(data: pd.DataFrame) -> Annotated[dict, "results"]:
    """Process data using method B."""
    return {
        'method': 'B', 
        'median': float(data['values'].median()),
        'count': len(data)
    }


@step
def process_type_c(data: pd.DataFrame) -> Annotated[dict, "results"]:
    """Process data using method C."""
    return {
        'method': 'C',
        'std': float(data['values'].std()),
        'count': len(data)
    }


@step
def fan_in_aggregator(step_prefix: str = "process_") -> Annotated[dict, "final_results"]:
    """
    Fan-in step: Aggregate results from parallel processing steps.
    
    This demonstrates the correct way to collect outputs from dynamically
    created parallel steps using the ZenML client.
    """
    # Get current run context
    run_name = get_step_context().pipeline_run.name
    run = Client().get_pipeline_run(run_name)
    
    # Collect results from all parallel processing steps
    processed_results = {}
    for step_name, step_info in run.steps.items():
        if step_name.startswith(step_prefix):
            # Load the output from each parallel step
            output = step_info.outputs["output"].load()
            processed_results[step_info.name] = output
    
    # Aggregate the results
    methods_used = [result['method'] for result in processed_results.values()]
    total_count = sum(result['count'] for result in processed_results.values())
    
    return {
        'total_methods': len(processed_results),
        'methods_used': methods_used,
        'total_count': total_count,
        'individual_results': processed_results
    }


# ✅ PATTERN 1: Dynamic pipeline based on parameters (Fan-out/Fan-in)
@pipeline
def dynamic_fan_out_pipeline(
    data_size: int = 100,
    processing_methods: List[str] = ["A", "B"]
):
    """
    ✅ CORRECT: Dynamic pipeline using fan-out/fan-in pattern.
    
    This works because:
    - processing_methods is a pipeline parameter (known at definition time)
    - We create parallel steps dynamically based on parameters
    - We use fan-in to aggregate results from parallel steps
    - We use `after` to ensure proper execution order
    """
    # Generate data once
    data = generate_data(size=data_size)
    
    # Fan-out: Create parallel processing steps dynamically
    parallel_steps = []
    
    if "A" in processing_methods:
        result_a = process_type_a(data, id="process_a")
        parallel_steps.append(result_a)
    
    if "B" in processing_methods:
        result_b = process_type_b(data, id="process_b") 
        parallel_steps.append(result_b)
    
    if "C" in processing_methods:
        result_c = process_type_c(data, id="process_c")
        parallel_steps.append(result_c)
    
    # Fan-in: Aggregate results after all parallel steps complete
    final_results = fan_in_aggregator(
        step_prefix="process_",
        after=parallel_steps  # Ensures proper execution order
    )
    
    return final_results


@step
def conditional_processor(
    data: pd.DataFrame,
    processing_mode: str = "basic",
    quality_threshold: float = 0.5
) -> Annotated[dict, "results"]:
    """
    ✅ CORRECT: Handle data-dependent conditionals INSIDE steps.
    
    This step demonstrates how to handle conditional logic that depends
    on actual data values - do it inside the step, not at pipeline level.
    """
    # Check data quality inside the step (where actual data is available)
    quality_score = 1.0 - (data['values'].isna().sum() / len(data))
    data_mean = data['values'].mean()
    
    # Conditional logic based on actual computed values
    if quality_score > quality_threshold and processing_mode == "advanced":
        result = {
            'method': 'advanced_conditional',
            'mean': float(data_mean),
            'std': float(data['values'].std()),
            'quality_score': quality_score,
            'quartiles': data['values'].quantile([0.25, 0.5, 0.75]).to_dict()
        }
    else:
        result = {
            'method': 'basic_conditional',
            'mean': float(data_mean),
            'quality_score': quality_score,
            'count': len(data)
        }
    
    return result


# ✅ PATTERN 2: Conditional logic inside steps
@pipeline
def conditional_processing_pipeline(
    data_size: int = 100,
    processing_mode: str = "basic",
    quality_threshold: float = 0.5
):
    """
    ✅ CORRECT: Handle conditionals inside steps, not at pipeline level.
    
    This demonstrates how to handle conditional logic that depends on
    computed values by moving the logic inside the step.
    """
    data = generate_data(size=data_size)
    
    # Pass parameters to control step behavior
    result = conditional_processor(
        data,
        processing_mode=processing_mode,
        quality_threshold=quality_threshold
    )
    
    return result


@step 
def batch_processor(
    data: pd.DataFrame,
    batch_size: int = 50
) -> Annotated[dict, "batch_results"]:
    """
    ✅ CORRECT: Handle iteration inside steps, not at pipeline level.
    
    This step demonstrates how to handle loops and batch processing
    inside the step where actual data is available.
    """
    # Process data in batches inside the step
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        batch_result = {
            'batch_id': i // batch_size,
            'size': len(batch),
            'mean': float(batch['values'].mean()),
            'std': float(batch['values'].std())
        }
        results.append(batch_result)
    
    return {
        'total_batches': len(results),
        'batch_results': results,
        'total_processed': len(data)
    }


# ✅ PATTERN 3: Iteration inside steps
@pipeline
def batch_processing_pipeline(
    data_size: int = 200,
    batch_size: int = 50
):
    """
    ✅ CORRECT: Handle loops inside steps, not at pipeline level.
    
    This demonstrates how to handle batch processing and iteration
    by moving the loop logic inside the step.
    """
    data = generate_data(size=data_size)
    
    # Let the step handle the iteration internally
    batch_results = batch_processor(data, batch_size=batch_size)
    
    return batch_results


def demonstrate_dynamic_patterns():
    """Demonstrate the correct patterns for dynamic pipelines."""
    print("=== Dynamic Pipeline Patterns Demo ===")
    
    # Pattern 1: Fan-out/Fan-in
    print("\n✅ Pattern 1: Fan-out/Fan-in for parallel processing")
    result1 = dynamic_fan_out_pipeline(
        data_size=150,
        processing_methods=["A", "B", "C"]
    )
    
    # Pattern 2: Conditional logic inside steps
    print("\n✅ Pattern 2: Conditional logic inside steps")
    result2 = conditional_processing_pipeline(
        data_size=100,
        processing_mode="advanced",
        quality_threshold=0.8
    )
    
    # Pattern 3: Iteration inside steps  
    print("\n✅ Pattern 3: Batch processing inside steps")
    result3 = batch_processing_pipeline(
        data_size=200,
        batch_size=75
    )


if __name__ == "__main__":
    """
    Key Patterns for Dynamic Pipelines:
    
    1. ✅ Use pipeline parameters for dynamic step creation
    2. ✅ Use fan-out/fan-in pattern for parallel processing
    3. ✅ Handle data-dependent conditionals INSIDE steps
    4. ✅ Handle loops and iteration INSIDE steps
    5. ✅ Use `after` parameter for execution dependencies
    
    Remember: Pipeline structure is static, step behavior can be dynamic!
    """
    demonstrate_dynamic_patterns()