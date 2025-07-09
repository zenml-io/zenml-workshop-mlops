"""
Example 8: Pipeline Antipatterns - Dynamic Pipelines
===============================

This example demonstrates common mistakes and antipatterns in ZenML pipelines.
Understanding what NOT to do helps avoid common pitfalls around dynamic pipelines.
"""

import pandas as pd
import numpy as np
from typing import Annotated

from zenml import step, pipeline


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
def check_data_quality(data: pd.DataFrame) -> Annotated[bool, "quality_check"]:
    """Check if data quality is good."""
    return len(data) > 50 and not data['values'].isna().any()


@step
def basic_processing(data: pd.DataFrame) -> Annotated[dict, "basic_results"]:
    """Basic processing for any data."""
    return {
        'method': 'basic',
        'count': len(data),
        'mean': float(data['values'].mean())
    }


@step
def advanced_processing(data: pd.DataFrame) -> Annotated[dict, "advanced_results"]:
    """Advanced processing for high-quality data."""
    return {
        'method': 'advanced',
        'count': len(data),
        'mean': float(data['values'].mean()),
        'std': float(data['values'].std())
    }


@step
def process_type_a(data: pd.DataFrame) -> Annotated[dict, "results"]:
    """Process data using method A."""
    return {'method': 'A', 'mean': float(data['values'].mean())}


@step
def process_type_b(data: pd.DataFrame) -> Annotated[dict, "results"]:
    """Process data using method B."""
    return {'method': 'B', 'median': float(data['values'].median())}


@step
def aggregate_wrong(results: list) -> Annotated[dict, "final_results"]:
    """This step signature is wrong - can't pass lists of step outputs."""
    return {'total': len(results)}


# ❌ ANTIPATTERN 1: Using step outputs in conditionals
@pipeline
def antipattern_conditional():
    """
    ❌ ANTIPATTERN: Cannot use step outputs in conditional logic.
    
    This will fail because:
    - Step outputs are "promises" that haven't been computed yet
    - Pipeline function runs CLIENT-SIDE at definition time
    - Conditionals need actual values, not promises
    """
    data = generate_data(size=100)
    
    # ❌ This would cause a TypeError if uncommented:
    quality_result = check_data_quality(data)
    if quality_result:  # Cannot use step output in conditional!
        result = advanced_processing(data)
    else:
    #     result = basic_processing(data)
    
    # Must define static pipeline structure instead
    #  basic_processing(data)


# ❌ ANTIPATTERN 2: Trying to collect step outputs in lists
@pipeline
def antipattern_collecting_outputs():
    """
    ❌ ANTIPATTERN: Cannot append step outputs to lists.
    
    This will fail because:
    - Step outputs are promises, not actual data
    - Cannot pass lists of promises to other steps
    - Lists get evaluated client-side, but step outputs aren't available yet
    """
    data = generate_data(size=100)
    
    # ❌ This would fail if uncommented:
    results = []
    result_a = process_type_a(data)
    result_b = process_type_b(data)
    results.append(result_a)  # Cannot append promises to lists!
    results.append(result_b)
    return aggregate_wrong(results)  # Cannot pass list of promises!
    
    # Must use fan-out/fan-in pattern instead -> https://docs.zenml.io/concepts/steps_and_pipelines/advanced_features#fan-out-and-fan-in
    


# ❌ ANTIPATTERN 3: Trying to loop over step outputs
@pipeline
def antipattern_looping_outputs():
    """
    ❌ ANTIPATTERN: Cannot loop over step outputs.
    
    This will fail because:
    - Step outputs are single promises, not iterables
    - Even if a step returns a list, you can't iterate over it at pipeline level
    - Pipeline logic runs before any steps execute
    """
    data = generate_data(size=100)
    
    # ❌ This would fail if the step returned a list:
    categories = some_step_that_returns_list(data)
    results = []
    for category in categories:  # Cannot iterate over step output!
        result = process_category(data, category)
        results.append(result)


# ❌ ANTIPATTERN 4: Accessing step output attributes
@pipeline
def antipattern_attribute_access():
    """
    ❌ ANTIPATTERN: Cannot access attributes of step outputs.
    
    This will fail because:
    - Step outputs are promises, not the actual objects
    - Cannot access attributes like .shape, .columns, etc.
    - All pipeline logic must be based on static information
    """
    data = generate_data(size=100)
    
    # ❌ This would fail if uncommented:
    if len(data) > 50:  # Cannot access len() of step output!
        result = advanced_processing(data)
    else:
        result = basic_processing(data)
    
    # ❌ This would also fail:
    if data.shape[0] > 50:  # Cannot access .shape of step output!
        result = advanced_processing(data)



# ❌ ANTIPATTERN 5: Dynamic step creation based on step outputs
@pipeline
def antipattern_dynamic_steps():
    """
    ❌ ANTIPATTERN: Cannot create steps dynamically based on step outputs.
    
    This will fail because:
    - Number of steps must be known at pipeline definition time
    - Cannot create steps based on computed values
    - Pipeline structure is static, determined client-side
    """
    data = generate_data(size=100)
    
    # ❌ This would fail if uncommented:
    # num_processes = get_process_count(data)  # Step output
    # for i in range(num_processes):  # Cannot use step output in range()!
    #     process_chunk(data, chunk_id=i)
    
    # Must define fixed number of steps instead
    return process_type_a(data)


def demonstrate_antipatterns():
    """Show what happens when you try to use these antipatterns."""
    print("=== Pipeline Antipatterns Demo ===")
    print("\nThese examples show what NOT to do in ZenML pipelines:")
    print("1. ❌ Using step outputs in conditionals")
    print("2. ❌ Collecting step outputs in lists") 
    print("3. ❌ Looping over step outputs")
    print("4. ❌ Accessing step output attributes")
    print("5. ❌ Dynamic step creation based on step outputs")
    
    print("\n✅ Solutions:")
    print("- Use pipeline parameters for conditionals")
    print("- Use fan-out/fan-in pattern for parallel processing")
    print("- Handle data-dependent logic inside steps")
    print("- Define static pipeline structure")
    
    # Run one working example
    result = antipattern_conditional()
    print(f"\nWorking pipeline result: {result}")


if __name__ == "__main__":
    """
    Key Takeaways - What NOT to do:
    
    1. ❌ Don't use step outputs in conditionals
    2. ❌ Don't append step outputs to lists  
    3. ❌ Don't loop over step outputs
    4. ❌ Don't access step output attributes
    5. ❌ Don't create steps dynamically based on step outputs
    
    Remember: Pipeline functions run CLIENT-SIDE at definition time!
    Step outputs are "promises" that haven't been computed yet.
    """
    demonstrate_antipatterns()