"""
Example 3: Custom Materialization
=================================

This example demonstrates how to create a custom materializer for pandas DataFrames.
"""

import os
import json
from typing import Annotated, Type, Any, Dict
import pandas as pd

from zenml import step, pipeline
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType, VisualizationType
from zenml.metadata.metadata_types import MetadataType


class CustomPandasMaterializer(BaseMaterializer):
    """Custom materializer for pandas DataFrames"""
    
    ASSOCIATED_TYPES = (pd.DataFrame,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA
    
    def load(self, data_type: Type[Any]) -> pd.DataFrame:
        """Load DataFrame from storage."""
        filepath = os.path.join(self.uri, "data.parquet")
        with self.artifact_store.open(filepath, "rb") as f:
            return pd.read_parquet(f)
    
    def save(self, data: pd.DataFrame) -> None:
        """Save DataFrame to storage as parquet."""
        filepath = os.path.join(self.uri, "data.parquet")
        with self.artifact_store.open(filepath, "wb") as f:
            data.to_parquet(f, index=False)
    
    def save_visualizations(self, data: pd.DataFrame) -> Dict[str, VisualizationType]:
        """Generate HTML visualization of the DataFrame."""
        vis_path = os.path.join(self.uri, "dataframe_preview.html")
        
        # Create a simple HTML representation
        html_content = f"""
        <html>
        <head><title>DataFrame Preview</title></head>
        <body>
            <h2>DataFrame Overview</h2>
            <p><strong>Shape:</strong> {data.shape}</p>
            <p><strong>Columns:</strong> {', '.join(data.columns)}</p>
            
            <h3>First 10 rows:</h3>
            {data.head(10).to_html(classes='table table-striped', table_id='dataframe-preview')}
            
            <h3>Data Types:</h3>
            {data.dtypes.to_frame('Type').to_html(classes='table table-striped')}
            
            <h3>Summary Statistics:</h3>
            {data.describe().to_html(classes='table table-striped')}
        </body>
        </html>
        """
        
        with self.artifact_store.open(vis_path, "w") as f:
            f.write(html_content)
        
        return {vis_path: VisualizationType.HTML}
    
    def extract_metadata(self, data: pd.DataFrame) -> Dict[str, MetadataType]:
        """Extract metadata from DataFrame."""
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            "shape": f"{data.shape[0]}x{data.shape[1]}",
            "num_rows": data.shape[0],
            "num_columns": data.shape[1],
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "missing_values": int(data.isnull().sum().sum()),
            "column_names": data.columns.tolist()
        }


@step(output_materializers=CustomPandasMaterializer)
def create_sample_dataframe() -> Annotated[pd.DataFrame, "sample_dataframe"]:
    """Create a sample DataFrame for demonstration."""
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['New York', 'London', 'Tokyo', 'Paris', 'Berlin'],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing']
    }
    return pd.DataFrame(data)


@step
def analyze_dataframe(df: pd.DataFrame) -> Annotated[Dict[str, Any], "analysis"]:
    """Analyze the DataFrame and return insights."""
    analysis = {
        'total_employees': len(df),
        'average_age': df['age'].mean(),
        'average_salary': df['salary'].mean(),
        'departments': df['department'].value_counts().to_dict(),
        'cities': df['city'].value_counts().to_dict(),
        'salary_by_department': df.groupby('department')['salary'].mean().to_dict()
    }
    
    print("DataFrame Analysis:")
    print(f"  Total employees: {analysis['total_employees']}")
    print(f"  Average age: {analysis['average_age']:.1f}")
    print(f"  Average salary: ${analysis['average_salary']:,.2f}")
    print(f"  Departments: {analysis['departments']}")
    
    return analysis


@pipeline
def custom_materialization_pipeline():
    """Pipeline demonstrating custom materialization."""
    df = create_sample_dataframe()
    analysis = analyze_dataframe(df)
    return df, analysis


if __name__ == "__main__":
    # Run the pipeline
    custom_materialization_pipeline()
    