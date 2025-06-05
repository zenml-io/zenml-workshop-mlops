"""
Utility functions for generating HTML reports and visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, List
from zenml.types import HTMLString


def generate_data_exploration_report(df: pd.DataFrame) -> HTMLString:
    """Generate comprehensive data exploration report with interactive visualizations.

    Args:
        df: Raw customer churn DataFrame

    Returns:
        HTML report with interactive visualizations
    """
    print("🔍 Creating data exploration report...")

    # Create subplots for main analysis
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Churn Rate by Contract Type",
            "Age Distribution by Churn Status",
            "Monthly Charges vs Tenure",
            "Churn Rate by Payment Method",
            "Feature Correlation Heatmap",
            "Customer Segments Analysis",
        ),
        specs=[
            [{"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "heatmap"}, {"type": "box"}],
        ],
    )

    # 1. Churn by contract type
    churn_by_contract = df.groupby("contract_type")["churned"].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=churn_by_contract["contract_type"],
            y=churn_by_contract["churned"],
            name="Churn Rate",
            marker_color="lightcoral",
        ),
        row=1,
        col=1,
    )

    # 2. Age distribution
    churned_ages = df[df["churned"] == 1]["age"]
    retained_ages = df[df["churned"] == 0]["age"]

    fig.add_trace(
        go.Histogram(
            x=retained_ages, name="Retained", opacity=0.7, marker_color="blue"
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=churned_ages, name="Churned", opacity=0.7, marker_color="red"),
        row=1,
        col=2,
    )

    # 3. Scatter plot: Monthly charges vs tenure
    fig.add_trace(
        go.Scatter(
            x=df["tenure_months"],
            y=df["monthly_charges"],
            mode="markers",
            marker=dict(
                color=df["churned"],
                colorscale=["blue", "red"],
                showscale=True,
                colorbar=dict(title="Churned"),
            ),
            name="Customers",
        ),
        row=1,
        col=3,
    )

    # 4. Churn by payment method
    churn_by_payment = df.groupby("payment_method")["churned"].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=churn_by_payment["payment_method"],
            y=churn_by_payment["churned"],
            name="Churn Rate",
            marker_color="orange",
        ),
        row=2,
        col=1,
    )

    # 5. Correlation heatmap
    numeric_cols = [
        "age",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_services",
        "churned",
    ]
    correlation_matrix = df[numeric_cols].corr()

    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            showscale=True,
        ),
        row=2,
        col=2,
    )

    # 6. Box plot for customer segments
    fig.add_trace(
        go.Box(
            x=df["contract_type"],
            y=df["monthly_charges"],
            name="Monthly Charges by Contract",
        ),
        row=2,
        col=3,
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="📊 Customer Churn Data Exploration Dashboard",
        showlegend=False,
    )

    # Create insights summary
    insights_html = f"""
    <div style="margin: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <h3>🔍 Key Data Insights</h3>
        <ul>
            <li><strong>Overall churn rate:</strong> {df['churned'].mean():.1%}</li>
            <li><strong>Month-to-month customers:</strong> {df[df['contract_type']=='month-to-month']['churned'].mean():.1%} churn rate</li>
            <li><strong>Two-year customers:</strong> {df[df['contract_type']=='two-year']['churned'].mean():.1%} churn rate</li>
            <li><strong>High-value customers (>$100/month):</strong> {df[df['monthly_charges']>100]['churned'].mean():.1%} churn rate</li>
            <li><strong>New customers (<6 months):</strong> {df[df['tenure_months']<6]['churned'].mean():.1%} churn rate</li>
            <li><strong>Average tenure (churned):</strong> {df[df['churned']==1]['tenure_months'].mean():.1f} months</li>
            <li><strong>Average tenure (retained):</strong> {df[df['churned']==0]['tenure_months'].mean():.1f} months</li>
        </ul>
    </div>
    """

    # Combine into final HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Exploration Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>🔍 Customer Churn Data Exploration Report</h1>
        {insights_html}
        <div id="main-dashboard">{fig.to_html(include_plotlyjs=False, div_id="main-dashboard")}</div>
    </body>
    </html>
    """

    print("✅ Data exploration report created")
    return HTMLString(html_report)


def generate_model_evaluation_report(
    all_metrics: Dict[str, Dict[str, Any]],
    best_model_name: str,
    best_model: Any,
    y_test: np.ndarray,
    feature_names: List[str]
) -> HTMLString:
    """Generate comprehensive model evaluation report with interactive visualizations.

    Args:
        all_metrics: Dictionary containing metrics for all models
        best_model_name: Name of the best performing model
        best_model: The best performing model object
        y_test: Test labels
        feature_names: List of feature names

    Returns:
        HTML report with model evaluation visualizations
    """
    print("📈 Creating model evaluation report...")

    # Create comprehensive evaluation report
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Model Comparison - Accuracy",
            "Model Comparison - F1 Score",
            f"Confusion Matrix - {best_model_name}",
            f"Feature Importance - {best_model_name}",
            "Precision vs Recall",
            "Model Performance Summary",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}],
        ],
    )

    # 1. Accuracy comparison
    model_names = list(all_metrics.keys())
    accuracies = [all_metrics[name]["accuracy"] for name in model_names]

    fig.add_trace(
        go.Bar(
            x=model_names,
            y=accuracies,
            name="Accuracy",
            marker_color=[
                "gold" if name == best_model_name else "lightblue"
                for name in model_names
            ],
        ),
        row=1,
        col=1,
    )

    # 2. F1 Score comparison
    f1_scores = [all_metrics[name]["f1_score"] for name in model_names]

    fig.add_trace(
        go.Bar(
            x=model_names,
            y=f1_scores,
            name="F1 Score",
            marker_color=[
                "gold" if name == best_model_name else "lightgreen"
                for name in model_names
            ],
        ),
        row=1,
        col=2,
    )

    # 3. Confusion Matrix for best model
    y_pred_best = all_metrics[best_model_name]["predictions"]
    cm = confusion_matrix(y_test, y_pred_best)

    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=["Predicted: No Churn", "Predicted: Churn"],
            y=["Actual: No Churn", "Actual: Churn"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        ),
        row=1,
        col=3,
    )

    # 4. Feature Importance (for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=True)

        fig.add_trace(
            go.Bar(
                x=importance_df["importance"],
                y=importance_df["feature"],
                orientation="h",
                name="Feature Importance",
                marker_color="lightcoral",
            ),
            row=2,
            col=1,
        )

    # 5. Precision vs Recall scatter
    precisions = [all_metrics[name]["precision"] for name in model_names]
    recalls = [all_metrics[name]["recall"] for name in model_names]

    fig.add_trace(
        go.Scatter(
            x=recalls,
            y=precisions,
            mode="markers+text",
            text=model_names,
            textposition="top center",
            marker=dict(
                size=15,
                color=[
                    "gold" if name == best_model_name else "lightblue"
                    for name in model_names
                ],
            ),
            name="Models",
        ),
        row=2,
        col=2,
    )

    # 6. Summary table
    summary_data = []
    for name in model_names:
        metrics = all_metrics[name]
        summary_data.append(
            [
                name,
                f"{metrics['accuracy']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}",
            ]
        )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
                fill_color="lightblue",
            ),
            cells=dict(values=list(zip(*summary_data)), fill_color="white"),
        ),
        row=2,
        col=3,
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"📈 Model Evaluation Dashboard - Best: {best_model_name}",
        showlegend=False,
    )

    # Calculate additional metrics for best model
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # Create detailed metrics summary
    metrics_html = f"""
    <div style="margin: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <h3>🏆 Best Model: {best_model_name}</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
            <div>
                <h4>📊 Classification Metrics</h4>
                <ul>
                    <li><strong>Accuracy:</strong> {all_metrics[best_model_name]['accuracy']:.3f}</li>
                    <li><strong>Precision:</strong> {all_metrics[best_model_name]['precision']:.3f}</li>
                    <li><strong>Recall:</strong> {all_metrics[best_model_name]['recall']:.3f}</li>
                    <li><strong>F1-Score:</strong> {all_metrics[best_model_name]['f1_score']:.3f}</li>
                </ul>
            </div>
            <div>
                <h4>🎯 Error Analysis</h4>
                <ul>
                    <li><strong>False Positive Rate:</strong> {fpr:.3f}</li>
                    <li><strong>False Negative Rate:</strong> {fnr:.3f}</li>
                    <li><strong>True Positives:</strong> {tp}</li>
                    <li><strong>True Negatives:</strong> {tn}</li>
                </ul>
            </div>
        </div>
    </div>
    """

    # Combine into final HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>📈 Model Evaluation Report</h1>
        {metrics_html}
        <div id="evaluation-dashboard">{fig.to_html(include_plotlyjs=False, div_id="evaluation-dashboard")}</div>
    </body>
    </html>
    """

    print("✅ Model evaluation report created")
    return HTMLString(html_report)
