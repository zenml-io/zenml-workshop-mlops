
"""
Utility functions for generating timeseries forecasting reports and visualizations
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any
from zenml.types import HTMLString


def generate_timeseries_exploration_report(df: pd.DataFrame) -> HTMLString:
    """Generate comprehensive timeseries data exploration report with interactive visualizations.

    Args:
        df: Timeseries DataFrame with product demand data

    Returns:
        HTML report with timeseries visualizations
    """
    print("🔍 Creating timeseries exploration report...")

    # Get first few products for visualization
    products = df['product_id'].unique()[:4]  # Show first 4 products
    
    # Create subplots for main analysis
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Demand Over Time (First 4 Products)",
            "Seasonal Patterns (Average by Month)",
            "Weekly Patterns (Average by Day of Week)", 
            "Demand Distribution by Product",
            "Quarterly Trends",
            "Weekend vs Weekday Demand",
        ),
        specs=[
            [{"colspan": 2}, None],
            [{"type": "bar"}, {"type": "violin"}],
            [{"type": "bar"}, {"type": "box"}],
        ],
    )

    # 1. Time series plot for first 4 products
    colors = ['blue', 'red', 'green', 'orange']
    for i, product_id in enumerate(products):
        product_data = df[df['product_id'] == product_id].sort_values('date')
        fig.add_trace(
            go.Scatter(
                x=product_data['date'],
                y=product_data['demand'],
                mode='lines',
                name=product_id,
                line=dict(color=colors[i]),
            ),
            row=1, col=1
        )

    # 2. Seasonal patterns (monthly)
    monthly_avg = df.groupby('month')['demand'].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=monthly_avg['month'],
            y=monthly_avg['demand'],
            name="Monthly Average",
            marker_color='lightblue',
        ),
        row=2, col=1
    )

    # 3. Demand distribution by product (violin plot)
    for i, product_id in enumerate(products):
        product_data = df[df['product_id'] == product_id]
        fig.add_trace(
            go.Violin(
                y=product_data['demand'],
                name=product_id,
                box_visible=True,
                meanline_visible=True,
            ),
            row=2, col=2
        )

    # 4. Weekly patterns
    weekly_avg = df.groupby('day_of_week')['demand'].mean().reset_index()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly_avg['day_name'] = [day_names[i] for i in weekly_avg['day_of_week']]
    
    fig.add_trace(
        go.Bar(
            x=weekly_avg['day_name'],
            y=weekly_avg['demand'],
            name="Weekly Average",
            marker_color='lightgreen',
        ),
        row=3, col=1
    )

    # 5. Weekend vs weekday comparison
    weekend_comparison = df.groupby('is_weekend')['demand'].mean().reset_index()
    weekend_comparison['day_type'] = weekend_comparison['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
    
    fig.add_trace(
        go.Box(
            x=df['is_weekend'].map({True: 'Weekend', False: 'Weekday'}),
            y=df['demand'],
            name="Weekend vs Weekday",
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        height=1000,
        title_text="📊 Timeseries Demand Data Exploration Dashboard",
        showlegend=True,
    )

    # Create insights summary
    total_products = len(df['product_id'].unique())
    date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    avg_demand = df['demand'].mean()
    demand_std = df['demand'].std()
    weekend_avg = df[df['is_weekend'] == True]['demand'].mean()
    weekday_avg = df[df['is_weekend'] == False]['demand'].mean()
    
    insights_html = f"""
    <div style="margin: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <h3>🔍 Key Timeseries Insights</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
            <div>
                <h4>📊 Data Overview</h4>
                <ul>
                    <li><strong>Total products:</strong> {total_products}</li>
                    <li><strong>Date range:</strong> {date_range}</li>
                    <li><strong>Total observations:</strong> {len(df):,}</li>
                    <li><strong>Days per product:</strong> {len(df) // total_products}</li>
                </ul>
            </div>
            <div>
                <h4>📈 Demand Statistics</h4>
                <ul>
                    <li><strong>Average demand:</strong> {avg_demand:.1f}</li>
                    <li><strong>Demand std deviation:</strong> {demand_std:.1f}</li>
                    <li><strong>Weekend average:</strong> {weekend_avg:.1f}</li>
                    <li><strong>Weekday average:</strong> {weekday_avg:.1f}</li>
                </ul>
            </div>
        </div>
        <div style="margin-top: 15px;">
            <h4>🎯 Key Patterns Identified</h4>
            <ul>
                <li><strong>Seasonality:</strong> Clear monthly and quarterly patterns visible</li>
                <li><strong>Weekly cycle:</strong> Weekend vs weekday demand differences</li>
                <li><strong>Trend:</strong> Overall demand trends across time period</li>
                <li><strong>Product variation:</strong> Different baseline demands per product</li>
            </ul>
        </div>
    </div>
    """

    # Combine into final HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Timeseries Exploration Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>🔍 Timeseries Demand Data Exploration Report</h1>
        {insights_html}
        <div id="timeseries-dashboard">{fig.to_html(include_plotlyjs=False, div_id="timeseries-dashboard")}</div>
    </body>
    </html>
    """

    print("✅ Timeseries exploration report created")
    return HTMLString(html_report)


def generate_forecast_validation_report(
    predictions_df: pd.DataFrame,
    validation_metrics: Dict[str, Any]
) -> HTMLString:
    """Generate comprehensive forecast validation report with interactive visualizations.

    Args:
        predictions_df: DataFrame with actual and predicted demand values
        validation_metrics: Dictionary containing validation metrics

    Returns:
        HTML report with forecast validation visualizations
    """
    print("📈 Creating forecast validation report...")

    # Get products for visualization (show all if <=6, otherwise first 6)
    all_products = predictions_df['product_id'].unique()
    display_products = all_products[:6] if len(all_products) > 6 else all_products
    
    # Create comprehensive validation report
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Actual vs Predicted (First 6 Products)",
            "Prediction Errors Distribution",
            "Per-Product Performance (MAE)",
            "Residuals vs Predicted",
            "Batch Processing Results",
            "Error Analysis by Product",
        ),
        specs=[
            [{"colspan": 2}, None],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "box"}],
        ],
    )

    # 1. Actual vs Predicted time series for multiple products
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, product_id in enumerate(display_products):
        product_data = predictions_df[predictions_df['product_id'] == product_id].sort_values('date')
        
        # Actual values
        fig.add_trace(
            go.Scatter(
                x=product_data['date'],
                y=product_data['demand'],
                mode='lines',
                name=f'{product_id} (Actual)',
                line=dict(color=colors[i % len(colors)]),
            ),
            row=1, col=1
        )
        
        # Predicted values
        fig.add_trace(
            go.Scatter(
                x=product_data['date'],
                y=product_data['predicted_demand'],
                mode='lines',
                name=f'{product_id} (Predicted)',
                line=dict(color=colors[i % len(colors)], dash='dash'),
            ),
            row=1, col=1
        )

    # 2. Per-product MAE
    product_metrics = validation_metrics['product_metrics']
    product_maes = [(p['product_id'], p['mae']) for p in product_metrics]
    product_maes.sort(key=lambda x: x[1])  # Sort by MAE
    
    fig.add_trace(
        go.Bar(
            x=[p[0] for p in product_maes],
            y=[p[1] for p in product_maes],
            name="MAE by Product",
            marker_color='lightcoral',
        ),
        row=2, col=1
    )

    # 3. Residuals vs Predicted (scatter plot)
    residuals = predictions_df['demand'] - predictions_df['predicted_demand']
    fig.add_trace(
        go.Scatter(
            x=predictions_df['predicted_demand'],
            y=residuals,
            mode='markers',
            name="Residuals",
            marker=dict(color='blue', opacity=0.6),
        ),
        row=2, col=2
    )
    
    # Add zero line for residuals
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

    # 4. Batch processing results
    batch_metrics = predictions_df.groupby('batch_id').agg({
        'demand': 'mean',
        'predicted_demand': 'mean',
        'product_id': 'count'
    }).reset_index()
    batch_metrics.columns = ['batch_id', 'avg_actual', 'avg_predicted', 'n_products']
    
    fig.add_trace(
        go.Bar(
            x=batch_metrics['batch_id'],
            y=batch_metrics['n_products'],
            name="Products per Batch",
            marker_color='lightgreen',
        ),
        row=3, col=1
    )

    # 5. Error distribution by product (box plot)
    fig.add_trace(
        go.Box(
            x=predictions_df['product_id'],
            y=residuals,
            name="Error Distribution",
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        height=1200,
        title_text=f"📈 Forecast Validation Dashboard - Overall MAE: {validation_metrics['overall_mae']:.2f}",
        showlegend=True,
    )

    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Demand", row=1, col=1)
    fig.update_xaxes(title_text="Product ID", row=2, col=1)
    fig.update_yaxes(title_text="MAE", row=2, col=1)
    fig.update_xaxes(title_text="Predicted Demand", row=2, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=2)

    # Create detailed metrics summary
    overall_metrics = validation_metrics
    best_product = min(overall_metrics['product_metrics'], key=lambda x: x['mae'])
    worst_product = max(overall_metrics['product_metrics'], key=lambda x: x['mae'])
    
    metrics_html = f"""
    <div style="margin: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <h3>🎯 Forecast Validation Results</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div>
                <h4>📊 Overall Performance</h4>
                <ul>
                    <li><strong>MAE:</strong> {overall_metrics['overall_mae']:.2f}</li>
                    <li><strong>RMSE:</strong> {overall_metrics['overall_rmse']:.2f}</li>
                    <li><strong>R²:</strong> {overall_metrics['overall_r2']:.3f}</li>
                    <li><strong>MAPE:</strong> {overall_metrics['overall_mape']:.1f}%</li>
                </ul>
            </div>
            <div>
                <h4>🏆 Best Product</h4>
                <ul>
                    <li><strong>Product:</strong> {best_product['product_id']}</li>
                    <li><strong>MAE:</strong> {best_product['mae']:.2f}</li>
                    <li><strong>R²:</strong> {best_product['r2']:.3f}</li>
                    <li><strong>Predictions:</strong> {best_product['n_predictions']}</li>
                </ul>
            </div>
            <div>
                <h4>🎯 Worst Product</h4>
                <ul>
                    <li><strong>Product:</strong> {worst_product['product_id']}</li>
                    <li><strong>MAE:</strong> {worst_product['mae']:.2f}</li>
                    <li><strong>R²:</strong> {worst_product['r2']:.3f}</li>
                    <li><strong>Predictions:</strong> {worst_product['n_predictions']}</li>
                </ul>
            </div>
        </div>
        <div style="margin-top: 15px;">
            <h4>📦 Batch Processing Summary</h4>
            <ul>
                <li><strong>Total predictions:</strong> {overall_metrics['n_predictions']:,}</li>
                <li><strong>Products processed:</strong> {overall_metrics['n_products']}</li>
                <li><strong>Processing completed in batches</strong> for scalability demonstration</li>
                <li><strong>Production scaling:</strong> Ready for 100,000+ products with larger batch sizes</li>
            </ul>
        </div>
    </div>
    """

    # Combine into final HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Forecast Validation Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>📈 Forecast Validation Report</h1>
        {metrics_html}
        <div id="validation-dashboard">{fig.to_html(include_plotlyjs=False, div_id="validation-dashboard")}</div>
        
        <div style="margin: 20px; padding: 20px; background-color: #e8f4f8; border-radius: 10px;">
            <h3>🚀 Production Deployment Insights</h3>
            <p><strong>Workshop Configuration:</strong> This demo processes {overall_metrics['n_products']} products to simulate production scenarios with 100,000+ products.</p>
            <p><strong>Batch Processing:</strong> Products are processed in batches to demonstrate scalable forecasting patterns used in production.</p>
            <p><strong>Model Performance:</strong> The RandomForest regression model achieves MAE of {overall_metrics['overall_mae']:.2f} with R² of {overall_metrics['overall_r2']:.3f}.</p>
            <p><strong>Scaling Recommendations:</strong> For production, increase batch_size to 1000+ and implement parallel batch processing.</p>
        </div>
    </body>
    </html>
    """

    print("✅ Forecast validation report created")
    return HTMLString(html_report)