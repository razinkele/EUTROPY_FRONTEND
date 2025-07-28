# Ecological Model Visualization App
# Install required packages if not already installed:
# pip install shiny pandas plotly numpy shinywidgets shinyswatch

from shiny import App, ui, render
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from datetime import datetime
from shiny.ui import HTML
from shinywidgets import output_widget, render_widget
import shinyswatch
import base64
import os

# Function to encode image as base64 for embedding
def get_logo_base64():
    logo_path = os.path.join(os.path.dirname(__file__), "logo.svg")
    try:
        with open(logo_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        # For SVG, we can embed directly or convert to base64
        encoded = base64.b64encode(svg_content.encode()).decode()
        return f"data:image/svg+xml;base64,{encoded}"
    except FileNotFoundError:
        # Fallback if logo not found
        return ""

# Define the UI
app_ui = ui.page_fluid(
    ui.div(
        ui.tags.div(
            ui.tags.img(
                src=get_logo_base64(),
                style="height: 60px; margin-right: 15px; vertical-align: middle;"
            ),
            ui.tags.h1(
                "EUTROPY",
                style="display: inline-block; vertical-align: middle; margin: 0; color: #2c3e50;"
            ),
            style="text-align: center; padding: 10px 0; border-bottom: 2px solid #ecf0f1; margin-bottom: 20px;"
        )
    ),
    ui.layout_sidebar(
        # Sidebar
        ui.sidebar(
            ui.card(
                ui.card_header("Settings"),
                shinyswatch.theme_picker_ui(),
                ui.hr(),
                ui.input_file("model_data", "Upload model results CSV:", accept=[".csv"]),
                ui.input_file("observation_data", "Upload chlorophyll a observations CSV:", accept=[".csv"]),
                ui.input_date_range(
                    "date_range",
                    "Select Date Range:",
                    start=datetime(2012, 1, 1),  # Default start date - adjust as needed
                    end=datetime(2016, 12, 31),  # Default end date - adjust as needed
                ),
                ui.input_select(
                    "plot_type",
                    "Select Plot Type:",
                    {
                        "time_series": "Time Series Comparison",
                        "scatter": "Model vs Observation Scatter",
                        "residuals": "Residuals Plot",
                        "heatmap": "Spatial Heatmap (if coordinates available)"
                    }
                ),
                ui.input_checkbox("show_statistics", "Show Statistics", True),
                ui.hr(),
                ui.h4("Statistics:"),
                ui.output_ui("statistics_output")
            )
        ),
        
        # Main content
        ui.navset_tab(
            ui.nav_panel("Visualization", 
                output_widget("main_plot"),
            ),
            ui.nav_panel("Data Table", 
                ui.output_data_frame("data_table"),
            ),
            ui.nav_panel("Model Performance", 
                output_widget("performance_plot"),
                ui.output_ui("performance_metrics")
            ),
            ui.nav_panel("About",
                ui.tags.div(
                    ui.tags.h3("EUTROPY: Ecological Model Visualization"),
                    ui.tags.p("""
                        This application is designed for visualizing and analyzing ecological model outputs,
                        specifically for chlorophyll a concentrations in aquatic systems. It allows comparison
                        between model predictions and field observations.
                    """),
                    ui.tags.h4("Features:"),
                    ui.tags.ul(
                        ui.tags.li("Time series visualization of chlorophyll a dynamics"),
                        ui.tags.li("Model vs Observation comparison plots"),
                        ui.tags.li("Statistical analysis of model performance"),
                        ui.tags.li("Spatial visualization (when coordinates are available)"),
                        ui.tags.li("Customizable themes")
                    ),
                    style="padding: 20px;"
                )
            ),
            id="analysis_tabs"
        )
    ),
    # Use the default bootstrap theme as base, then let shinyswatch handle the theme switching
    theme=shinyswatch.theme.flatly
)

# Define the server
def server(input, output, session):    
    # Add theme picker server
    shinyswatch.theme_picker_server()
    
    # Reactive functions to load data
    def load_model_data():
        model_file = input.model_data()
        
        # For testing - use the provided file if no file uploaded
        if model_file is None:
            try:
                df = pd.read_csv("boxOut_19.csv", parse_dates=['date'])
                # Explicitly ensure date is datetime type
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                return df
            except:
                return pd.DataFrame()
        
        try:
            # Correct indexing for file upload
            df = pd.read_csv(model_file[0]['datapath'])
            df['Cpy'] = df['Cpy']/50*1000  # Ensure Cpy is float
            # Explicitly ensure date column is in datetime format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            return df
        except Exception as e:
            print(f"Error loading model data: {e}")
            return pd.DataFrame()
    
    def load_observation_data():
        obs_file = input.observation_data()
        
        # For testing - use the provided file if no file uploaded
        if obs_file is None:
            try:
                df = pd.read_csv("Vidmares_EPA_Chl.csv", parse_dates=['date'])
                # Explicitly ensure date is datetime type
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                return df
            except:
                return pd.DataFrame()
        
        try:
            # Correct indexing for file upload
            df = pd.read_csv(obs_file[0]['datapath'])
            # Explicitly ensure date column is in datetime format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            return df
        except Exception as e:
            print(f"Error loading observation data: {e}")
            return pd.DataFrame()
    
    # Function to merge model and observation data
    def merge_data():
        model_df = load_model_data()
        obs_df = load_observation_data()
        
        if model_df.empty or obs_df.empty:
            return pd.DataFrame()
        
        # These files use 'Cpy' as the chlorophyll parameter
        if 'Cpy' in model_df.columns:
            model_df = model_df.rename(columns={'Cpy': 'model_chl'})
        
        if 'Cpy' in obs_df.columns:
            obs_df = obs_df.rename(columns={'Cpy': 'obs_chl'})
            
        # Merge on date
        if 'date' in model_df.columns and 'date' in obs_df.columns:
            # Double-check date columns are datetime
            model_df['date'] = pd.to_datetime(model_df['date'], errors='coerce')
            obs_df['date'] = pd.to_datetime(obs_df['date'], errors='coerce')
            
            # Check for invalid date values
            model_df = model_df.dropna(subset=['date'])
            obs_df = obs_df.dropna(subset=['date'])
            
            merged = pd.merge(model_df, obs_df, on='date', how='outer', suffixes=('_model', '_obs'))
        else:
            # If no date column, try to create a matching index
            merged = pd.DataFrame()
            
        return merged
    
    # Function to filter data by date range
    def filtered_data():
        data = merge_data()
        if data.empty or 'date' not in data.columns:
            return data
        
        # Update date range based on actual data
        if data['date'].min() != data['date'].max():
            # Only update if not first run
            if hasattr(filtered_data, 'initialized'):
                pass
            else:
                min_date = data['date'].min()
                max_date = data['date'].max()
                ui.update_date_range(
                    "date_range",
                    start=min_date.to_pydatetime().date(),
                    end=max_date.to_pydatetime().date()
                )
                filtered_data.initialized = True
        
        # Convert Python date objects to pandas datetime for comparison
        start_date = pd.to_datetime(input.date_range()[0])
        end_date = pd.to_datetime(input.date_range()[1])
        
        return data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    
    # Render main plot using Plotly
    @output
    @render_widget
    def main_plot():
        data = filtered_data()
        if data.empty:
            # Empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="Please upload both model and observation data files",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            return fig
        
        plot_type = input.plot_type()
        
        if plot_type == "time_series":
            # Time series plot
            fig = go.Figure()
            
            # Ensure data is sorted by date for proper line display
            data = data.sort_values('date')
            
            # Create a copy of the data to prevent modifying the original
            plot_data = data.copy()
            
            # Format dates as strings for x-axis (this is a workaround for the decimal display issue)
            if 'date' in plot_data.columns:
                # Convert to string format that Plotly will interpret as dates
                plot_data['date_str'] = plot_data['date'].dt.strftime('%Y-%m-%d')
            
            if 'model_chl' in plot_data.columns:
                fig.add_trace(go.Scatter(
                    x=plot_data['date_str'] if 'date_str' in plot_data.columns else plot_data['date'], 
                    y=plot_data['model_chl'],
                    mode='lines',
                    name='Model Chlorophyll a',
                    line=dict(color='blue')
                ))
                
            if 'obs_chl' in plot_data.columns:
                fig.add_trace(go.Scatter(
                    x=plot_data['date_str'] if 'date_str' in plot_data.columns else plot_data['date'], 
                    y=plot_data['obs_chl'],
                    mode='markers',
                    name='Observed Chlorophyll a',
                    marker=dict(color='red', size=8)
                ))
            
            # Simple date formatting without range slider
            fig.update_layout(
                title='Chlorophyll a Dynamics: Model vs Observations',
                xaxis_title='Date',
                yaxis_title='Chlorophyll a (μg/L)',
                legend=dict(y=0.99, x=0.01),
                hovermode='closest',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            # Use categorical axis with formatted date labels
            fig.update_xaxes(
                type='category',  # Use category type for explicit control
                categoryorder='array',
                categoryarray=plot_data['date_str'] if 'date_str' in plot_data.columns else plot_data['date'],
                tickangle=45,
                tickfont=dict(size=10),
                nticks=12  # Limit the number of ticks to avoid overcrowding
            )
            
        elif plot_type == "scatter":
            # Scatter plot of model vs observations
            if 'model_chl' in data.columns and 'obs_chl' in data.columns:
                # Drop NaN values
                plot_data = data.dropna(subset=['model_chl', 'obs_chl'])
                
                fig = go.Figure()
                
                # Add scatter plot
                fig.add_trace(go.Scatter(
                    x=plot_data['obs_chl'],
                    y=plot_data['model_chl'],
                    mode='markers',
                    marker=dict(color='blue', size=8, opacity=0.7),
                    name='Data Points'
                ))
                
                # Add 1:1 line
                min_val = min(plot_data['obs_chl'].min(), plot_data['model_chl'].min())
                max_val = max(plot_data['obs_chl'].max(), plot_data['model_chl'].max())
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='black', dash='dash', width=1),
                    name='1:1 Line'
                ))
                
                # Add regression line
                if len(plot_data) > 1:
                    z = np.polyfit(plot_data['obs_chl'], plot_data['model_chl'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(plot_data['obs_chl'].min(), plot_data['obs_chl'].max(), 100)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=p(x_range),
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'Regression: y = {z[0]:.3f}x + {z[1]:.3f}'
                    ))
                    
                    # Calculate R²
                    corr_matrix = np.corrcoef(plot_data['obs_chl'], plot_data['model_chl'])
                    r_squared = corr_matrix[0, 1]**2
                    
                    # Add R² annotation
                    fig.add_annotation(
                        x=0.05,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f'R² = {r_squared:.3f}',
                        showarrow=False,
                        font=dict(size=12),
                        align="left"
                    )
                
                fig.update_layout(
                    title='Model vs Observation Comparison',
                    xaxis_title='Observed Chlorophyll a (μg/L)',
                    yaxis_title='Modeled Chlorophyll a (μg/L)',
                    legend=dict(y=0.99, x=0.01),
                    hovermode='closest',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
            else:
                # Create empty plot with message
                fig = go.Figure()
                fig.add_annotation(
                    text="Missing required data columns",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14)
                )
                fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                
        elif plot_type == "residuals":
            # Residuals plot
            if 'model_chl' in data.columns and 'obs_chl' in data.columns:
                # Drop NaN values
                plot_data = data.dropna(subset=['model_chl', 'obs_chl'])
                
                # Calculate residuals
                plot_data['residuals'] = plot_data['model_chl'] - plot_data['obs_chl']
                
                fig = go.Figure()
                
                # Add scatter plot
                fig.add_trace(go.Scatter(
                    x=plot_data['obs_chl'],
                    y=plot_data['residuals'],
                    mode='markers',
                    marker=dict(color='blue', size=8, opacity=0.7),
                    name='Residuals'
                ))
                
                # Add zero line
                x_range = [plot_data['obs_chl'].min(), plot_data['obs_chl'].max()]
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Zero Line'
                ))
                
                fig.update_layout(
                    title='Residuals Plot',
                    xaxis_title='Observed Chlorophyll a (μg/L)',
                    yaxis_title='Residuals (Model - Observed)',
                    legend=dict(y=0.99, x=0.01),
                    hovermode='closest',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
            else:
                # Create empty plot with message
                fig = go.Figure()
                fig.add_annotation(
                    text="Missing required data columns",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14)
                )
                fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                
        elif plot_type == "heatmap":
            # Check if spatial coordinates are available
            lat_cols = [col for col in data.columns if 'lat' in col.lower()]
            lon_cols = [col for col in data.columns if 'lon' in col.lower() or 'long' in col.lower()]
            
            if lat_cols and lon_cols and 'model_chl' in data.columns:
                lat_col = lat_cols[0]
                lon_col = lon_cols[0]
                
                # Create a scatter plot colored by chlorophyll values using Plotly
                fig = px.scatter_mapbox(
                    data,
                    lat=lat_col,
                    lon=lon_col,
                    color='model_chl',
                    color_continuous_scale='Viridis',
                    hover_name='date' if 'date' in data.columns else None,
                    hover_data=['model_chl'],
                    title='Spatial Distribution of Chlorophyll a',
                    mapbox_style="open-street-map",
                    opacity=0.8,
                    size_max=15
                )
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    coloraxis_colorbar=dict(title='Chlorophyll a (μg/L)')
                )
                
            else:
                # Fallback to regular scatter plot if mapbox not ideal
                if lat_cols and lon_cols and 'model_chl' in data.columns:
                    lat_col = lat_cols[0]
                    lon_col = lon_cols[0]
                    
                    fig = px.scatter(
                        data,
                        x=lon_col, 
                        y=lat_col,
                        color='model_chl',
                        color_continuous_scale='Viridis',
                        title='Spatial Distribution of Chlorophyll a',
                        labels={
                            lon_col: 'Longitude',
                            lat_col: 'Latitude',
                            'model_chl': 'Chlorophyll a (μg/L)'
                        }
                    )
                    
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                else:
                    # Create empty plot with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Spatial coordinates not found in data",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=14)
                    )
                    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        
        # Add grid to all plots
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    
    # Render performance plot using Plotly
    @output
    @render_widget
    def performance_plot():
        data = filtered_data()
        if data.empty or 'model_chl' not in data.columns or 'obs_chl' not in data.columns:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for performance analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            return fig
        
        # Drop NaN values
        plot_data = data.dropna(subset=['model_chl', 'obs_chl'])
        
        if len(plot_data) < 2:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough matching data points for analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            return fig
        
        # Calculate residuals
        residuals = plot_data['model_chl'] - plot_data['obs_chl']
        
        # Create subplot structure
        fig = go.Figure(data=[])
        
        # Add subplot titles
        fig.update_layout(
            title='Model Performance Analysis',
            grid=dict(rows=1, columns=2)
        )
        
        # Add first subplot - Histogram of residuals
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=15,
                marker=dict(color='skyblue', line=dict(color='black', width=1)),
                opacity=0.7,
                name='Residual Distribution',
                xaxis='x',
                yaxis='y'
            )
        )
        
        # Add zero line to histogram
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, len(residuals)/3],  # Approximate height for visibility
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Zero Line',
                xaxis='x',
                yaxis='y'
            )
        )
        
        # Create QQ plot data
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.quantile(np.random.normal(0, np.std(residuals), len(residuals)), 
                                        np.linspace(0, 1, len(residuals)))
        
        # Add second subplot - QQ plot
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode='markers',
                marker=dict(color='blue', size=8, opacity=0.7),
                name='Q-Q Plot',
                xaxis='x2',
                yaxis='y2'
            )
        )
        
        # Add reference line to QQ plot
        min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
        max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Reference Line',
                xaxis='x2',
                yaxis='y2'
            )
        )
        
        # Update layout for subplots
        fig.update_layout(
            grid=dict(rows=1, columns=2, pattern='independent'),
            xaxis=dict(title='Residuals (Model - Observed)', domain=[0, 0.45]),
            yaxis=dict(title='Frequency'),
            xaxis2=dict(title='Theoretical Quantiles', domain=[0.55, 1.0]),
            yaxis2=dict(title='Sample Quantiles'),
            legend=dict(y=0.99, x=0.01),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Add grid to all plots
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # Add subplot titles
        fig.add_annotation(
            x=0.225, y=1.05,
            xref='paper', yref='paper',
            text='Histogram of Residuals',
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.add_annotation(
            x=0.775, y=1.05,
            xref='paper', yref='paper',
            text='Q-Q Plot (Normality Check)',
            showarrow=False,
            font=dict(size=14)
        )
        
        return fig
    
    # Render statistics
    @output
    @render.text
    def statistics_output():
        if not input.show_statistics():
            return ""
            
        data = filtered_data()
        if data.empty or 'model_chl' not in data.columns or 'obs_chl' not in data.columns:
            return "Insufficient data for statistical analysis"
        
        # Drop NaN values
        clean_data = data.dropna(subset=['model_chl', 'obs_chl'])
        
        if len(clean_data) < 2:
            return "Not enough matching data points for statistical analysis"
        
        # Calculate statistics
        residuals = clean_data['model_chl'] - clean_data['obs_chl']
        mean_error = residuals.mean()
        rmse = np.sqrt((residuals ** 2).mean())
        mae = np.abs(residuals).mean()
        
        # Calculate R²
        corr_matrix = np.corrcoef(clean_data['obs_chl'], clean_data['model_chl'])
        r_squared = corr_matrix[0, 1]**2
        
        # Calculate bias
        bias = mean_error / clean_data['obs_chl'].mean() * 100  # percent bias
        
        stats_text = (
            f"Number of data points: {len(clean_data)} \n\n "
            f"Model Mean: {clean_data['model_chl'].mean():.3f} μg/L\n "
            f"Observation Mean: {clean_data['obs_chl'].mean():.3f} μg/L\n\n "
            f"Mean Error: {mean_error:.3f} μg/L\n "
            f"Root Mean Square Error (RMSE): {rmse:.3f} μg/L\n "
            f"Mean Absolute Error (MAE): {mae:.3f} μg/L\n "
            f"R²: {r_squared:.3f}\n "
            f"Percent Bias: {bias:.2f}%\n "
        )
        
        # return stats_text
        stats_html = stats_text.replace("\n", "<br>")
        return HTML(stats_html)

    # Render performance metrics
    @output
    @render.text
    def performance_metrics():
        data = filtered_data()
        if data.empty or 'model_chl' not in data.columns or 'obs_chl' not in data.columns:
            return "Insufficient data for performance metrics"
        
        # Drop NaN values
        clean_data = data.dropna(subset=['model_chl', 'obs_chl'])
        
        if len(clean_data) < 2:
            return "Not enough matching data points for performance metrics"
        
        # Calculate additional performance metrics
        residuals = clean_data['model_chl'] - clean_data['obs_chl']
        
        # Nash-Sutcliffe Efficiency
        ss_res = np.sum((clean_data['obs_chl'] - clean_data['model_chl'])**2)
        ss_tot = np.sum((clean_data['obs_chl'] - clean_data['obs_chl'].mean())**2)
        nse = 1 - (ss_res / ss_tot if ss_tot != 0 else 0)
        
        # Index of Agreement
        sum_squared_diff = np.sum((np.abs(clean_data['model_chl'] - clean_data['obs_chl'].mean()) + 
                                  np.abs(clean_data['obs_chl'] - clean_data['obs_chl'].mean()))**2)
        d = 1 - (ss_res / sum_squared_diff if sum_squared_diff != 0 else 0)
        
        # Percent Bias
        pbias = 100 * np.sum(residuals) / np.sum(clean_data['obs_chl'])
        
        metrics_text = (
            "Model Performance Metrics:\n\n"
            f"Nash-Sutcliffe Efficiency (NSE): {nse:.3f}\n"
            f"  (1 = perfect fit, <0 = worse than mean)\n\n"
            f"Index of Agreement (d): {d:.3f}\n"
            f"  (1 = perfect agreement, 0 = no agreement)\n\n"
            f"Percent Bias (PBIAS): {pbias:.2f}%\n"
            f"  (0 = no bias, + = underestimation, - = overestimation)\n\n"
            
        )
        
        metrics_html = metrics_text.replace("\n", "<br>")
        return HTML(metrics_html)
    
    # Render data table
    @output
    @render.data_frame
    def data_table():
        data = filtered_data()
        if data.empty:
            return pd.DataFrame()
        
        # Select relevant columns for display
        display_cols = ['date']
        
        if 'model_chl' in data.columns:
            display_cols.append('model_chl')
        
        if 'obs_chl' in data.columns:
            display_cols.append('obs_chl')
            
        # Add residuals if both columns exist
        if 'model_chl' in data.columns and 'obs_chl' in data.columns:
            data = data.copy()
            data['residual'] = data['model_chl'] - data['obs_chl']
            display_cols.append('residual')
        
        # Add location information if available
        for col in data.columns:
            if any(x in col.lower() for x in ['lat', 'lon', 'long', 'station', 'site']):
                display_cols.append(col)
        
        return data[display_cols].sort_values('date')

# Create and run the app
app = App(app_ui, server)

# To run the app locally:
if __name__ == "__main__":
    app.run()