'''import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Read CPU and GPU data
cpu_data = pd.read_csv('merged_simulation_data_cpu.csv')
gpu_data = pd.read_csv('merged_simulation_data.csv')

def create_comparison_plot(cpu_data, gpu_data, x_col, y_cols, title, y_axis_title):
    fig = make_subplots(rows=len(y_cols), cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=[f"{y_col} vs {x_col}" for y_col in y_cols])
    
    for i, y_col in enumerate(y_cols, start=1):
        cpu_trace = go.Scatter(x=cpu_data[x_col], y=cpu_data[y_col], mode='lines', name=f'CPU {y_col}')
        gpu_trace = go.Scatter(x=gpu_data[x_col], y=gpu_data[y_col], mode='lines', name=f'GPU {y_col}')
        
        fig.add_trace(cpu_trace, row=i, col=1)
        fig.add_trace(gpu_trace, row=i, col=1)
        
        # Set y-axis range dynamically
        y_min = min(cpu_data[y_col].min(), gpu_data[y_col].min())
        y_max = max(cpu_data[y_col].max(), gpu_data[y_col].max())
        y_range = y_max - y_min
        fig.update_yaxes(range=[y_min - 0.1*y_range, y_max + 0.1*y_range], row=i, col=1)
    
    fig.update_layout(
        height=300*len(y_cols),
        width=800,
        title_text=title,
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="linear"
        )
    )
    
    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y_axis_title)
    
    return fig

# Create position vs timestep plots
pos_fig = create_comparison_plot(cpu_data, gpu_data, 'timestep', ['r', 'z'], 'Position vs Timestep', 'Position')

# Create velocity vs timestep plots
vel_fig = create_comparison_plot(cpu_data, gpu_data, 'timestep', ['vr', 'vz'], 'Velocity vs Timestep', 'Velocity')

# Create Erfi after vs timestep plot
erfi_fig = create_comparison_plot(cpu_data, gpu_data, 'timestep', ['Erfi_after'], 'Erfi After vs Timestep', 'Erfi After')

# Create Ezfi after vs timestep plot
ezfi_fig = create_comparison_plot(cpu_data, gpu_data, 'timestep', ['Ezfi_after'], 'Ezfi After vs Timestep', 'Ezfi After')

# Show plots
pos_fig.show()
vel_fig.show()
erfi_fig.show()
ezfi_fig.show()'''

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

# Read CPU and GPU data
cpu_data = pd.read_csv('merged_simulation_data_cpu.csv')
gpu_data = pd.read_csv('merged_simulation_data.csv')

def create_comparison_plot(cpu_data, gpu_data, x_col, y_col, title):
    fig = go.Figure()

    # Add traces for CPU and GPU data
    fig.add_trace(go.Scatter(x=cpu_data[x_col], y=np.abs(cpu_data[y_col]), mode='lines', name=f'CPU {y_col}'))
    fig.add_trace(go.Scatter(x=gpu_data[x_col], y=np.abs(gpu_data[y_col]), mode='lines', name=f'GPU {y_col}'))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        yaxis_type="log",
        height=600,
        width=800
    )

    return fig

# List of columns to plot
columns_to_plot = ['r', 'z', 'vr', 'vz', 'Erfi_after', 'Ezfi_after']

# Create and show plots
for column in columns_to_plot:
    fig = create_comparison_plot(cpu_data, gpu_data, 'timestep', column, f'{column} vs Timestep')
    fig.show()