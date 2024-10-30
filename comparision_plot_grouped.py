'''import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

# Read CPU and GPU data
cpu_data = pd.read_csv('merged_simulation_data.csv')
gpu_data = pd.read_csv('merged_simulation_data_cpu.csv')

def group_by_magnitude(value):
    if value == 0:
        return 0
    return np.floor(np.log10(np.abs(value)))

def create_comparison_plot(cpu_data, gpu_data, x_col, y_col, title):
    fig = go.Figure()

    # Group CPU and GPU data by magnitude
    cpu_grouped = cpu_data[y_col].apply(group_by_magnitude)
    gpu_grouped = gpu_data[y_col].apply(group_by_magnitude)

    # Add traces for CPU and GPU data
    fig.add_trace(go.Scatter(x=cpu_data[x_col], y=cpu_grouped, mode='lines', name=f'CPU {y_col}'))
    fig.add_trace(go.Scatter(x=gpu_data[x_col], y=gpu_grouped, mode='lines', name=f'GPU {y_col}'))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=f'{y_col} (Magnitude)',
        height=600,
        width=800
    )

    return fig

# List of columns to plot
columns_to_plot = ['r', 'z', 'vr', 'vz', 'Erfi_after', 'Ezfi_after']

# Create and show plots
for column in columns_to_plot:
    fig = create_comparison_plot(cpu_data, gpu_data, 'timestep', column, f'{column} vs Timestep (Grouped by Magnitude)')
    fig.show()'''


#ROLLING AVERAGE

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

# Read CPU and GPU data
cpu_data = pd.read_csv('merged_simulation_data_cpu.csv')
gpu_data = pd.read_csv('merged_simulation_data.csv')

def calculate_relative_error(cpu_val, gpu_val):
    if cpu_val == 0 and gpu_val == 0:
        return 0
    elif cpu_val == 0:
        return 1  
    else:
        return abs((cpu_val - gpu_val) / cpu_val)

def moving_average(data, window_size):
    return data.rolling(window=window_size, center=True).mean()

def create_comparison_plot(cpu_data, gpu_data, x_col, y_col, title):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f"Absolute Difference", f"Relative Error"))

    # Calculate differences and relative error
    diff = cpu_data[y_col] - gpu_data[y_col]
    rel_error = [calculate_relative_error(cpu, gpu) for cpu, gpu in zip(cpu_data[y_col], gpu_data[y_col])]

    # Apply moving average
    window_size = 50  # Adjust this value as needed
    diff_ma = moving_average(diff, window_size)
    rel_error_ma = moving_average(pd.Series(rel_error), window_size)

    # Plot absolute difference
    fig.add_trace(go.Scatter(x=cpu_data[x_col], y=np.log10(np.abs(diff) + 1e-20), mode='lines', name='Log Abs Diff'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=cpu_data[x_col], y=np.log10(np.abs(diff_ma) + 1e-20), mode='lines', name='Log Abs Diff (MA)',
                             line=dict(color='red')), row=1, col=1)

    # Plot relative error
    fig.add_trace(go.Scatter(x=cpu_data[x_col], y=rel_error, mode='lines', name='Relative Error'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=cpu_data[x_col], y=rel_error_ma, mode='lines', name='Relative Error (MA)',
                             line=dict(color='red')), row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=800, title_text=f"{y_col} vs {x_col} Comparison")
    fig.update_xaxes(title_text=x_col, row=2, col=1)
    fig.update_yaxes(title_text="Log10 Absolute Difference", row=1, col=1)
    fig.update_yaxes(title_text="Relative Error", row=2, col=1)

    return fig

# List of columns to plot
columns_to_plot = ['r', 'z', 'vr', 'vz', 'Erfi_after', 'Ezfi_after']

# Create and show plots
for column in columns_to_plot:
    fig = create_comparison_plot(cpu_data, gpu_data, 'timestep', column, f'{column} vs Timestep Comparison')
    fig.show()