import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

def log_linear_model(X, a, b, c, d):
    """Log-linear model for computation time prediction."""
    x, y = X
    return np.exp(a * np.log(x) + b * np.log(y) + c) + d

def extract_unique_values(data):
    """Extract unique values from the nested JSON structure."""
    cell_sizes = sorted(set(int(key) for key in data.keys()))
    ion_sizes = sorted(set(int(key) for subdict in data.values() 
                         for key in subdict.keys()))
    shot_sizes = sorted(set(int(key) for subdict in data.values() 
                          for ion_dict in subdict.values() 
                          for key in ion_dict.keys()))
    return cell_sizes, ion_sizes, shot_sizes

def prepare_data_for_fitting(data, cell_sizes, ion_size, shot_sizes):
    """Prepare data for surface fitting."""
    X = []
    Y = []
    Z = []
    
    for cell in cell_sizes:
        for shot in shot_sizes:
            try:
                time = data[str(cell)][str(ion_size)][str(shot)]
                X.append(cell)
                Y.append(shot)
                Z.append(time)
            except KeyError:
                continue
    return np.array(X), np.array(Y), np.array(Z)

def get_data_ranges(X, Y, Z):
    """Get the ranges of the training data."""
    return {
        'cell_size': (min(X), max(X)),
        'shot_size': (min(Y), max(Y)),
        'time': (min(Z), max(Z))
    }

def create_prediction_plots(predictions):
    """Create visualization plots for each ion size."""
    ion_sizes = set(pred['ion_size'] for pred in predictions)
    
    for ion_size in ion_sizes:
        ion_preds = [p for p in predictions if p['ion_size'] == ion_size]
        if not ion_preds:
            continue
            
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get the training data
        ranges = ion_preds[0]['ranges']
        X_train = ion_preds[0]['X_train']
        Y_train = ion_preds[0]['Y_train']
        Z_train = ion_preds[0]['Z_train']
        
        # Plot training points
        scatter = ax.scatter(X_train, Y_train, Z_train, c='blue', marker='o', 
                          label='Training Data', alpha=0.6)
        
        # Create surface mesh
        X_mesh = np.linspace(ranges['cell_size'][0], ranges['cell_size'][1], 50)
        Y_mesh = np.linspace(ranges['shot_size'][0], ranges['shot_size'][1], 50)
        X_grid, Y_grid = np.meshgrid(X_mesh, Y_mesh)
        Z_grid = log_linear_model((X_grid, Y_grid), *ion_preds[0]['model_params'])
        
        # Plot surface
        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', 
                            alpha=0.3)
        
        # Plot predictions
        pred_X = [p['cell_size'] for p in ion_preds]
        pred_Y = [p['shot_size'] for p in ion_preds]
        pred_Z = [p['predicted_time'] for p in ion_preds]
        ax.scatter(pred_X, pred_Y, pred_Z, c='red', marker='*', s=100,
                 label='Predictions')
        
        ax.set_xlabel('Cell Size')
        ax.set_ylabel('Shot Size')
        ax.set_zlabel('Time (s)')
        ax.set_title(f'Model Fit and Predictions\nIon Size: {ion_size}')
        ax.legend()
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, label='Computation Time (s)')
        
        # Adjust view angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()

def predict_combinations(data, combinations):
    """Make predictions for multiple combinations."""
    predictions = []
    
    for ion_size, cell_size, shot_size in combinations:
        cell_sizes, _, shot_sizes = extract_unique_values(data)
        X, Y, Z = prepare_data_for_fitting(data, cell_sizes, ion_size, shot_sizes)
        ranges = get_data_ranges(X, Y, Z)
        
        if len(X) > 4:
            try:
                # Fit the model
                popt, pcov = curve_fit(log_linear_model, (X, Y), Z, 
                                     p0=[1, 1, 0, 0], maxfev=10000)
                
                # Make prediction
                predicted_time = log_linear_model((cell_size, shot_size), *popt)
                
                # Calculate confidence interval
                perr = np.sqrt(np.diag(pcov))
                log_std = np.sqrt(np.sum(perr**2))
                ci_lower = predicted_time * np.exp(-2 * log_std)
                ci_upper = predicted_time * np.exp(2 * log_std)
                
                # Check extrapolation
                cell_extrapolation = (cell_size < ranges['cell_size'][0] * 0.5 or 
                                    cell_size > ranges['cell_size'][1] * 2)
                shot_extrapolation = (shot_size < ranges['shot_size'][0] * 0.5 or 
                                    shot_size > ranges['shot_size'][1] * 2)
                
                # Check existing data
                existing_time = None
                try:
                    existing_time = data[str(cell_size)][str(ion_size)][str(shot_size)]
                except KeyError:
                    pass
                
                predictions.append({
                    'ion_size': ion_size,
                    'cell_size': cell_size,
                    'shot_size': shot_size,
                    'predicted_time': predicted_time,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'existing_time': existing_time,
                    'cell_extrapolation': cell_extrapolation,
                    'shot_extrapolation': shot_extrapolation,
                    'ranges': ranges,
                    'model_params': popt,
                    'X_train': X,
                    'Y_train': Y,
                    'Z_train': Z
                })
            except RuntimeError as e:
                print(f"Could not fit model for ion size {ion_size}: {str(e)}")
                continue
        else:
            print(f"Not enough data points for ion size {ion_size} to make predictions")
    
    return predictions

def main():
    # Load the data
    with open('computation_times.json', 'r') as f:
        data = json.load(f)

    # Get combinations to predict
    print("Enter combinations (ion_size,cell_size,shot_size), blank line to finish:")
    combinations = []
    while True:
        line = input().strip()
        if not line:
            break
        try:
            ion_size, cell_size, shot_size = map(int, line.split(','))
            combinations.append((ion_size, cell_size, shot_size))
        except ValueError:
            print("Invalid format, skipping this line")

    if not combinations:
        print("No valid combinations to predict.")
        return

    # Make predictions
    predictions = predict_combinations(data, combinations)

    # Display results
    print("\nPrediction Results:")
    print("-" * 100)
    print(f"{'Ion Size':<10} {'Cell Size':<12} {'Shot Size':<12} {'Predicted Time':<18} {'95% CI':<25} {'Existing Time':<15}")
    print("-" * 100)

    for pred in predictions:
        ci_range = f"({pred['ci_lower']:.2f}, {pred['ci_upper']:.2f})"
        existing = f"{pred['existing_time']:.2f}" if pred['existing_time'] is not None else "N/A"
        print(f"{pred['ion_size']:<10} {pred['cell_size']:<12} {pred['shot_size']:<12} "
              f"{pred['predicted_time']:.2f}s{' ':>8} {ci_range:<25} {existing:<15}")
        
        if pred['cell_extrapolation'] or pred['shot_extrapolation']:
            print("\nWarning: Prediction involves extrapolation!")
            print(f"Training data ranges for ion_size={pred['ion_size']}:")
            print(f"Cell size: {pred['ranges']['cell_size'][0]} to {pred['ranges']['cell_size'][1]}")
            print(f"Shot size: {pred['ranges']['shot_size'][0]} to {pred['ranges']['shot_size'][1]}")
            print(f"Time: {pred['ranges']['time'][0]:.2f}s to {pred['ranges']['time'][1]:.2f}s")
    
    # Create visualization plots
    create_prediction_plots(predictions)

if __name__ == "__main__":
    main()