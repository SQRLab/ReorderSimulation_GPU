import json
import numpy as np
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

def extract_unique_values(data):
    """Extract unique values from the nested JSON structure."""
    nr_sizes = sorted(set(int(key) for key in data.keys()))
    nz_sizes = sorted(set(int(nz) for nr in data.values() 
                         for nz in nr.keys()))
    ion_sizes = sorted(set(int(ion) for nr in data.values() 
                          for nz in nr.values() 
                          for ion in nz.keys()))
    shot_sizes = sorted(set(int(shot) for nr in data.values() 
                          for nz in nr.values() 
                          for ions in nz.values()
                          for shot in ions.keys()))
    return nr_sizes, nz_sizes, ion_sizes, shot_sizes

class RefinedTimePredictor:
    def __init__(self):
        self.rf_dc_params = None
        self.sim_params = None
        self.known_grid_points = None
        self.known_shot_sizes = None
        
    def rf_dc_function(self, grid_points, a, b, c):
        """RF/DC time model: a * (grid_points)^b + c"""
        return a * np.power(grid_points, b) + c
    
    def sim_function(self, shots, ions, a, b, c):
        """Simulation time model: a * shots * ions + b * shots + c"""
        return a * shots * ions + b * shots + c
    
    def fit(self, data):
        """Fit models to the data."""
        # Prepare training data
        grid_points_list = []
        rf_dc_times = []
        shots_list = []
        ions_list = []
        sim_times = []
        
        for nr in data:
            for nz in data[nr]:
                grid_points = int(nr) * int(nz)
                for ion in data[nr][nz]:
                    for shot in data[nr][nz][ion]:
                        # Take the first RF/DC time for each grid size
                        if grid_points not in grid_points_list:
                            grid_points_list.append(grid_points)
                            rf_dc_times.append(data[nr][nz][ion][shot]['rf_dc_time'])
                        
                        shots_list.append(int(shot))
                        ions_list.append(int(ion))
                        sim_times.append(data[nr][nz][ion][shot]['simulation_time'])
        
        # Fit RF/DC model
        self.rf_dc_params, _ = curve_fit(
            self.rf_dc_function,
            np.array(grid_points_list),
            np.array(rf_dc_times),
            p0=[1e-6, 1.0, 1.0],
            bounds=([0, 0.5, 0], [1e-4, 2.0, 100])
        )
        
        # Fit simulation model
        def sim_fit_func(X, a, b, c):
            shots, ions = X
            return self.sim_function(shots, ions, a, b, c)
        
        self.sim_params, _ = curve_fit(
            sim_fit_func,
            (np.array(shots_list), np.array(ions_list)),
            np.array(sim_times),
            p0=[1e-4, 1e-3, 1.0],
            bounds=([0, 0, 0], [1e-3, 1.0, 100])
        )
        
        # Store known ranges
        self.known_grid_points = sorted(grid_points_list)
        self.known_shot_sizes = sorted(set(shots_list))
    
    def predict(self, nr, nz, ion_count, shot_size):
        """Predict times for given parameters."""
        grid_points = nr * nz
        
        # RF/DC time prediction
        rf_dc_time = float(self.rf_dc_function(grid_points, *self.rf_dc_params))
        
        # Simulation time prediction
        sim_time = float(self.sim_function(shot_size, ion_count, *self.sim_params))
        
        # Calculate uncertainties
        rf_dc_uncertainty = self._calculate_uncertainty(grid_points, self.known_grid_points)
        sim_uncertainty = self._calculate_uncertainty(shot_size, self.known_shot_sizes)
        
        rf_dc_error = rf_dc_time * rf_dc_uncertainty
        sim_error = sim_time * sim_uncertainty
        
        return {
            'rf_dc': (rf_dc_time, rf_dc_error),
            'simulation': (sim_time, sim_error),
            'total': (rf_dc_time + sim_time, rf_dc_error + sim_error)
        }
    
    def _calculate_uncertainty(self, value, known_values):
        """Calculate relative uncertainty based on extrapolation distance."""
        min_val, max_val = min(known_values), max(known_values)
        if min_val <= value <= max_val:
            return 0.1  # 10% base uncertainty
        
        # Increase uncertainty based on extrapolation distance
        relative_distance = min(abs(value - min_val), abs(value - max_val)) / max_val
        return 0.1 + 0.2 * relative_distance

def main():
    try:
        with open('computation_times_detailed.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: computation_times_detailed.json not found!")
        return
    
    nr_sizes, nz_sizes, ion_sizes, shot_sizes = extract_unique_values(data)
    
    print("\nAvailable parameters:")
    print(f"Nr sizes: {nr_sizes}")
    print(f"Nz sizes: {nz_sizes}")
    print(f"Ion sizes: {ion_sizes}")
    print(f"Shot sizes: {shot_sizes}")
    
    # Train model
    model = RefinedTimePredictor()
    model.fit(data)
    
    print("\nModel trained. Enter combinations (ion_size,nr_size,nz_size,shot_size), blank line to finish:")
    while True:
        line = input().strip()
        if not line:
            break
        
        try:
            ion_size, nr_size, nz_size, shot_size = map(int, line.split(','))
            predictions = model.predict(nr_size, nz_size, ion_size, shot_size)
            
            print(f"\nPredictions for: Ion={ion_size}, Nr={nr_size}, Nz={nz_size}, Shots={shot_size}")
            print("-" * 60)
            for key, (value, uncertainty) in predictions.items():
                print(f"{key.replace('_', ' ').title()} time: {value:.2f} ± {uncertainty:.2f} seconds")
            
            grid_points = nr_size * nz_size
            max_known_grid = max(model.known_grid_points)
            max_known_shots = max(model.known_shot_sizes)
            
            if grid_points > max_known_grid:
                print(f"\nWarning: Grid size ({grid_points}) exceeds maximum training size ({max_known_grid})")
            if shot_size > max_known_shots:
                print(f"Warning: Shot count ({shot_size}) exceeds maximum training size ({max_known_shots})")
            
        except ValueError:
            print("Invalid format. Please use: ion_size,nr_size,nz_size,shot_size")
        except Exception as e:
            print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()