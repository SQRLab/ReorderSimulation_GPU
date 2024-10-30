import pandas as pd
import re

# Define the paths to your data files
ion_data_file = 'simulation_data.txt'
collision_data_file = 'collisional_particle_debug.txt'

# Initialize lists to store parsed ion data
ion_timesteps = []
ion_indices = []
r_positions = []
z_positions = []
vr_velocities = []
vz_velocities = []
erfi_before = []
ezfi_before = []
erfi_after = []
ezfi_after = []

# Initialize lists to store parsed collision data
collision_timesteps = []
collision_rs = []
collision_zs = []
collision_vr = []
collision_vz = []

# Regular expression to identify a timestep line in ion data
ion_timestep_pattern = re.compile(r'^Timestep\s+(\d+)')

# Regular expression to identify an ion data line
ion_pattern = re.compile(
    r'^Ion\s+(\d+)\s+Position\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
    r'\s+Velocity\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
    r'\s+Erfi_before\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
    r'\s+Ezfi_before\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
    r'\s+Erfi_after\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
    r'\s+Ezfi_after\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
)

# Regular expression to parse the collision data line
collision_pattern = re.compile(
    r'Timestep\s+(\d+)\s+Position:\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
    r'\s+Velocity:\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
)

# Parse ion data file
with open(ion_data_file, 'r') as file:
    current_timestep = None
    for line in file:
        line = line.strip()
        
        # Check for timestep
        timestep_match = ion_timestep_pattern.match(line)
        if timestep_match:
            current_timestep = int(timestep_match.group(1))
            continue
        
        # Check for ion data
        ion_match = ion_pattern.match(line)
        if ion_match and current_timestep is not None:
            ion_index = int(ion_match.group(1))
            r = float(ion_match.group(2))
            z = float(ion_match.group(3))
            vr = float(ion_match.group(4))
            vz = float(ion_match.group(5))
            erfi_b = float(ion_match.group(6))
            ezfi_b = float(ion_match.group(7))
            erfi_a = float(ion_match.group(8))
            ezfi_a = float(ion_match.group(9))
            
            ion_timesteps.append(current_timestep)
            ion_indices.append(ion_index)
            r_positions.append(r)
            z_positions.append(z)
            vr_velocities.append(vr)
            vz_velocities.append(vz)
            erfi_before.append(erfi_b)
            ezfi_before.append(ezfi_b)
            erfi_after.append(erfi_a)
            ezfi_after.append(ezfi_a)

# Parse collision data file
with open(collision_data_file, 'r') as file:
    for line in file:
        match = collision_pattern.match(line.strip())
        if match:
            timestep = int(match.group(1))
            r = float(match.group(2))
            z = float(match.group(3))
            vr = float(match.group(4))
            vz = float(match.group(5))
            
            collision_timesteps.append(timestep)
            collision_rs.append(r)
            collision_zs.append(z)
            collision_vr.append(vr)
            collision_vz.append(vz)

# Create DataFrame for ion data
ion_df = pd.DataFrame({
    'timestep': ion_timesteps,
    'ion': ion_indices,
    'r': r_positions,
    'z': z_positions,
    'vr': vr_velocities,
    'vz': vz_velocities,
    'Erfi_before': erfi_before,
    'Ezfi_before': ezfi_before,
    'Erfi_after': erfi_after,
    'Ezfi_after': ezfi_after
})

# Create DataFrame for collision data
collision_df = pd.DataFrame({
    'timestep': collision_timesteps,
    'r_collision': collision_rs,
    'z_collision': collision_zs,
    'vr_collision': collision_vr,
    'vz_collision': collision_vz
})

# Merge ion data with collision data on timestep
merged_df = pd.merge(ion_df, collision_df, on='timestep', how='outer')

# Save the parsed data to CSV files
ion_df.to_csv('parsed_ion_data_cpu.csv', index=False)
collision_df.to_csv('parsed_collision_data_cpu.csv', index=False)
merged_df.to_csv('merged_simulation_data_cpu.csv', index=False)

# Print summary of parsed data
print("Ion data summary:")
print(ion_df.head())
print(f"Total ion data rows: {len(ion_df)}")

print("\nCollision data summary:")
print(collision_df.head())
print(f"Total collision data rows: {len(collision_df)}")

print("\nMerged data summary:")
print(merged_df.head())
print(f"Total merged data rows: {len(merged_df)}")