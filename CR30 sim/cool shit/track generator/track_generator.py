import pandas as pd
import numpy as np

def add_velocity_targets(x, y, max_lat_g=1.5, max_long_g=0.8, max_v_mps=35.0):
    """
    Calculates physics-limited target velocity and acceleration for each waypoint.
    """
    n = len(x)
    g = 9.81
    a_lat_max = max_lat_g * g
    a_long_max = max_long_g * g
    
    # 1. Calculate distances and curvature
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    # Calculate segment distances (ds)
    dists = np.sqrt(np.diff(x, append=x[0])**2 + np.diff(y, append=y[0])**2)
    
    # 2. Raw Cornering Limit: V = sqrt(a_lat / curvature)
    # Avoid division by zero on straightaways
    v_target = np.ones(n) * max_v_mps
    nonzero_curvature = curvature > 1e-5
    v_target[nonzero_curvature] = np.sqrt(a_lat_max / curvature[nonzero_curvature])
    v_target = np.minimum(v_target, max_v_mps)

    # 3. Longitudinal Constraints (Braking and Acceleration)
    # Backward pass: Ensure we can slow down in time for upcoming corners
    for i in range(n - 1, -1, -1):
        next_idx = (i + 1) % n
        # v_f^2 = v_i^2 + 2as -> v_i = sqrt(v_f^2 - 2as)
        v_limit = np.sqrt(v_target[next_idx]**2 + 2 * a_long_max * dists[i])
        v_target[i] = min(v_target[i], v_limit)
        
    # Forward pass: Ensure we aren't asking for impossible acceleration
    for i in range(n):
        prev_idx = (i - 1) % n
        v_limit = np.sqrt(v_target[prev_idx]**2 + 2 * a_long_max * dists[prev_idx])
        v_target[i] = min(v_target[i], v_limit)

    # 4. Calculate required longitudinal acceleration (A_target)
    # a = (v_f^2 - v_i^2) / (2 * ds)
    a_target = np.zeros(n)
    for i in range(n):
        next_idx = (i + 1) % n
        # Small epsilon to avoid jitter
        accel = (v_target[next_idx]**2 - v_target[i]**2) / (2 * dists[i] + 1e-6)
        a_target[i] = np.clip(accel, -a_long_max, a_long_max)
        
    return v_target, a_target

def process_track_coordinates(input_csv, output_csv):
    """
    Converts track coordinates from feet to meters, zeros them at the 
    upper-left corner, and ensures counter-clockwise winding.
    """
    try:
        # 1. Load the data
        # skipinitialspace=True handles the space in " Y coord (ft)"
        df = pd.read_csv(input_csv, skipinitialspace=True)
        
        # Extract raw values
        x_ft = df['X coord (ft)'].values
        y_ft = df['Y coord (ft)'].values
        
        # 2. Convert Feet to Meters (1 ft = 0.3048 m)
        x_m = x_ft * 0.3048
        y_m = y_ft * 0.3048
        
        # 3. Zero data at the Upper Left Corner
        # In Cartesian space, Upper Left is Min X and Max Y
        x_origin = np.min(x_m)
        y_origin = np.max(y_m)
        
        x_zeroed = x_m - x_origin
        y_zeroed = y_m - y_origin
        
        # 4. Order points Counter-Clockwise (CCW)
        # We calculate the signed area using the Shoelace formula.
        # In a standard coordinate system:
        # Positive Area -> Counter-Clockwise
        # Negative Area -> Clockwise
        area = 0.0
        n = len(x_zeroed)
        for i in range(n):
            j = (i + 1) % n
            area += x_zeroed[i] * y_zeroed[j]
            area -= x_zeroed[j] * y_zeroed[i]
            
        # If area is negative, the points are CW, so we reverse them
        if area < 0:
            print("Detected Clockwise order. Reversing to Counter-Clockwise...")
            x_final = x_zeroed[::-1]
            y_final = y_zeroed[::-1]
        else:
            print("Detected Counter-Clockwise order. Maintaining sequence...")
            x_final = x_zeroed
            y_final = y_zeroed
            
        # 5. Generate Physics Targets
        v_target, a_target = add_velocity_targets(x_final, y_final)
            
        # 6. Export results
        output_df = pd.DataFrame({
            'X coord (m)': x_final,
            'Y coord (m)': y_final,
            'V_target (m/s)': v_target,
            'A_target (m/s^2)': a_target
        })
        
        output_df.to_csv(output_csv, index=False)
        print(f"Successfully processed {n} waypoints.")
        print(f"Output saved to: {output_csv}")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Assuming endurance.csv is in the current directory
    process_track_coordinates('endurance.csv', 'endurance_meters.csv')
