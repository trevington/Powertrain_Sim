import pandas as pd
import numpy as np

def add_velocity_targets(input_csv, output_csv, max_lat_g=1.5, max_long_g=0.8, max_v_mps=28.41):
    """
    Calculates physics-limited target velocity and acceleration for processed waypoints.
    """
    try:
        df = pd.read_csv(input_csv)
        x = df['X coord (m)'].values
        y = df['Y coord (m)'].values
        
        n = len(x)
        g = 9.81
        a_lat_max = max_lat_g * g
        a_long_max = max_long_g * g
        
        # 1. Calculate distances and curvature
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        
        # Segment distances
        dists = np.sqrt(np.diff(x, append=x[0])**2 + np.diff(y, append=y[0])**2)
        
        # 2. Raw Cornering Limit
        v_target = np.ones(n) * max_v_mps
        nonzero_curvature = curvature > 1e-5
        v_target[nonzero_curvature] = np.sqrt(a_lat_max / curvature[nonzero_curvature])
        v_target = np.minimum(v_target, max_v_mps)

        # 3. Longitudinal Smoothing (Braking and Acceleration passes)
        # Backward pass (Braking)
        for i in range(n - 1, -1, -1):
            next_idx = (i + 1) % n
            v_limit = np.sqrt(v_target[next_idx]**2 + 2 * a_long_max * dists[i])
            v_target[i] = min(v_target[i], v_limit)
            
        # Forward pass (Acceleration)
        for i in range(n):
            prev_idx = (i - 1) % n
            v_limit = np.sqrt(v_target[prev_idx]**2 + 2 * a_long_max * dists[prev_idx])
            v_target[i] = min(v_target[i], v_limit)

        # 4. Calculate required longitudinal acceleration (A_target)
        a_target = np.zeros(n)
        for i in range(n):
            next_idx = (i + 1) % n
            accel = (v_target[next_idx]**2 - v_target[i]**2) / (2 * dists[i] + 1e-6)
            a_target[i] = np.clip(accel, -a_long_max, a_long_max)
            
        # Update and save
        df['V_target (m/s)'] = v_target
        df['A_target (m/s^2)'] = a_target
        df.to_csv(output_csv, index=False)
        
        print(f"Physics profiling complete. Final data saved to: {output_csv}")
        return True

    except Exception as e:
        print(f"Error in physics profiling: {e}")
        return False