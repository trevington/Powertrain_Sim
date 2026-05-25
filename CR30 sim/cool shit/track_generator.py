import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev

def build_track_from_gps(csv_path, output_npy_path="michigan_track.npy"):
    # 1. Load the raw endurance GPS data
    df = pd.read_csv(csv_path)
    
    # Extract columns based on your exact CSV headers
    latitudes = df['GPS Latitude (deg)'].values
    longitudes = df['GPS Longitude (deg)'].values
    
    # 2. Data Cleaning: Filter out duplicate/stationary points
    # Scipy's splprep requires strictly increasing or changing coordinates
    unique_indices = [0]
    for i in range(1, len(latitudes)):
        # Check if the car moved more than a minute threshold (roughly a few centimeters)
        dist_lat = latitudes[i] - latitudes[unique_indices[-1]]
        dist_lon = longitudes[i] - longitudes[unique_indices[-1]]
        if np.hypot(dist_lat, dist_lon) > 1e-6: 
            unique_indices.append(i)
            
    lat_filtered = latitudes[unique_indices]
    lon_filtered = longitudes[unique_indices]
    
    # 3. Project Geodetic (Lat/Lon) to Local Cartesian Coordinates (X, Y in meters)
    # We choose the first active coordinate as our local coordinate origin (0,0)
    lat0, lon0 = lat_filtered[0], lon_filtered[0]
    
    R_earth = 6371000.0  # Radius of Earth in meters
    lat0_rad = np.radians(lat0)
    
    # Flat-Earth / Equirectangular projection (highly accurate for track-scale distances)
    raw_x = np.radians(lon_filtered - lon0) * R_earth * np.cos(lat0_rad)
    raw_y = np.radians(lat_filtered - lat0) * R_earth
    
    # Force the track to close cleanly if it's a closed lap
    # (Connects the final telemetry point back to the origin start point)
    raw_x = np.append(raw_x, raw_x[0])
    raw_y = np.append(raw_y, raw_y[0])
    
    # 4. Parametric Spline Interpolation with explicit smoothing
    # Real GPS data contains jitter. 's' is the smoothing factor. 
    # Increase 's' if the resulting path looks jagged; decrease it if it cuts corners.
    tck, u = splprep([raw_x, raw_y], s=15.0, per=True)
    
    # Generate a dense path of equidistant points (e.g., 2500 points for a ~1km track)
    num_points = 2500
    u_dense = np.linspace(0, 1, num_points)
    
    # Evaluate smooth track layout positions and analytical derivatives
    x_smooth, y_smooth = splev(u_dense, tck, der=0)
    dx, dy = splev(u_dense, tck, der=1)
    ddx, ddy = splev(u_dense, tck, der=2)
    
    # 5. Calculate Track Curvature and Local Radii
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(1.5)
    
    radius_of_curvature = np.zeros_like(curvature)
    mask = curvature > 1e-5
    radius_of_curvature[mask] = 1.0 / curvature[mask]
    radius_of_curvature[~mask] = 1000.0  # Safe fallback default for dead-straight sections
    
    # 6. Apply Performance Metrics for Target Velocity Profile
    mu = 1.50           # Estimated tire lateral grip coefficient 
    g = 9.81            # m/s^2
    v_top_speed = 35.7  # Capped max vehicle speed (~80 mph / 128.7 km/h)
    
    # Lateral-grip capped speed profile
    v_target = np.sqrt(mu * g * radius_of_curvature)
    v_target = np.clip(v_target, 4.0, v_top_speed) # Floor speed at 4 m/s so solver doesn't stall
    
    # 7. Backward Pass: Dynamic Braking Constraints
    # Ensure the car slows down *before* reaching sharp corners
    max_decel = 1.1 * g  # Mechanical longitudinal deceleration limit
    
    for i in range(num_points - 2, -1, -1):
        ds = np.hypot(x_smooth[i+1] - x_smooth[i], y_smooth[i+1] - y_smooth[i])
        # v_init^2 <= v_final^2 + 2 * a * d
        v_max_allowed = np.sqrt(v_target[i+1]**2 + 2 * max_decel * ds)
        if v_target[i] > v_max_allowed:
            v_target[i] = v_max_allowed
            
    # Clean wrap-around step for lap continuity 
    ds_boundary = np.hypot(x_smooth[0] - x_smooth[-1], y_smooth[0] - y_smooth[-1])
    v_max_boundary = np.sqrt(v_target[0]**2 + 2 * max_decel * ds_boundary)
    if v_target[-1] > v_max_boundary:
        v_target[-1] = v_max_boundary

    # 8. Stack and Save 
    waypoints = np.vstack((x_smooth, y_smooth, v_target)).T
    np.save(output_npy_path, waypoints)
    
    print(f"Ingested {len(latitudes)} raw entries.")
    print(f"Cleaned down to {len(lat_filtered)} distinct coordinates.")
    print(f"Saved optimized waypoint array with shape {waypoints.shape} to '{output_npy_path}'")

# Execute using your file name
build_track_from_gps("cr28_endurance_gps_position.csv")