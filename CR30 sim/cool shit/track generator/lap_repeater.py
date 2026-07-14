import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def process_and_plot_laps(csv_path, timestep=0.05, laps=22):
    """
    Reads track data, interpolates it to a fixed time interval,
    repeats it for 22 laps, and visualizes the results.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load the processed track geometry and velocity targets
    df = pd.read_csv(csv_path)
    x = df['X coord (m)'].values
    y = df['Y coord (m)'].values
    v = df['V_target (m/s)'].values

    # 1. Calculate the time elapsed between each waypoint based on target velocity
    # Calculate Euclidean distance between consecutive points (wrapping to start)
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    ds = np.sqrt(dx**2 + dy**2)

    # time = distance / velocity (adding a floor to velocity to avoid division by zero)
    dt_segments = ds / np.maximum(v, 0.5)
    t_waypoints = np.cumsum(np.insert(dt_segments[:-1], 0, 0))
    total_lap_time = t_waypoints[-1] + dt_segments[-1]

    # 2. Interpolate coordinates and velocity at the fixed 0.05s timestep
    t_single_lap = np.arange(0, total_lap_time, timestep)
    
    # Use periodic interpolation to ensure a smooth transition between laps
    x_interp = np.interp(t_single_lap, t_waypoints, x, period=total_lap_time)
    y_interp = np.interp(t_single_lap, t_waypoints, y, period=total_lap_time)
    v_interp = np.interp(t_single_lap, t_waypoints, v, period=total_lap_time)

    # 3. Repeat data for the full endurance distance (22 laps)
    full_x = np.tile(x_interp, laps)
    full_y = np.tile(y_interp, laps)
    full_v = np.tile(v_interp, laps)
    full_t = np.arange(len(full_x)) * timestep

    # 4. Save to CSV
    output_df = pd.DataFrame({
        'Time (s)': full_t,
        'X coord (m)': full_x,
        'Y coord (m)': full_y,
        'V_target (m/s)': full_v,
        'A_target (m/s^2)': np.gradient(full_v, timestep)
    })
    output_csv = csv_path.replace('.csv', '_timed.csv')
    output_df.to_csv(output_csv, index=False)

    # 5. Visualization
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Spatial Trajectory heatmap
    plt.subplot(2, 1, 1)
    sc = plt.scatter(full_x, full_y, c=full_v, cmap='plasma', s=1, alpha=0.3)
    plt.colorbar(sc, label='Target Velocity (m/s)')
    plt.title(f'Track Trajectory: {laps} Laps at {timestep}s Intervals')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Velocity Profile over the entire race
    plt.subplot(2, 1, 2)
    plt.plot(full_t, full_v, color='tab:red', linewidth=0.5)
    plt.title(f'Velocity vs. Time (Total Race Duration)')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Simulation Summary:")
    print(f"Saved processed data to: {output_csv}")
    print(f"Estimated Lap Time: {total_lap_time:.2f}s")
    print(f"Total Race Time: {full_t[-1]:.2f} seconds")

if __name__ == "__main__":
    csv_file = 'tracks/endurance_track.csv'
    process_and_plot_laps(csv_file)
