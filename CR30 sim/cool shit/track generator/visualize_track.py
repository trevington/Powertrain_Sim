import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_track(csv_path):
    """
    Visualizes the track waypoints and indicates the driving direction.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run track_generator.py first.")
        return

    # Load the processed data
    df = pd.read_csv(csv_path)
    x = df['X coord (m)'].values
    y = df['Y coord (m)'].values
    v = df['V_target (m/s)'].values

    plt.figure(figsize=(12, 10))
    
    # 1. Plot the trajectory line to show the overall shape
    plt.plot(x, y, color='black', alpha=0.3, label='Track Boundary')
    
    # 2. Use a scatter plot with a colormap to show target velocity
    # The color represents the speed limit calculated by the physics profiler
    sc = plt.scatter(x, y, c=v, cmap='plasma', s=25, label='Velocity Profile')
    plt.colorbar(sc, label='Target Velocity (m/s)')

    # 3. Add directional arrows using Quiver
    # Calculate vectors between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Subsample indices for clarity (draw an arrow every 5 points)
    skip = 5
    plt.quiver(x[:-1:skip], y[:-1:skip], dx[::skip], dy[::skip], 
               color='red', scale_units='xy', angles='xy', scale=1, 
               width=0.003, headwidth=5, label='Direction of Travel')

    # 4. Highlight the start point in green
    plt.plot(x[0], y[0], 'go', markersize=10, label='Start Point')

    plt.axis('equal')  # Critical to maintain correct geometric proportions
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('Track Velocity Heatmap\n(Arrows indicate CCW direction)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    csv_file = '/home/tadayal100/Powertrain_Sim/CR30 sim/cool shit/endurance_meters.csv'
    visualize_track(csv_file)