import numpy as np
import pandas as pd  # <--- Added import
from vehicle_sim import VehicleSimulator
import matplotlib.pyplot as plt

# --- SIMULATION RUNTIME SCRIPT ---
# 1. Load waypoints representing the track layout (X, Y, Target_Velocity)
michigan_endurance_circuit = np.load("michigan_track.npy") 

# 2. Instantiate the master simulation block
sim = VehicleSimulator(track_waypoints=michigan_endurance_circuit, layout="RWD")

dt = 0.001  # Fixed time-step interval
time = 0.0
log_interval = 1  # Log data every 1 ms for analysis
step_counter = 0
data_log = []

print("Running simulation...")
# 3. Core Simulation Execution Loop
while time < 300.0:  # Simulate a 5-minute track run segment
    sim.step(dt)

    if step_counter % log_interval == 0:
        # Log metrics to analyze heat signatures and lap times
        data_log.append({
            'time': time,
        'position_x': sim.x,
        'position_y': sim.y,
        'velocity': sim.v_x,
        'motor_temp': sim.motors['Rear'].temp,
        'coolant_temp': sim.cooling.fluid_temp
    })
    time += dt
    step_counter += 1

print("Simulation complete. Exporting data...")

# 4. Convert data log to a DataFrame and save as CSV
df = pd.DataFrame(data_log)
df.to_csv("simulation_telemetry.csv", index=False)

print("Successfully saved data to 'simulation_telemetry.csv'")

# Quick analytical plotting right after export
plt.figure(figsize=(10, 5))
plt.plot(df['time'], df['motor_temp'], label='Motor Core Temp (°C)', color='red')
plt.plot(df['time'], df['coolant_temp'], label='Coolant Liquid Temp (°C)', color='blue')
plt.axhline(y=105.0, color='black', linestyle='--', label='VCU Derating Threshold (105°C)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Temperature (°C)')
plt.title('Powertrain Thermal Saturation Profile - Endurance Loop')
plt.grid(True)
plt.legend()
plt.show()