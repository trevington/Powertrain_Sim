import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Ensure we can import the simulation engine
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
from main import run_simulation

def run_regen_study():
    # Parameters to sweep
    front_torque = np.linspace(2.35, 7, 10)  # Sweep from 12kW to 64kW
    fixed_bias = 0.3  # Fixed braking bias for this power sweep

    regen_data = []
    time_data = []

    print(f"Starting Front Hub Motor Power Study (Fixed Bias: {fixed_bias})...")
    print("-" * 70)
    
    for tq in front_torque:
        params = {
            'mode': 'Endurance',
            'laps_to_run': 1,
            #'motor_config': 'AWD' if pwr > 0 else 'RWD',
            #'front_motor_max_power_kw': 10,
            'front_motor_max_torque': tq,
            #'front_gear_ratio': front_ratio,
            'front_bias': fixed_bias,

        }
        
        # Run simulation in headless mode
        results = run_simulation(params, verbose=False, plot=False)
        
        regen_data.append(results['total_regen_wh'])
        time_data.append(results['best_lap_time'])
        print(f"Front Torque: {tq:4.1f} kW | Regen: {results['total_regen_wh']:6.2f} Wh | Lap: {results['best_lap_time']:.3f}s")

    # Create the plot with two Y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Axis: Energy Recovered
    color_regen = 'tab:green'
    ax1.set_xlabel('Front Hub Motor Max Power (kW)')
    ax1.set_ylabel('Energy Recovered (Wh)', color=color_regen, fontsize=12, fontweight='bold')
    line1 = ax1.plot(front_power_levels, regen_data, marker='o', color=color_regen, linewidth=2, label='Energy Recovered')
    ax1.tick_params(axis='y', labelcolor=color_regen)
    ax1.grid(True, alpha=0.3)

    # Secondary Axis: Lap Time
    ax2 = ax1.twinx()
    color_time = 'tab:blue'
    ax2.set_ylabel('Best Lap Time (s)', color=color_time, fontsize=12, fontweight='bold')
    line2 = ax2.plot(front_power_levels, time_data, marker='s', color=color_time, linewidth=2, linestyle='--', label='Lap Time')
    ax2.tick_params(axis='y', labelcolor=color_time)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title(f'Performance Impact of Front Hub Power\n(Fixed Front Braking Bias: {fixed_bias})')
    plt.tight_layout()
    plt.savefig('power_impact_study.png')
    plt.show()

if __name__ == "__main__":
    run_regen_study()