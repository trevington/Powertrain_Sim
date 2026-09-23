import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from concurrent.futures import ProcessPoolExecutor

# Ensure we can import the simulation engine
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
from main import run_simulation

def simulate_cell_count(count):
    """Worker function to run a single simulation instance."""
    params = {
        'mode': 'Endurance',
        'laps_to_run': 22,
        'motor_config': 'AWD',
        'regen_enabled': True,
        'max_regen_power_kw': 2.0,
        'max_power_kw': count,
    }
    # Run simulation in headless mode
    results = run_simulation(params, verbose=False, plot=False)
    return count, results

def run_study():
    lis = np.arange(10, 90, 5)
    
    session_times = []
    final_soe = []
    final_temps = []

    print("Starting Study...")
    print(f"{'values':<10} | {'Total Time (s)':<15} | {'Final SoE (%)':<15} | {'Temp (C)':<10}")
    print("-" * 75)

    # Execute simulations in parallel using all available cores
    with ProcessPoolExecutor() as executor:
        # executor.map preserves the order of the input iterable
        all_results = list(executor.map(simulate_cell_count, lis))

    for count, results in all_results:
        session_times.append(results['total_session_time'] / results['laps_completed'])  # Average time per lap
        final_soe.append(results['final_soe'])
        final_temps.append(results['final_temp'])
        
        print(f"{count:<10} | {results['total_session_time']:<15.3f} | {results['final_soe']:<15.2f} | {results['final_temp']:<10.2f}")

    # Plotting the Results
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary Axis: Session Time
    color_time = 'tab:blue'
    ax1.set_xlabel('Max Power (kW)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Lap Time (s)', color=color_time, fontsize=12, fontweight='bold')
    line1 = ax1.plot(lis, session_times, marker='o', color=color_time, linewidth=2, label='Average Lap Time')
    ax1.tick_params(axis='y', labelcolor=color_time)
    ax1.grid(True, alpha=0.3)

    # Secondary Axis: Energy Consumption
    ax2 = ax1.twinx()
    color_energy = 'tab:red'
    ax2.set_ylabel('Final SoE (%)', color=color_energy, fontsize=12, fontweight='bold')
    line2 = ax2.plot(lis, final_soe, marker='s', color=color_energy, linewidth=2, linestyle='--', label='Final SoE')
    ax2.tick_params(axis='y', labelcolor=color_energy)

    # Tertiary Axis: Final Temperature
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color_temp = 'tab:orange'
    ax3.set_ylabel('Final Battery Temp (°C)', color=color_temp, fontsize=12, fontweight='bold')
    line3 = ax3.plot(lis, final_temps, marker='^', color=color_temp, linewidth=2, linestyle=':', label='Final Temp')
    ax3.tick_params(axis='y', labelcolor=color_temp)

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.title('Impact of max power on Endurance Performance')
    plt.tight_layout()
    
    output_plot = 'study_results.png'
    plt.savefig(output_plot)
    print(f"\nStudy complete. Results saved to {output_plot}")

if __name__ == "__main__":
    run_study()