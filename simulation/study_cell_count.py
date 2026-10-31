import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from concurrent.futures import ProcessPoolExecutor

# Ensure we can import the simulation engine
sys.path.append(os.path.join(os.path.dirname(__file__), '\\simulation\\maine'))
#from main import run_simulation

from main import run_simulation


def simulate_cell_count(count):
    """Worker function to run a single simulation instance."""
    params = {
        'mode': 'Endurance',
        'laps_to_run': 22,
        'num_cells': int(count),
        'motor_config': 'RWD',
        'regen_enabled': False,
        'aero_dist_rear': 0.55
    }
    # Run simulation in headless mode
    results = run_simulation(params, verbose=False, plot=False)
    return count, results

def run_cell_count_study():
    """
    Sweeps battery cell count to analyze the trade-off between 
    voltage/energy capacity and vehicle mass.
    """
    # Sweep from a smaller pack (e.g., 80 cells) to the FSAE voltage limit (~140 cells)
    aero_v_1 = 
    
    session_times = []
    energy_used_pct = []
    vehicle_masses = []
    final_temps = []

    print("Starting Cell Count Sensitivity Study...")
    print(f"{'Cells':<10} | {'Total Time (s)':<15} | {'Energy Used (%)':<15} | {'Temp (C)':<10} | {'Mass (kg)':<10}")
    print("-" * 75)

    # Execute simulations in parallel using all available cores
    with ProcessPoolExecutor() as executor:
        # executor.map preserves the order of the input iterable
        all_results = list(executor.map(simulate_cell_count, cell_counts))

    for count, results in all_results:
        cell_mass = 0.2667
        current_mass = (292.0 - (140 * cell_mass)) + (count * cell_mass)

        total_capacity_wh = count * 3.5 * 13.0  # Nom Voltage * Ah * Series
        energy_pct = (results['energy_used_wh'] / total_capacity_wh) * 100

        session_times.append(results['total_session_time'])
        energy_used_pct.append(energy_pct)
        vehicle_masses.append(current_mass)
        final_temps.append(results['final_temp'])
        
        print(f"{count:<10} | {results['total_session_time']:<15.3f} | {energy_pct:<15.2f} | {results['final_temp']:<10.2f} | {current_mass:<10.2f}")

    # Plotting the Results
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary Axis: Session Time
    color_time = 'tab:blue'
    ax1.set_xlabel('Number of Cells (Series)')
    ax1.set_ylabel('Total Session Time (s)', color=color_time, fontsize=12, fontweight='bold')
    line1 = ax1.plot(cell_counts, session_times, marker='o', color=color_time, linewidth=2, label='Total Session Time')
    ax1.tick_params(axis='y', labelcolor=color_time)
    ax1.grid(True, alpha=0.3)

    # Secondary Axis: Energy Consumption
    ax2 = ax1.twinx()
    color_energy = 'tab:red'
    ax2.set_ylabel('Total Energy Used (%)', color=color_energy, fontsize=12, fontweight='bold')
    line2 = ax2.plot(cell_counts, energy_used_pct, marker='s', color=color_energy, linewidth=2, linestyle='--', label='Energy Used %')
    ax2.tick_params(axis='y', labelcolor=color_energy)

    # Tertiary Axis: Final Temperature
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color_temp = 'tab:orange'
    ax3.set_ylabel('Final Battery Temp (°C)', color=color_temp, fontsize=12, fontweight='bold')
    line3 = ax3.plot(cell_counts, final_temps, marker='^', color=color_temp, linewidth=2, linestyle=':', label='Final Temp')
    ax3.tick_params(axis='y', labelcolor=color_temp)

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.title('Impact of Battery Cell Count on Performance\n(Mass vs. Voltage/Energy Trade-off)')
    plt.tight_layout()
    
    output_plot = 'cell_count_study.png'
    plt.savefig(output_plot)
    print(f"\nStudy complete. Results saved to {output_plot}")

if __name__ == "__main__":
    run_cell_count_study()