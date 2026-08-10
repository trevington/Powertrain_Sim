import optuna
import os
import sys
import numpy as np
import glob

# Ensure we can import the simulation engine
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
from main import SimulationEngine

def objective(trial):
    # Define the parameter search space
    params = {
        # Scaling the empirical resistance (Generation)
        'dcir_safety_factor': trial.suggest_float('dcir_safety_factor', 0.5, 2.0),
        # Thermal mass (Inertia)
        'battery_cp': trial.suggest_float('battery_cp', 600.0, 1500.0),
        # Convection coefficient (Cooling)
        'h_fan': trial.suggest_float('h_fan', 0.0, 20.0),
    }

    sim = SimulationEngine(params)
    logs_dir = '/home/tadayal100/Powertrain_Sim/ai sim/logs/'
    all_logs = glob.glob(os.path.join(logs_dir, "**/*.CSV"), recursive=True) + \
               glob.glob(os.path.join(logs_dir, "**/*.csv"), recursive=True)

    errors = []
    for log_path in all_logs:
        try:
            # Run validation for each log using the suggested params
            res = sim.validate_thermal_from_log(log_path, show_plot=False)
            # We want to minimize Mean Absolute Error (MAE)
            errors.append(res['mean_error'])
        except Exception:
            continue

    if not errors:
        return 1e6

    # Return the average mean error across all logs
    return np.mean(errors)

if __name__ == "__main__":
    storage_name = "sqlite:///optuna_thermal_study.db"
    study = optuna.create_study(direction='minimize', storage=storage_name, load_if_exists=True)
    
    print("Starting Thermal Model Parameter Optimization...")
    study.optimize(objective, n_trials=100, n_jobs=-1)

    print("\n" + "="*40)
    print("OPTIMIZED THERMAL PARAMETERS")
    print("="*40)
    print(f"Best Mean Error: {study.best_value:.4f} °C")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.6f}")

    # Verification: Run the batch study one last time with best params
    print("\nGenerating final comparison plot...")
    best_sim = SimulationEngine(study.best_params)
    best_sim.run_thermal_batch_study('/home/tadayal100/Powertrain_Sim/ai sim/logs/')
    print("Results saved to thermal_batch_comparison.png")