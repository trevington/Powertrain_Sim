import optuna
import sys
import os
import numpy as np

# Ensure we can import the simulation engine from the 'simulation' subdirectory
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
from main import run_simulation

def objective(trial):
    # Suggest parameters for the driver's steering PID and path following lookahead
    params = {
        'steer_kp': trial.suggest_float('steer_kp', 0.1, 15.0, log=True),
        'steer_kd': trial.suggest_float('steer_kd', 0.01, 5.0, log=True),
        'lookahead_dist': trial.suggest_float('lookahead_dist', 1.0, 10.0),
        'kp_v': trial.suggest_float('kp_v', 0.1, 10.0, log=True),
        'ki_v': trial.suggest_float('ki_v', 0.01, 2.0, log=True),
        'kd_v': trial.suggest_float('kd_v', 0.01, 2.0, log=True),
        'laps_to_run': 2 # Run fewer laps to speed up the optimization process
    }
    
    # Run simulation in headless mode
    try:
        results = run_simulation(params, verbose=False, plot=False)
    except Exception as e:
        return 1e6 # Large penalty for simulations that crash
    
    # Penalty for not completing laps (indicates car went off track and couldn't recover)
    if results['laps_completed'] < params['laps_to_run']:
        return 10000.0 + (params['laps_to_run'] - results['laps_completed']) * 5000.0
    
    # The deviation is 12m, which is massive. We must penalize deviation heavily.
    # Goal: Minimize Lap Time AND Cross Track Error (CTE)
    lap_time = results['best_lap_time']
    mean_cte = results['mean_cte']
    max_cte = results['max_cte']
    
    # Weighted Score:
    # We want low lap times, but the car must stay on track.
    # If max deviation > 2m (typical track edge), we apply an exponential penalty.
    stability_penalty = 0
    if max_cte > 2.0:
        stability_penalty = (max_cte ** 2) * 50.0
        
    score = lap_time + (mean_cte * 5.0) + stability_penalty
    
    return score

if __name__ == "__main__":
    # Create an Optuna study to minimize the score
    # Using a SQLite storage backend enables multi-process parallelization
    storage_name = "sqlite:///optuna_pid_study.db"
    study = optuna.create_study(
        study_name="pid_optimization",
        storage=storage_name,
        direction='minimize',
        load_if_exists=True)

    # n_jobs=-1 uses all available CPU cores
    study.optimize(objective, n_trials=100, n_jobs=-1)
    
    print("\n" + "="*40)
    print("OPTIMIZATION RESULTS")
    print("="*40)
    print(f"Best Weighted Score: {study.best_value:.4f}")
    print("Best PID/Lookahead Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.6f}")
    
    # Run final simulation with best params and show plots
    print("\nRunning final verification simulation with best parameters...")
    run_simulation(study.best_params, verbose=True, plot=True)
