import optuna
import sys
import os
import numpy as np

from main.main import run_simulation

def objective(trial):
    # Suggest parameters for the driver's steering PID and path following lookahead
    params = {
        'k_stanley': trial.suggest_float('k_stanley', 8, 15.0, log=True),
        'kp_accel': trial.suggest_float('kp_accel', 0.01, 0.5, log=True),
        'ki_accel': trial.suggest_float('ki_accel', 0.000001, 0.0002, log=True),
        'kd_accel': trial.suggest_float('kd_accel', 0.8, 1.9, log=True),
        'kp_brake': trial.suggest_float('kp_brake', 0.002, 0.5, log=True),
        'ki_brake': trial.suggest_float('ki_brake', 0.00005, 0.01, log=True),
        'kd_brake': trial.suggest_float('kd_brake', 0.005, 1.0, log=True),
        'look_ahead_indices': 1,
        'look_ahead_v_scale': trial.suggest_float('look_ahead_v_scale', 0.0005, 0.25),
        'laps_to_run': 2, # Run fewer laps to speed up the optimization process
        'motor_config': 'RWD',
        'regen_enabled': True,
        'max_power_kw': 45.0,
    }
    
    # Run simulation in headless mode
    try:
        results = run_simulation(params, verbose=False, plot=False)
    except Exception as e:
        return 1e6 # Large penalty for simulations that crash
    
    # Penalty for not completing laps (indicates car went off track and couldn't recover)
    if results['laps_completed'] < params['laps_to_run']:
        return 100000.0 + (params['laps_to_run'] - results['laps_completed']) * 5000.0
    
    # The deviation is 12m, which is massive. We must penalize deviation heavily.
    # Goal: Minimize Lap Time AND Cross Track Error (CTE)
    lap_time = results['best_lap_time']
    mean_cte = results['mean_cte']
    max_cte = results['max_cte']
    energy_used = results['energy_used_wh']
    
    
    # Weighted Score:
    # We want low lap times, but the car must stay on track.
    # If max deviation > 2m (typical track edge), we apply an exponential penalty.
    stability_penalty = 0
    if max_cte > 2.25: # NOTE: minimum 2.25m track edge in michigan
        stability_penalty = (max_cte ** 2) * 22.5
        
    # Jerk penalty: Excessively high jerk indicates PID chatter or unstable longitudinal control.
    # This weight balances the penalty against lap time and stability.

    score = 2 * lap_time + (mean_cte * 5.0) + stability_penalty + (energy_used * 0.15)
    
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
    study.optimize(objective, n_trials=10, n_jobs=-1)
    
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
