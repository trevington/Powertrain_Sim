import optuna
import sys
import os
import numpy as np
import tempfile
from func_timeout import func_timeout, FunctionTimedOut 
from concurrent.futures import ProcessPoolExecutor, TimeoutError

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(MAIN_DIR)

from main.main import run_simulation

def objective(trial):
    # Suggest parameters for the driver's steering PID and path following lookahead
    params = {
        'k_stanley': trial.suggest_float('k_stanley', 0.01, 5.0, log=True),
        'kp_accel': trial.suggest_float('kp_accel', 0.01, 2.5, log=True),
        'ki_accel': trial.suggest_float('ki_accel', 0.000001, 2.0002, log=True),
        'kd_accel': trial.suggest_float('kd_accel', 0.2, 5.9, log=True),
        'kp_brake': trial.suggest_float('kp_brake', 0.0002, 4.5, log=True),
        'ki_brake': trial.suggest_float('ki_brake', 0.00005, 5.01, log=True),
        'kd_brake': trial.suggest_float('kd_brake', 0.0005, 1.0, log=True),
        'look_ahead_v_scale': trial.suggest_float('look_ahead_v_scale', 0.0005, 4.25, log=True),
        'laps_to_run': 1, # Run fewer laps to speed up the optimization process
        'motor_config': 'RWD',
        'regen_enabled': False,
        'look_ahead_indices': 1,
        'max_power_kw': trial.suggest_float('max_power_kw', 20.0, 80.0, log=True),
        'max_regen_power_kw': trial.suggest_float('max_regen_power_kw', 0.5, 15.0, log=True),
        'gear_ratio': trial.suggest_float('gear_ratio', 1.0, 8.0, log=True),
        'max_v': trial.suggest_float('max_v', 20.0, 80.0, log=True),
        'initial_battery_temp': trial.suggest_float('initial_battery_temp', 15.0, 40.0, log=True),
    }
    
    # Run simulation in headless mode
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_simulation, params, verbose=False, plot=False)
        try:
            results = future.result(timeout=15.0)

            # Penalty for not completing laps (indicates car went off track and couldn't recover)
            if results['laps_completed'] < params['laps_to_run']:
                raise optuna.TrialPruned()

            # Goal: Minimize Lap Time AND Cross Track Error (CTE)
            lap_time = results['best_lap_time']
            mean_cte = results['mean_cte']
            max_cte = results['max_cte']
            energy_used = results['energy_used_wh']
            min_v_error = results['min_v_error']
            
            # Weighted Score:
            # We want low lap times, but the car must stay on track.
            # If max deviation > 2.25m (typical track edge), we apply an exponential penalty.
            stability_penalty = 0
            if max_cte > 2.25: # NOTE: minimum 2.25m track edge in michigan
                stability_penalty = (max_cte ** 2) * 22.5

            if min_v_error > 0:
                min_v_error = 0

            score = 4 * lap_time + (mean_cte * 5.0) + stability_penalty + np.abs(min_v_error**6 + min_v_error * 50.0)

            return score
            
        except (TimeoutError):
            for pid, process in list(executor._processes.items()):
                process.terminate()

            executor.shutdown(wait=False)

            raise optuna.TrialPruned('Simulation timed out.')

        except Exception as e:
            raise optuna.TrialPruned(f'Simulation failed with exception: {e}')

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
    study.optimize(objective, n_trials=(12*48), n_jobs=-1)
    
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