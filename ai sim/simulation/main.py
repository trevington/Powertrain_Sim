import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv
import os
import glob
from scipy.interpolate import RegularGridInterpolator
from driver import PathCommander
from powertrain import PowertrainController
from motor import PowertrainSystem
from vd import VehicleDynamics
from tb_thermal import TractiveBatteryThermal
from scoring import FSAEScorer

'''
NOTES:
- Regen not fully working (issues with max regen power)
- Torque vectoring not implemented yet
'''

class SimulationEngine:
    def __init__(self, params=None):
        self.track_file = 'tracks/endurance.csv'
        self.p = {
            'num_cells': 130, 'max_power_kw': 80.0, 'gear_ratio': 4.6, 
            'max_lat_accel_g': 2.0753, 'max_brake_accel_g': 1.49, 'max_grip_accel_g': 0.9656, # NOTE: taken from  CR30 tractive battery spreadsheet
            'max_v': 28.41, 'ambient_temp': 25.0, 'initial_battery_temp': 38.0, # NOTE: taken from CR30 tractive battery spreadsheet
            'laps_to_run': 22, 'motor_config': 'RWD',
            'steer_kp': 0.276638, 'steer_kd': 2.353120, 'lookahead_dist': 3.317667, 'lookahead_gain': 0.3, # NOTE: optimized values (optimize_driver.py)
            'wheelbase': 1.535, 'mode': 'Endurance', 'h_cg': 0.3302, # 13in NOTE: from erick accel sim
            'weight_dist_rear': 0.53, 'aero_dist_rear': 0.55, # NOTE: from erick accel sim
            'downforce_coeff': 2.45, # N/(m/s)^2, NOTE: from erick accel sim
            'rear_motor_max_torque': 150.0, 'rear_motor_max_rpm': 7000.0, # Emrax 208
            'front_motor_max_power_kw': 10.0, 'front_motor_max_torque': 4.7, 'front_motor_max_rpm': 20000.0, 'front_gear_ratio': 13.4, # MotorCAD, 480V 25A sim (NOTE: needs to be updated aaa)
            'rear_efficiency_csv': 'emrax208_efficiency.csv', # NOTE: from Emrax 208 datasheet
            'regen_enabled': False,
            'max_regen_power_kw': 2.0,
            'dcir_safety_factor': 1.5 * 1.096226, # NOTE: optmized 1.096226 (optimize_thermal.py)
            'battery_cp': 1377.265173, # NOTE: optimized 1377.265173 (optimize_thermal.py) 
            'h_fan': 0.717338 # NOTE: optimized 0.71738 (optimize_thermal.py)
        }
        if params: self.p.update(params)
        self.mode = self.p.get('mode', 'Endurance')

        self.dt, self.total_time, self.g = 0.008, 1800.0, 9.81
        cell_mass = 0.2667
        self.mass = (292.0 - (140 * cell_mass)) + (self.p['num_cells'] * cell_mass)
        # Aero
        self.aero_v_ms = np.array([25, 35, 45, 55, 65, 75]) * 0.44704
        self.aero_drag_n = np.array([30, 59, 99, 148, 207, 275]) * 4.44822
        # Rolling Resistance
        self.rr_v_ms = np.array([5.56, 11.11, 16.67, 22.22, 27.78, 33.33])
        self.rr_coeff = np.array([0.015, 0.0165, 0.018, 0.021, 0.025, 0.028])

        self.waypoints = self._prepare_waypoints()
        self._init_components()
        self._reset_state()

    def _prepare_waypoints(self):
        if self.p.get('mode') == 'Acceleration':
            # Return a simple straight line waypoints: (x, y, target_v, curvature)
            return [[0.0, 0.0, self.p['max_v'], 0.0], [150.0, 0.0, self.p['max_v'], 0.0]]

        coords = []
        try:
            with open(self.track_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    coords.append((float(row['X coord (m)']), float(row['Y coord (m)'])))
        except FileNotFoundError:
            return []

        waypoints = []
        max_lat_accel, max_brake_accel, max_grip_accel = self.p['max_lat_accel_g']*self.g, self.p['max_brake_accel_g']*self.g, self.p['max_grip_accel_g']*self.g
        for i in range(len(coords)):
            p_prev, p_curr, p_next = np.array(coords[i-1]), np.array(coords[i]), np.array(coords[(i+1)%len(coords)])
            a, b, c = np.linalg.norm(p_prev-p_curr), np.linalg.norm(p_curr-p_next), np.linalg.norm(p_next-p_prev)
            tri_area = 0.5 * abs(p_prev[0]*(p_curr[1]-p_next[1]) + p_curr[0]*(p_next[1]-p_prev[1]) + p_next[0]*(p_prev[1]-p_curr[1]))
            
            # Menger Curvature calculation
            if tri_area > 0.005:
                radius = (a * b * c) / (4 * tri_area)
                curvature = 1.0 / radius
                target_v = min(np.sqrt(max_lat_accel * radius), self.p['max_v'])
            else: 
                target_v, curvature = self.p['max_v'], 0.0
            waypoints.append([p_curr[0], p_curr[1], target_v, curvature])

        self.track_min_x = min(w[0] for w in waypoints)
        self.track_max_x = max(w[0] for w in waypoints)
        self.track_min_y = min(w[1] for w in waypoints)
        self.track_max_y = max(w[1] for w in waypoints)

        # Backward Pass: Ensure the car can slow down for upcoming corners
        for _ in range(2):
            for i in range(len(waypoints) - 1, -1, -1):
                next_idx = (i + 1) % len(waypoints)
                wp_next = waypoints[next_idx]
                wp_curr = waypoints[i]
                dist = np.hypot(wp_next[0] - wp_curr[0], wp_next[1] - wp_curr[1])
                
                # Using absolute max_brake_accel as a positive value for the limit
                v_allowed = np.sqrt(waypoints[next_idx][2]**2 + 2 * max_brake_accel * dist)
                waypoints[i][2] = min(waypoints[i][2], v_allowed)

        # Forward Pass: Account for acceleration limits and aero/rolling resistance
        for _ in range(2):
            for i in range(len(waypoints)):
                next_idx = (i + 1) % len(waypoints)
                v_curr = max(waypoints[i][2], 1.0) # Avoid division by zero
                
                p_prop = (self.p['max_power_kw'] * 1000.0) / (self.mass * v_curr) # F = P/v
                f_drag = np.interp(v_curr, self.aero_v_ms, self.aero_drag_n, left=0.0)
                f_rr = 0.015 * self.mass * self.g

                dynamic_max_accel = max(0.0, min(max_grip_accel, p_prop) - (f_drag / self.mass) - (f_rr / self.mass))
                
                dist = np.hypot(waypoints[next_idx][0] - waypoints[i][0], waypoints[next_idx][1] - waypoints[i][1])
                v_allowed = np.sqrt(waypoints[i][2]**2 + 2 * dynamic_max_accel * dist)
                waypoints[next_idx][2] = min(waypoints[next_idx][2], v_allowed)

        v_smooth = np.convolve(np.pad([wp[2] for wp in waypoints], 2, mode='wrap'), np.ones(5)/5, mode='valid')
        for i in range(len(waypoints)): waypoints[i][2] = v_smooth[i]
        return waypoints

    def _init_components(self):
        self.driver = PathCommander(track_waypoints=self.waypoints, mode=self.mode, params=self.p)
        self.controller = PowertrainController(motor_config=self.p['motor_config'], max_power_kw=self.p['max_power_kw'], params=self.p)
        self.powertrain = PowertrainSystem(config=self.p['motor_config'], params=self.p)
        self.vd = VehicleDynamics(mass=self.mass, params=self.p)
        self.thermal = TractiveBatteryThermal(ambient_temp=self.p['ambient_temp'], params=self.p)

    def _reset_state(self):
        if not self.waypoints: return
        initial_yaw = np.arctan2(self.waypoints[1][1] - self.waypoints[0][1], self.waypoints[1][0] - self.waypoints[0][0])
        self.state = {'x': self.waypoints[0][0], 'y': self.waypoints[0][1], 'yaw': initial_yaw, 'v': 0.0, 'yaw_rate': 0.0, 'a': 0.0, 'f_downforce': 0.0}
        self.vd.x, self.vd.y, self.wheel_speeds, self.battery_temp = self.state['x'], self.state['y'], {'rear': 0.0, 'front_left': 0.0, 'front_right': 0.0}, self.p['initial_battery_temp']
        self.history = {k: [] for k in [
            'time', 'v', 'v_target', 'cte', 'soc', 'temp', 'p_draw', 'throttle', 'brake_pct', 
            'x', 'y', 'voltage', 'current', 'motor_rpm', 'soe',
            't_front', 't_rear', 't_fl', 't_fr', 'redline', 'f_grip_f', 'f_grip_r', 'f_regen_f', 'f_regen_r'
        ]}
        self._redline_warned = False
        self.grip_limited_time = self.power_limited_time = self.braking_time = self.other_time = self.total_heat_j = self.total_aero_j = self.total_rr_j = self.total_accel_j = self.total_regen_j = 0.0
        self.laps_completed, self.lap_times, self.lap_segment_indices, self.lap_start_time, self.sector_times, self.final_time = 0, [], [], 0.0, [0.0]*4, 0.0
        self.t = 0.0
        self.t_idx = 0
        self.running = True
        self.prev_wp_idx = 0
        self.wps_per_sector = max(1, len(self.waypoints) // 4)
        self.start_x = self.state['x']
        self.last_demands = {}
        self.last_motor_outputs = {}

    def step(self, throttle_pct=None, steering_angle=None, verbose=True):
        """Advances the simulation by one time step."""
        if not self.running: return

        t, t_idx = self.t, self.t_idx

        # Check for 75m finish line in Acceleration mode
        if self.mode == 'Acceleration' and (self.state['x'] - self.start_x) >= 75.0:
            if verbose: print(f"75m Acceleration Finish! Time: {t:.3f}s")
            self.final_time = t
            self.lap_times.append(t)
            self.lap_segment_indices.append((0, t_idx))
            self.laps_completed = 1
            self.running = False
            return

        if throttle_pct is None or steering_angle is None:
            throttle_pct, steering_angle = self.driver.get_commands(self.state, self.dt)

        # Calculate instantaneous acceleration and aero downforce for the controller
        v_prev = self.history['v'][-1] if self.history['v'] else 0.0
        self.state['a'] = (self.state['v'] - v_prev) / self.dt if t > 0 else 0.0
        
        # Updated Aero Model: Quadratic Downforce
        v_curr = self.state['v']
        self.state['f_downforce'] = (self.p['downforce_coeff'] * v_curr**2)
        f_drag_lookup = np.interp(v_curr, self.aero_v_ms, self.aero_drag_n, left=0.0)

        demands = self.controller.calculate_demands(throttle_pct, steering_angle, self.state, self.wheel_speeds, self.dt)
        motor_outputs = self.powertrain.step(demands['motor_torques'], self.wheel_speeds, self.dt, self.battery_temp)
        
        self.last_demands = demands
        self.last_motor_outputs = motor_outputs

        heat_removed = self.thermal.calculate_heat_removed(self.battery_temp, v_curr)
        self.battery_temp += self.thermal.get_temperature_delta(motor_outputs.get('battery_heat_w', 0.0), heat_removed, self.dt)
        self.total_heat_j += motor_outputs['total_heat_w'] * self.dt

        is_redline = motor_outputs.get('motor_redline', False)
        if is_redline and verbose and not self._redline_warned:
            print(f"--- WARNING: MOTOR REDLINE REACHED at t={t:.3f}s ---")
            self._redline_warned = True

        v_prev = self.state['v']
        
        # Pass external drag to VD to ensure consistency with lookup table
        physics_in = {
            'rear_axle_torque': motor_outputs.get('rear_axle_torque', 0),
            'fl_wheel_torque': motor_outputs.get('fl_wheel_torque', 0),
            'fr_wheel_torque': motor_outputs.get('fr_wheel_torque', 0),
            'mechanical_brake_pct': demands['mechanical_brake_pct'],
            'external_drag': f_drag_lookup
        }
        new_state, self.wheel_speeds = self.vd.step(physics_in, steering_angle, self.dt)
        self.state.update(new_state)

        v_curr = self.state['v']
        self.total_aero_j += f_drag_lookup * v_curr * self.dt
        self.total_rr_j += np.interp(v_curr, self.rr_v_ms, self.rr_coeff) * self.mass * self.g * v_curr * self.dt
        if v_curr > v_prev: self.total_accel_j += self.mass * ((v_curr - v_prev) / self.dt) * v_curr * self.dt
        if motor_outputs['p_net_w'] < 0: self.total_regen_j += abs(motor_outputs['p_net_w']) * self.dt

        # Reference calculations based on CSV log parameters
        v_terminal = motor_outputs.get('voltage', 350.0)
        current_a = motor_outputs['p_net_w'] / v_terminal if v_terminal > 10 else 0.0
        # Simple motor RPM calc: Wheel rad/s -> RPM * gear ratio (Assuming 8-inch radius wheel)
        motor_rpm = (v_curr / (2 * np.pi * 0.2032)) * 60 * self.p['gear_ratio'] if v_curr > 0.1 else 0.0
        
        curr_wp = self.waypoints[self.driver.current_waypoint_idx]

        values = [
            t, v_curr, curr_wp[2], np.sqrt((self.state['x'] - curr_wp[0])**2 + (self.state['y'] - curr_wp[1])**2),
            motor_outputs['soc_pct'], self.battery_temp, motor_outputs['p_net_w'], throttle_pct, demands['mechanical_brake_pct'],
            self.state['x'], self.state['y'], v_terminal, current_a, motor_rpm, motor_outputs.get('soe_pct', 0.0),
            motor_outputs.get('fl_wheel_torque', 0.0) + motor_outputs.get('fr_wheel_torque', 0.0),
            motor_outputs.get('rear_axle_torque', 0.0), motor_outputs.get('fl_wheel_torque', 0.0), motor_outputs.get('fr_wheel_torque', 0.0),
            1.0 if is_redline else 0.0, demands['f_grip_f'], demands['f_grip_r'], demands['f_regen_f'], demands['f_regen_r']
        ]
        for k, v in zip(self.history.keys(), values):
            self.history[k].append(v)

        # Safety Stops: Battery Temperature (> 60C) and Energy Depletion (SOE <= 0)
        if self.battery_temp > 60.0 or motor_outputs.get('soe_pct', 0.0) <= 0.0:
            self.final_time = t
            if verbose:
                reason = "Overheat" if self.battery_temp > 60.0 else "Depletion"
                val = f"{self.battery_temp:.1f}°C" if reason == "Overheat" else "0%"
                print(f"--- CRITICAL: SIMULATION STOPPED - Battery {reason} ({val}) at t={t:.3f}s ---")
            
            if not self.lap_times:
                self.lap_times.append(t)
                self.lap_segment_indices.append((0, t_idx))
            self.running = False
            return

        self.sector_times[min(self.driver.current_waypoint_idx // self.wps_per_sector, 3)] += self.dt
        
        # Determine limiting state (Longitudinal + Lateral)
        a_lat = abs(v_curr**2 * np.tan(steering_angle) / self.p['wheelbase'])
        a_lat_limit = self.p['max_lat_accel_g'] * self.g

        if a_lat > 0.95 * a_lat_limit:
            self.grip_limited_time += self.dt
        elif throttle_pct > 0.95:
            if (self.p['max_power_kw'] * 1000.0) / (self.mass * max(v_curr, 1.0)) < self.p['max_grip_accel_g'] * self.g:
                self.power_limited_time += self.dt
            else: self.grip_limited_time += self.dt
        elif throttle_pct < -0.05: self.braking_time += self.dt
        else: self.other_time += self.dt

        if self.driver.current_waypoint_idx < self.prev_wp_idx - len(self.waypoints) // 2:
            self.laps_completed += 1
            self.lap_times.append(t - self.lap_start_time)
            self.lap_segment_indices.append((0 if self.laps_completed == 1 else self.lap_segment_indices[-1][1] + 1, t_idx))
            self.lap_start_time = t
            if verbose: print(f"Lap {self.laps_completed}/{self.p['laps_to_run']} completed in {self.lap_times[-1]:.3f}s")
        
        self.prev_wp_idx = self.driver.current_waypoint_idx
        if self.laps_completed >= self.p['laps_to_run']:
            self.final_time = t
            if len(self.lap_segment_indices) < self.laps_completed: self.lap_segment_indices.append((self.lap_segment_indices[-1][1] + 1 if self.laps_completed > 1 else 0, t_idx))
            self.running = False

        self.t += self.dt
        self.t_idx += 1

    def run(self, verbose=True, input_log=None):
        """Runs the simulation."""
        if not self.waypoints: return
        if verbose: print(f"Starting Simulation: {self.controller.motor_config} configuration | {self.mode} mode | Regen: {self.p['regen_enabled']}")
        if verbose: print(self.p)
        if input_log is not None and verbose:
            print("--- RUNNING IN LOG REPLAY VALIDATION MODE ---")
            
        while self.running and self.t < self.total_time:
            throttle = None
            steering = None
            if input_log:
                throttle = input_log['throttle'][self.t_idx] if self.t_idx < len(input_log['throttle']) else 0.0
                steering = input_log['steering'][self.t_idx] if self.t_idx < len(input_log['steering']) else 0.0
            
            self.step(throttle, steering, verbose=verbose)

    def validate_thermal_from_log(self, csv_path, show_plot=True):
        """
        Validates the thermal model by replaying electrical data from a CSV log.
        """
        log_data = []
        with open(csv_path, 'r') as f:
            rows = list(csv.DictReader(f))
            # Find the first non-zero battery temperature to avoid comparing against sensor lag/zeros
            first_valid_temp = next((float(row['Bat_Max_C']) for row in rows if float(row['Bat_Max_C']) > 0), 25.0)
            
            for row in rows:
                t_raw = float(row['Bat_Max_C'])
                log_data.append({
                    'time': float(row['Time_ms']) / 1000.0,
                    'p_pack': float(row['Pack_Power_W']),
                    'v_bat': float(row['Bat_Volts']),
                    't_actual': t_raw if t_raw > 0 else first_valid_temp,
                    'soc': float(row.get('Bat_SOC_%', 100.0)) / 100.0,
                    'v_vehicle': (float(row.get('DTI_MotorRPM', 0)) * 2 * np.pi * 0.2032) / (60.0 * self.p['gear_ratio'])
                })

        # Sync start time
        t_start = log_data[0]['time']
        sim_temp = log_data[0]['t_actual']
        self.p['initial_battery_temp'] = sim_temp
        self._reset_state()
        
        results = {'time': [], 't_sim': [], 't_actual': [], 'p_pack': [], 'r_int': []}
        non_zero_power_time = 0.0

        for i in range(1, len(log_data)):
            dt = log_data[i]['time'] - log_data[i-1]['time']
            if dt <= 0: continue

            p_pack = log_data[i]['p_pack']
            v_bat = max(log_data[i]['v_bat'], 10.0)
            current = p_pack / v_bat
            
            if abs(p_pack) > 1.0: # Track time where battery is under load (>1W)
                non_zero_power_time += dt

            # Calculate internal heat generation: Q = I^2 * R
            # Pointing to the empirical DCIR calculation in motor.py for accuracy
            r_int = self.powertrain.battery.get_internal_resistance(log_data[i]['soc'], sim_temp) 
            heat_w = (current**2) * r_int
            
            # Update simulation temperature
            v_ms = log_data[i]['v_vehicle']
            heat_removed = self.thermal.calculate_heat_removed(sim_temp, v_ms)
            sim_temp += self.thermal.get_temperature_delta(heat_w, heat_removed, dt)

            results['time'].append(log_data[i]['time'] - t_start)
            results['t_sim'].append(sim_temp)
            results['t_actual'].append(log_data[i]['t_actual'])
            results['p_pack'].append(p_pack)
            results['r_int'].append(r_int)

        error = np.array(results['t_sim']) - np.array(results['t_actual'])
        mean_err = np.mean(np.abs(error))
        max_err = np.max(np.abs(error))

        # Calculate Temperature Delta per active second (Heating Rate)
        total_delta_t = log_data[-1]['t_actual'] - log_data[0]['t_actual']
        heating_rate = total_delta_t / non_zero_power_time if non_zero_power_time > 0 else 0
        
        rel_path = os.path.join(os.path.basename(os.path.dirname(csv_path)), os.path.basename(csv_path))
        print(f"Validation Complete [{rel_path}]: Mean Error: {mean_err:.3f}°C | Max Error: {max_err:.3f}°C | Heating Rate: {heating_rate:.5f}°C/s")
        
        if show_plot:
            # Plotting Validation
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            ax1.plot(results['time'], results['t_actual'], 'r--', label='Physical Log (Bat_Max_C)')
            ax1.plot(results['time'], results['t_sim'], 'b-', label='Simulation (TractiveBatteryThermal)')
            ax1.set_ylabel('Temperature (°C)')
            ax1.set_title(f'Thermal Model Validation: {csv_path}')
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(results['time'], results['r_int'], 'g-', label='Simulated Pack Resistance')
            ax2.set_ylabel('Resistance (Ohms)')
            ax2.set_xlabel('Time (s)')
            ax2.legend()
            ax2.grid(True)
            plt.tight_layout()
            plt.savefig('thermal_validation.png')
            plt.show()
            
        return {
            'log_name': os.path.basename(csv_path),
            'log_folder': os.path.dirname(csv_path),
            'mean_error': mean_err,
            'max_error': max_err,
            'heating_rate': heating_rate,
            'non_zero_power_time': non_zero_power_time
        }

    def run_thermal_batch_study(self, logs_dir):
        """
        Runs thermal validation on all CSV files in logs_dir and plots a comparison.
        """
        all_logs = glob.glob(os.path.join(logs_dir, "**/*.CSV"), recursive=True) + \
                   glob.glob(os.path.join(logs_dir, "**/*.csv"), recursive=True)
        
        if not all_logs:
            print(f"No log files found in {logs_dir}")
            return

        summary = []
        for log_path in all_logs:
            try:
                res = self.validate_thermal_from_log(log_path, show_plot=False)
                summary.append(res)
            except Exception as e:
                print(f"Skipping {os.path.basename(log_path)}: {e}")

        if not summary: return

        summary.sort(key=lambda x: x['mean_error'])
        names = [f"{os.path.basename(s['log_folder'])}/{s['log_name']}" for s in summary]
        mean_errs = [s['mean_error'] for s in summary]
        max_errs = [s['max_error'] for s in summary]
        heating_rates = [s['heating_rate'] for s in summary]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        x = np.arange(len(names))
        
        ax1.bar(x - 0.2, mean_errs, 0.4, label='Mean Absolute Error (°C)', color='skyblue')
        ax1.bar(x + 0.2, max_errs, 0.4, label='Max Absolute Error (°C)', color='salmon')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=30, ha='right')
        ax1.set_ylabel('Temperature Error (°C)')
        ax1.set_title('Battery Thermal Model Accuracy Comparison Across All Logs')
        ax1.legend(); ax1.grid(axis='y', linestyle='--', alpha=0.7)

        ax2.bar(x, heating_rates, color='lightgreen', label='Heating Rate (°C/s non-zero pwr)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=30, ha='right')
        ax2.set_ylabel('Rate (°C / s)')
        ax2.set_title('Normalized Heating Characteristic (Delta T / Active Work Duration)')
        ax2.legend(); ax2.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('thermal_batch_comparison.png')
        plt.show()

    def get_results(self, verbose=True, log_file=None):
        scorer = FSAEScorer()
        # Total energy = capacity * change in SOE
        energy_used_wh = self.powertrain.battery.capacity_wh * (1.0 - (self.history['soe'][-1] / 100.0))
        scores = scorer.calculate_endurance_score(self.final_time, energy_used_wh, self.laps_completed)

        if verbose:
            header = "ACCELERATION SUMMARY" if self.mode == 'Acceleration' else "LAP TIME SUMMARY"
            time_label = "75m Run Time:     " if self.mode == 'Acceleration' else "Best Lap Time:    "
            primary_time = self.final_time if self.mode == 'Acceleration' else (min(self.lap_times) if self.lap_times else self.final_time)

            print("\n" + "="*30 + f"\n      {header}\n" + "="*30)
            print(f"Total Session Time:  {self.final_time:.3f} s")
            print(f"Grip Limited:        {self.grip_limited_time:.2f} s")
            print(f"Power Limited:       {self.power_limited_time:.2f} s")
            print(f"{time_label} {primary_time:.3f} s")
            print(f"Max Battery Temp:    {max(self.history['temp']):.2f} °C")
            print(f"Final SoE:           {self.history['soe'][-1]:.2f} %")
            print(f"Energy Recovered:    {self.total_regen_j / 3600.0:.2f} Wh")
            print(f"Energy Used:         {energy_used_wh:.2f} Wh")
            print(f"Energy Remaining:     {self.powertrain.battery.capacity_wh - energy_used_wh:.2f} Wh")
            '''if self.mode == 'Endurance':
                print(f"Endurance Points:    {scores['endurance_points']}")
                print(f"Efficiency Points:   {scores['efficiency_points']}")'''
            if any(self.history['redline']):
                print(f"Motor Redline:       REACHED")

        if log_file:
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.history.keys())
                writer.writerows(zip(*self.history.values()))
        
        res_time = self.final_time if self.mode == 'Acceleration' else (min(self.lap_times) if self.lap_times else 9999.0)
        return {
            'best_lap_time': res_time,
            'total_session_time': self.final_time,
            'mean_cte': np.mean(self.history['cte']) if self.history['cte'] else 999.0,
            'max_cte': max(self.history['cte']) if self.history['cte'] else 999.0,
            'laps_completed': self.laps_completed,
            'total_regen_wh': self.total_regen_j / 3600.0,
            'endurance_score': scores,
            'energy_used_wh': energy_used_wh,
            'final_temp': self.battery_temp
        }

def run_simulation(params=None, verbose=True, plot=True):
    sim = SimulationEngine(params)
    sim.run(verbose=verbose)
    res = sim.get_results(verbose=verbose, log_file='performance_comparison.csv' if verbose else '/tmp/performance_comparison_opt.csv')
    if plot:
        eb = {'Aero Drag': sim.total_aero_j, 'Roll Resistance': sim.total_rr_j, 'Kinetic Energy': sim.total_accel_j, 'Heat Losses': sim.total_heat_j, 'GLV System': 1000 * sim.final_time, 'Regen Recovery': sim.total_regen_j}
        idx = sim.lap_times.index(res['best_lap_time'])
        start, end = sim.lap_segment_indices[idx]
        best_lap = {k: v[start:end+1] for k, v in sim.history.items()}
        best_lap['time'] = [t - best_lap['time'][0] for t in best_lap['time']]
        _plot_results([best_lap], [f"Best Lap (Lap {idx + 1})"], sim.waypoints, eb, session_data=sim.history)
    return res

def _plot_results(laps_data, labels, waypoints=None, energy_breakdown=None, session_data=None):
    fig, axs = plt.subplots(8, 2, figsize=(14, 40))
    
    linestyles = ['-', '--', ':', '-.']
    
    # Helper to plot multiple laps on the same axes
    for lap, label, ls in zip(laps_data, labels, linestyles):
        # Velocity Tracking
        axs[0, 0].plot(lap['time'], lap['v'], label=f'{label} Actual', linestyle=ls)
        axs[0, 0].plot(lap['time'], lap['v_target'], alpha=0.3, linestyle=ls, color='gray')
        
        # Throttle
        axs[1, 1].plot(lap['time'], lap['throttle'], label=label, linestyle=ls)
        
        # Path - Color coded by velocity
        x = np.array(lap['x'])
        y = np.array(lap['y'])
        v = np.array(lap['v'])
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(v.min(), v.max())
        lc = LineCollection(segments, cmap='jet', norm=norm, alpha=0.9)
        lc.set_array(v)
        lc.set_linewidth(2)
        path_collection = axs[2, 0].add_collection(lc)
        
        # Proxy for legend since LineCollection doesn't support labels directly
        axs[2, 0].plot([], [], color='black', label=f'Path {label}', linestyle=ls)
        
        if label == labels[0]:
            cbar = fig.colorbar(path_collection, ax=axs[2, 0], pad=0.1)
            cbar.set_label('Velocity (m/s)')
        
        # Power
        axs[2, 1].plot(lap['time'], lap['p_draw'], label=label, linestyle=ls)
        
        # CTE
        axs[3, 0].plot(lap['time'], lap['cte'], label=label, linestyle=ls)
        
        # Velocity Error
        v_err = np.array(lap['v_target']) - np.array(lap['v'])
        axs[3, 1].plot(lap['time'], v_err, label=label, linestyle=ls)

        # Axle Torque Distribution (%)
        tf = np.abs(np.array(lap['t_front']))
        tr = np.abs(np.array(lap['t_rear']))
        tt = tf + tr
        mask = tt > 0.1
        p_front = np.zeros_like(tf)
        p_rear = np.zeros_like(tr)
        p_front[mask] = (tf[mask] / tt[mask]) * 100
        p_rear[mask] = (tr[mask] / tt[mask]) * 100
        axs[5, 0].plot(lap['time'], p_front, label=f'{label} Front %', color='blue', linestyle=ls)
        axs[5, 0].plot(lap['time'], p_rear, label=f'{label} Rear %', color='red', linestyle=ls, alpha=0.6)

        # Front Wheel Torque Comparison (%)
        tfl = np.abs(np.array(lap['t_fl']))
        tfr = np.abs(np.array(lap['t_fr']))
        ttf = tfl + tfr
        mask_f = ttf > 0.1
        pf_fl = np.zeros_like(tfl)
        pf_fr = np.zeros_like(tfr)
        pf_fl[mask_f] = (tfl[mask_f] / ttf[mask_f]) * 100
        pf_fr[mask_f] = (tfr[mask_f] / ttf[mask_f]) * 100
        axs[5, 1].plot(lap['time'], pf_fl, label=f'{label} FL %', color='blue', linestyle=ls)
        axs[5, 1].plot(lap['time'], pf_fr, label=f'{label} FR %', color='cyan', linestyle=ls, alpha=0.6)

        # Braking Traction Analysis (N)
        axs[7, 0].plot(lap['time'], lap['f_grip_f'], label=f'{label} Front Grip Limit', color='blue', linestyle='--')
        axs[7, 0].plot(lap['time'], lap['f_regen_f'], label=f'{label} Front Regen Force', color='blue', linestyle='-')
        
        axs[7, 1].plot(lap['time'], lap['f_grip_r'], label=f'{label} Rear Grip Limit', color='red', linestyle='--')
        axs[7, 1].plot(lap['time'], lap['f_regen_r'], label=f'{label} Rear Regen Force', color='red', linestyle='-')

    # Plot session-wide data for specific metrics (Temperature, SOC, Voltage, SOE)
    if session_data:
        axs[0, 1].plot(session_data['time'], session_data['temp'], label='Total Session', color='brown')
        axs[1, 0].plot(session_data['time'], session_data['soc'], label='Total Session', color='green')
        axs[4, 0].plot(session_data['time'], session_data['voltage'], label='Total Session', color='purple')
        axs[4, 1].plot(session_data['time'], session_data['soe'], label='Total Session', color='darkgreen')
    else:
        # Fallback to plotting from laps_data if no session_data is provided
        for lap, label, ls in zip(laps_data, labels, linestyles):
            axs[0, 1].plot(lap['time'], lap['temp'], label=label, linestyle=ls)
            axs[1, 0].plot(lap['time'], lap['soc'], label=label, linestyle=ls)
            axs[4, 0].plot(lap['time'], lap['voltage'], label=label, linestyle=ls)
            axs[4, 1].plot(lap['time'], lap['soe'], label=label, linestyle=ls)

    # Energy Loss Breakdown Pie Chart
    if energy_breakdown:
        names = list(energy_breakdown.keys())
        values = list(energy_breakdown.values())
        filtered_indices = [i for i, v in enumerate(values) if v > 0]
        if filtered_indices:
            f_names = [names[i] for i in filtered_indices]
            f_values = [values[i] for i in filtered_indices]
            colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'plum', 'yellowgreen']
            axs[6, 0].pie(f_values, labels=f_names, autopct='%1.1f%%', startangle=140, colors=colors[:len(f_values)])
            axs[6, 0].set_title('Total Session Energy Consumption Breakdown')

    axs[6, 1].axis('off')

    # Formatting
    axs[0, 0].set_title('Velocity Tracking Comparison')
    axs[0, 0].set_ylabel('Velocity (m/s)')

    axs[0, 1].set_title('Battery Temperature (C)')
    axs[0, 1].set_ylabel('Temperature (°C)')

    axs[1, 0].set_title('State of Charge (%)')
    axs[1, 0].set_ylabel('SOC (%)')

    axs[1, 1].set_title('Driver Throttle Input')
    axs[1, 1].set_ylabel('Throttle/Brake (-1 to 1)')

    if waypoints:
        wp_x = [wp[0] for wp in waypoints]
        wp_y = [wp[1] for wp in waypoints]
        axs[2, 0].plot(wp_x, wp_y, 'g--', alpha=0.3, label='Track Waypoints')
    
    axs[2, 0].set_title('Vehicle Path (X-Y Mapping)')
    axs[2, 0].set_xlabel('X Position (m)')
    axs[2, 0].set_ylabel('Y Position (m)')
    axs[2, 0].axis('equal')

    axs[2, 1].set_title('Power / Heat Demand (W)')
    axs[2, 1].set_ylabel('Power (W)')

    axs[3, 0].set_title('Cross Track Error (Path Deviation)')
    axs[3, 0].set_ylabel('Error (m)')

    axs[3, 1].set_title('Velocity Error (Target - Actual)')
    axs[3, 1].set_ylabel('Error (m/s)')

    axs[4, 0].set_title('Battery Voltage (V)')
    axs[4, 0].set_ylabel('Voltage (V)')

    axs[4, 1].set_title('State of Energy (%)')
    axs[4, 1].set_ylabel('SOE (%)')
    
    axs[5, 0].set_title('Axle Torque Distribution (%)')
    axs[5, 0].set_ylabel('Percentage of Total (%)')
    axs[5, 0].set_ylim(-5, 105)

    axs[5, 1].set_title('Front Wheel Torque Comparison (%)')
    axs[5, 1].set_ylabel('Percentage of Axle (%)')
    axs[5, 1].set_ylim(-5, 105)

    axs[7, 0].set_title('Front Axle Braking Traction')
    axs[7, 0].set_ylabel('Force (N)')

    axs[7, 1].set_title('Rear Axle Braking Traction')
    axs[7, 1].set_ylabel('Force (N)')

    for ax in axs.flat:
        if not ax.has_data(): continue
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    #plt.show()
    
    
if __name__ == "__main__":
    # Create the simulation engine
    #sim = SimulationEngine()
    # Run the thermal validation study against all physical logs in the folder
    #logs_dir = '/home/tadayal100/Powertrain_Sim/ai sim/logs/'
    #sim.run_thermal_batch_study(logs_dir)

    run_simulation({'mode': 'Endurance', 'laps_to_run': 22, 'motor_config': 'RWD', 'regen_enabled': True, 'num_cells': 130, 'max_regen_power_kw': 2.0, 'max_power_kw': 30.0})

    # NOTE: change log file directory
    '''log = '/home/tadayal100/Powertrain_Sim/ai sim/logs/5-30-test-weekend-logs/LOG_037.CSV'
    sim.validate_thermal_from_log(log)'''