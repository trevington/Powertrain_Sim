import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv
from scipy.interpolate import RegularGridInterpolator
from simulation.driver import PathCommander
from simulation.powertrain import PowertrainController
from simulation.motor import PowertrainSystem
from simulation.vd import VehicleDynamics
from simulation.tb_thermal import TractiveBatteryThermal

class SimulationEngine:
    def __init__(self, params=None):
        self.track_file = '/home/tadayal100/Powertrain_Sim/ai sim/tracks/endurance.csv'
        self.p = {
            'num_cells': 140, 'max_power_kw': 80.0, 'gear_ratio': 4.6,
            'max_lat_accel_g': 2.0753, 'max_brake_accel_g': 1.49, 'max_grip_accel_g': 0.9656,
            'max_v': 28.41, 'ambient_temp': 25.0, 'initial_battery_temp': 38.0,
            'laps_to_run': 22, 'motor_config': 'RWD', 'tv_kp': 600.0, 'tv_kd': 25.0,
            'steer_kp': 0.8, 'steer_kd': 0.2, 'lookahead_dist': 4.0, 'lookahead_gain': 0.3,
            'wheelbase': 1.535
        }
        if params: self.p.update(params)

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
            p1, p2, p3 = np.array(coords[i-1]), np.array(coords[i]), np.array(coords[(i+1)%len(coords)])
            a, b, c = np.linalg.norm(p1-p2), np.linalg.norm(p2-p3), np.linalg.norm(p3-p1)
            tri_area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
            curvature = 0.0
            if tri_area > 0.005:
                radius = (a * b * c) / (4 * tri_area)
                curvature = 1.0 / radius
                target_v = min(np.sqrt(max_lat_accel * radius), self.p['max_v'])
            else: target_v = self.p['max_v']
            waypoints.append([coords[i][0], coords[i][1], target_v, curvature])

        for _ in range(2):
            for i in range(len(waypoints) - 1, -1, -1):
                next_idx = (i + 1) % len(waypoints)
                dist = np.linalg.norm(np.array(waypoints[next_idx][:2]) - np.array(waypoints[i][:2]))
                v_allowed = np.sqrt(waypoints[next_idx][2]**2 + 2 * max_brake_accel * dist)
                if waypoints[i][2] > v_allowed: waypoints[i][2] = v_allowed

        for _ in range(2):
            for i in range(len(waypoints)):
                next_idx = (i + 1) % len(waypoints)
                v_curr = max(waypoints[i][2], 1.0)
                dynamic_max_accel = max(0.0, min(max_grip_accel, (self.p['max_power_kw']*1000.0)/(self.mass*v_curr)) - (np.interp(v_curr, self.aero_v_ms, self.aero_drag_n, left=0.0)/self.mass) - np.interp(v_curr, self.rr_v_ms, self.rr_coeff)*self.g)
                dist = np.linalg.norm(np.array(waypoints[i][:2]) - np.array(waypoints[next_idx][:2]))
                v_allowed = np.sqrt(waypoints[i][2]**2 + 2 * dynamic_max_accel * dist)
                if waypoints[next_idx][2] > v_allowed: waypoints[next_idx][2] = v_allowed

        v_smooth = np.convolve(np.pad(np.array([wp[2] for wp in waypoints]), 2, mode='wrap'), np.ones(5)/5, mode='valid')
        for i in range(len(waypoints)): waypoints[i][2] = v_smooth[i]
        return waypoints

    def _init_components(self):
        self.driver = PathCommander(track_waypoints=self.waypoints, mode='Endurance', params=self.p)
        self.controller = PowertrainController(motor_config=self.p['motor_config'], max_power_kw=self.p['max_power_kw'], params=self.p)
        self.powertrain = PowertrainSystem(config=self.p['motor_config'], params=self.p)
        self.vd = VehicleDynamics(mass=self.mass, params=self.p)
        self.thermal = TractiveBatteryThermal(ambient_temp=self.p['ambient_temp'], params=self.p)

    def _reset_state(self):
        if not self.waypoints: return
        initial_yaw = np.arctan2(self.waypoints[1][1] - self.waypoints[0][1], self.waypoints[1][0] - self.waypoints[0][0])
        self.state = {'x': self.waypoints[0][0], 'y': self.waypoints[0][1], 'yaw': initial_yaw, 'v': 0.0, 'yaw_rate': 0.0}
        self.vd.x, self.vd.y, self.wheel_speeds, self.battery_temp = self.state['x'], self.state['y'], {'rear': 0.0, 'front_left': 0.0, 'front_right': 0.0}, self.p['initial_battery_temp']
        self.history = {k: [] for k in ['time', 'v', 'v_target', 'cte', 'soc', 'temp', 'p_draw', 'throttle', 'x', 'y', 'voltage', 'soe', 't_front', 't_rear', 't_fl', 't_fr']}
        self.grip_limited_time = self.power_limited_time = self.braking_time = self.other_time = self.total_heat_j = self.total_aero_j = self.total_rr_j = self.total_accel_j = self.total_regen_j = 0.0
        self.laps_completed, self.lap_times, self.lap_segment_indices, self.lap_start_time, self.sector_times = 0, [], [], 0.0, [0.0]*4

    def run(self, verbose=True):
        if not self.waypoints: return
        if verbose: print(f"Starting Simulation: {self.controller.motor_config} configuration...")
        wps_per_sector, prev_wp_idx = max(1, len(self.waypoints) // 4), 0

        for t_idx, t in enumerate(np.arange(0, self.total_time, self.dt)):
            throttle_pct, _ = self.driver.get_commands(self.state, self.dt)
            Ld = max(self.p['lookahead_dist'], self.p['lookahead_gain'] * self.state['v'])
            lookahead_idx = self.driver.current_waypoint_idx
            for i in range(len(self.waypoints)):
                idx = (self.driver.current_waypoint_idx + i) % len(self.waypoints)
                if np.linalg.norm(np.array(self.waypoints[idx][:2]) - np.array([self.state['x'], self.state['y']])) >= Ld:
                    lookahead_idx = idx
                    break
            target_wp = self.waypoints[lookahead_idx]
            alpha = (np.arctan2(target_wp[1] - self.state['y'], target_wp[0] - self.state['x']) - self.state['yaw'] + np.pi) % (2*np.pi) - np.pi
            steering_angle = np.arctan2(2.0 * self.p['wheelbase'] * np.sin(alpha), Ld)

            demands = self.controller.calculate_demands(throttle_pct, steering_angle, self.state, self.wheel_speeds, self.dt)
            motor_outputs = self.powertrain.step(demands['motor_torques'], self.wheel_speeds, self.dt, self.battery_temp)
            self.battery_temp += self.thermal.get_temperature_delta(motor_outputs.get('battery_heat_w', 0.0), self.thermal.calculate_heat_removed(self.battery_temp, self.state['v']), self.dt)
            self.total_heat_j += motor_outputs['total_heat_w'] * self.dt

            v_prev = self.state['v']
            self.state, self.wheel_speeds = self.vd.step({'rear_axle_torque': motor_outputs.get('rear_axle_torque', 0), 'fl_wheel_torque': motor_outputs.get('fl_wheel_torque', 0), 'fr_wheel_torque': motor_outputs.get('fr_wheel_torque', 0), 'mechanical_brake_pct': demands['mechanical_brake_pct']}, steering_angle, self.dt)

            v_curr = self.state['v']
            self.total_aero_j += np.interp(v_curr, self.aero_v_ms, self.aero_drag_n, left=0.0) * v_curr * self.dt
            self.total_rr_j += np.interp(v_curr, self.rr_v_ms, self.rr_coeff) * self.mass * self.g * v_curr * self.dt
            if v_curr > v_prev: self.total_accel_j += self.mass * ((v_curr - v_prev) / self.dt) * v_curr * self.dt
            if demands['battery_power_demand_w'] < 0: self.total_regen_j += abs(demands['battery_power_demand_w']) * self.dt

            curr_wp = self.waypoints[self.driver.current_waypoint_idx]
            for k, v in zip(self.history.keys(), [t, v_curr, curr_wp[2], np.sqrt((self.state['x'] - curr_wp[0])**2 + (self.state['y'] - curr_wp[1])**2), motor_outputs['soc_pct'], self.battery_temp, demands['battery_power_demand_w'], throttle_pct, self.state['x'], self.state['y'], motor_outputs.get('voltage', 0.0), motor_outputs.get('soe_pct', 0.0), motor_outputs.get('fl_wheel_torque', 0.0) + motor_outputs.get('fr_wheel_torque', 0.0), motor_outputs.get('rear_axle_torque', 0.0), motor_outputs.get('fl_wheel_torque', 0.0), motor_outputs.get('fr_wheel_torque', 0.0)]):
                self.history[k].append(v)

            self.sector_times[min(self.driver.current_waypoint_idx // wps_per_sector, 3)] += self.dt
            if throttle_pct > 0.95:
                if (self.p['max_power_kw'] * 1000.0) / (self.mass * max(v_curr, 1.0)) < self.p['max_grip_accel_g']*self.g: self.power_limited_time += self.dt
                else: self.grip_limited_time += self.dt
            elif throttle_pct < -0.05: self.braking_time += self.dt
            else: self.other_time += self.dt

            if self.driver.current_waypoint_idx < prev_wp_idx - len(self.waypoints) // 2:
                self.laps_completed += 1
                self.lap_times.append(t - self.lap_start_time)
                self.lap_segment_indices.append((0 if self.laps_completed == 1 else self.lap_segment_indices[-1][1] + 1, t_idx))
                self.lap_start_time = t
                if verbose: print(f"Lap {self.laps_completed}/{self.p['laps_to_run']} completed in {self.lap_times[-1]:.3f}s")
            prev_wp_idx = self.driver.current_waypoint_idx
            if self.laps_completed >= self.p['laps_to_run']:
                self.final_time = t
                if len(self.lap_segment_indices) < self.laps_completed: self.lap_segment_indices.append((self.lap_segment_indices[-1][1] + 1 if self.laps_completed > 1 else 0, t_idx))
                break

    def get_results(self, verbose=True, log_file=None):
        if verbose:
            print("\n" + "="*30 + "\n      LAP TIME SUMMARY\n" + "="*30)
            print(f"Total Session Time:  {self.final_time:.3f} s\nGrip Limited:        {self.grip_limited_time:.2f} s\nPower Limited:       {self.power_limited_time:.2f} s\nBest Lap Time:       {min(self.lap_times):.3f} s")
            print(f"Max Battery Temp:    {max(self.history['temp']):.2f} °C\nFinal SoE:           {self.history['soe'][-1]:.2f} %")
        if log_file:
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.history.keys())
                writer.writerows(zip(*self.history.values()))
        return {
            'best_lap_time': min(self.lap_times) if self.lap_times else 9999.0,
            'mean_cte': np.mean(self.history['cte']) if self.history['cte'] else 999.0,
            'max_cte': max(self.history['cte']) if self.history['cte'] else 999.0,
            'laps_completed': self.laps_completed
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
        _plot_results([best_lap], [f"Best Lap (Lap {idx + 1})"], sim.waypoints, eb)
    return res

def _plot_results(laps_data, labels, waypoints=None, energy_breakdown=None):
    fig, axs = plt.subplots(7, 2, figsize=(14, 35))
    
    linestyles = ['-', '--', ':', '-.']
    
    # Helper to plot multiple laps on the same axes
    for lap, label, ls in zip(laps_data, labels, linestyles):
        # Velocity Tracking
        axs[0, 0].plot(lap['time'], lap['v'], label=f'{label} Actual', linestyle=ls)
        axs[0, 0].plot(lap['time'], lap['v_target'], alpha=0.3, linestyle=ls, color='gray')
        
        # Temperature
        axs[0, 1].plot(lap['time'], lap['temp'], label=label, linestyle=ls)
        
        # SOC
        axs[1, 0].plot(lap['time'], lap['soc'], label=label, linestyle=ls)
        
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

        # Voltage
        axs[4, 0].plot(lap['time'], lap['voltage'], label=label, linestyle=ls)
        
        # SOE
        axs[4, 1].plot(lap['time'], lap['soe'], label=label, linestyle=ls)

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

    for ax in axs.flat[:-2]:
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.show()
    

if __name__ == "__main__":
    run_simulation()