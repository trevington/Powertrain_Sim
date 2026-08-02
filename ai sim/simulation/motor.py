import numpy as np
from scipy.interpolate import LinearNDInterpolator

class Battery:
    """
    Simulates Tractive Battery electrical and thermal behavior.
    Handles voltage sag and heat generation due to internal resistance.
    """
    def __init__(self, params=None):
        self.params = params if params is not None else {}
        self.n_series = self.params.get('num_cells', 140)
        self.cell_cap_ah = self.params.get('cell_cap_ah', 13.0)
        self.v_nom_cell = self.params.get('v_nom_cell', 3.5)
        
        # Total pack resistance including junctions (rbj)
        self.r_cell = self.params.get('r_cell', 0.0001)
        self.r_internal = self.n_series * self.r_cell
        self.dcir_safety_factor = self.params.get('dcir_safety_factor', 1.0)

        self.last_heat_gen = 0.1            # Small initial heat to avoid zero rise at start
        self.peak_power_capability = 80000.0 # 80kW FSAE Limit

        # Total Pack Energy Capacity (Wh) = N * V_nom * Ah
        self.capacity_wh = self.n_series * self.v_nom_cell * self.cell_cap_ah
        self.energy_remaining_wh = self.capacity_wh
        self.temperature = 25.0             # Celsius

        # SOE-V Curve Data (Cell Level)
        # Flipped to ensure x-coordinates (SOE) are increasing for np.interp
        self._soe_data = np.array([
            0, 1.9608, 3.921584, 5.882368, 7.843152, 9.803936, 11.76472, 13.725504, 15.686288,
            17.647072, 19.607856, 21.56864, 23.529424, 25.490208, 27.450992, 29.411776, 31.37256,
            33.333344, 35.294128, 37.254912, 39.215696, 41.17648, 43.137264, 45.098048, 47.058832,
            49.019616, 50.9804, 52.941184, 54.901968, 56.862752, 58.823536, 60.78432, 62.745104,
            64.705888, 66.666672, 68.627456, 70.58824, 72.549024, 74.509808, 76.470592, 78.431376,
            80.39216, 82.352944, 84.313728, 86.274512, 88.235296, 90.19608, 92.156864, 94.117648,
            96.078432, 98.039216, 100
        ])
        self._v_cell_data = np.array([
            3.2, 3.21, 3.246, 3.246, 3.276, 3.296, 3.3, 3.313, 3.33, 3.342, 3.353, 3.362, 3.378,
            3.391, 3.395, 3.404, 3.414, 3.419, 3.428, 3.435, 3.44, 3.447, 3.454, 3.466, 3.47,
            3.474, 3.484, 3.489, 3.5, 3.505, 3.518, 3.528, 3.54, 3.552, 3.566, 3.594, 3.614,
            3.632, 3.65, 3.668, 3.681, 3.699, 3.714, 3.737, 3.758, 3.77, 3.792, 3.8005, 3.845,
            3.868, 3.892, 4.2
        ])

    def draw_power(self, power_w, dt, battery_temp):
        """
        Calculates voltage drop and energy consumption.
        :return: (actual_voltage, heat_gen_w)
        """
        # Calculate Open Circuit Voltage (Voc) based on current SOE
        soe = (self.energy_remaining_wh / self.capacity_wh) * 100
        v_cell_oc = np.interp(soe, self._soe_data, self._v_cell_data)
        v_oc = v_cell_oc * self.n_series

        # Calculate current using Voc (approximate)
        current = power_w / v_oc

        # Dynamic DC Internal Resistance (DCIR) calculation using empirical polynomial
        ocv = v_cell_oc
        temp = battery_temp
        rbj = self.params.get('rbj', self.r_cell)
        temp_rise = self.params.get('temp_rise', 1.0)

        dcir = rbj + (0.2935676 + 
                      -0.171888 * ocv + 
                      -0.00373033 * temp + 
                      0.03292212 * ocv**2 + 
                      0.001604118 * ocv * temp + 
                      0.000009206928 * temp**2 + 
                      -0.001941274 * ocv**3 + 
                      -0.0001845216 * ocv**2 * temp + 
                      -0.000001707152 * ocv * temp**2 + 
                      -0.000000002083755 * temp**3) * temp_rise
        
        # Apply safety factor to account for aging or worst-case cell variance
        dcir *= self.dcir_safety_factor

        # Heat generation (P = I^2R)
        heat_gen_w = ((current)**2) * dcir
        
        # Update state of charge
        self.energy_remaining_wh -= (power_w * (dt / 3600))
        
        return v_oc, heat_gen_w

class MotorInverter:
    """
    Combined Motor and Inverter model.
    Handles the Emrax 208 or custom front hub motors.
    """
    def __init__(self, name="Emrax 208", max_torque=150, peak_efficiency=0.86, max_rpm=7000, max_power_kw=86.0, efficiency_csv=None):
        self.name = name
        self.max_torque = max_torque
        self.peak_efficiency = peak_efficiency
        self.max_power_w = max_power_kw * 1000.0
        self.max_omega = max_rpm * (2 * np.pi / 60)
        self.temp = 25.0

        self.efficiency_map = None
        if efficiency_csv:
            try:
                data = np.genfromtxt(efficiency_csv, delimiter=',')
                # Data format: Speed (RPM), Torque (Nm), Efficiency

                # --- Dynamic Data Decimation ---
                # Calculate resolution based on range (Aiming for ~60 bins per dimension)
                num_bins = 60.0
                rpm_range = np.ptp(data[:, 0])
                trq_range = np.ptp(data[:, 1])
                
                res_rpm = max(1.0, rpm_range / num_bins)
                res_trq = max(0.1, trq_range / num_bins)
                
                rounded_coords = np.round(data[:, :2] / [res_rpm, res_trq]) * [res_rpm, res_trq]
                unique_coords, inverse = np.unique(rounded_coords, axis=0, return_inverse=True)
                avg_eff = np.bincount(inverse, weights=data[:, 2]) / np.bincount(inverse)
                data_clean = np.column_stack((unique_coords, avg_eff))

                self.efficiency_map = LinearNDInterpolator(data_clean[:, :2], data_clean[:, 2])
                self._raw_data = data # Store for visualization
                self._res_info = (res_rpm, res_trq)
                print(f"[MotorInverter] {self.name}: Efficiency map loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load efficiency map from {efficiency_csv}: {e}")
        else:
            print(f"[MotorInverter] {self.name}: No map provided. Using peak efficiency {self.peak_efficiency}.")

    def get_mechanical_output(self, torque_request_nm, omega_rad_s):
        """
        Calculates shaft output and electrical power demand.
        """
        # Clamp torque to motor physical limits
        torque_actual = np.clip(torque_request_nm, -self.max_torque, self.max_torque)

        # Speed Limiter: If motor speed exceeds max_rpm, torque is restricted to zero or braking
        is_redline = False
        if abs(omega_rad_s) >= self.max_omega:
            is_redline = True
            torque_actual = min(0.0, torque_actual) if omega_rad_s > 0 else max(0.0, torque_actual)
        
        # Mechanical power: P = tau * omega
        p_mech = torque_actual * omega_rad_s
        
        # Determine efficiency using map or peak value
        if self.efficiency_map:
            rpm = abs(omega_rad_s) * 60 / (2 * np.pi)
            torque_abs = abs(torque_actual)

            efficiency = float(self.efficiency_map(rpm, torque_abs))
            # Fallback to peak if lookup is outside the convex hull of provided data
            if np.isnan(efficiency):
                if not hasattr(self, '_warned_fallback'):
                    #print(f"DEBUG: {self.name} operating at {rpm:.0f} RPM / {torque_abs:.1f} Nm (outside CSV range). Falling back to peak efficiency.")
                    self._warned_fallback = True
                efficiency = self.peak_efficiency
        else:
            efficiency = self.peak_efficiency
        
        if p_mech < 0: # Regeneration
            p_elec = p_mech * efficiency
            # Power Limit: If regen power exceeds motor/inverter capacity, cap it
            if abs(p_elec) > self.max_power_w:
                p_elec = -self.max_power_w
                # Adjust mechanical output to match the electrical power limit
                p_mech = p_elec / efficiency
                torque_actual = p_mech / omega_rad_s if abs(omega_rad_s) > 0.01 else 0.0
        else: # Drive
            p_elec = p_mech / efficiency
            
        heat_gen_w = abs(p_elec - p_mech)
        
        return p_mech, p_elec, heat_gen_w, torque_actual, is_redline

class Transmission:
    """
    Models the mechanical link (Chain or Gearbox).
    """
    def __init__(self, ratio=3.5, efficiency=0.98):
        self.ratio = ratio
        self.efficiency = efficiency

    def convert_to_wheel(self, motor_torque, motor_omega):
        wheel_torque = motor_torque * self.ratio * self.efficiency
        wheel_omega = motor_omega / self.ratio
        return wheel_torque, wheel_omega

    def convert_to_motor(self, wheel_torque, wheel_omega):
        motor_torque = (wheel_torque / self.ratio) / self.efficiency
        motor_omega = wheel_omega * self.ratio
        return motor_torque, motor_omega

class PowertrainSystem:
    """
    Top-level wrapper for the motor.py components.
    """
    def __init__(self, config='AWD', params=None):
        self.params = params if params is not None else {}
        self.battery = Battery(params=self.params)
        
        # CR30/CR31 Rear: Emrax 208 + Chain
        self.rear_motor = MotorInverter(
            "Emrax 208", 
            max_torque=self.params.get('rear_motor_max_torque', 150.0), 
            max_rpm=self.params.get('rear_motor_max_rpm', 7000.0), 
            max_power_kw=self.params.get('max_power_kw', 80.0),
            efficiency_csv=self.params.get('rear_efficiency_csv', '/home/tadayal100/Powertrain_Sim/ai sim/emrax208_efficiency.csv')
        )
        self.rear_trans = Transmission(ratio=self.params.get('rear_gear_ratio', self.params.get('gear_ratio', 4.6))) # Chain
        
        self.front_left = None
        self.front_right = None
        
        if config == 'AWD':
            # Hub motors typically have integrated gearboxes
            f_pwr = self.params.get('front_motor_max_power_kw', 15.0) # Configurable for study
            f_trq = self.params.get('front_motor_max_torque', 7.0)
            f_rpm = self.params.get('front_motor_max_rpm', 20000.0)
            self.front_left = MotorInverter("Hub Motor", max_torque=f_trq, max_rpm=f_rpm, max_power_kw=f_pwr, efficiency_csv=self.params.get('front_efficiency_csv'))
            self.front_right = MotorInverter("Hub Motor", max_torque=f_trq, max_rpm=f_rpm, max_power_kw=f_pwr, efficiency_csv=self.params.get('front_efficiency_csv'))
            self.front_trans = Transmission(ratio=self.params.get('front_gear_ratio', 13.4)) # Gearbox

    def step(self, motor_torque_demands, wheel_speeds, dt, battery_temp):
        """
        Updates the entire powertrain state for one time step.
        :param motor_torque_demands: Dict from powertrain.py
        :param wheel_speeds: Dict from vd.py (rad/s)
        """
        total_elec_power = 0
        total_heat_gen = 0
        powertrain_losses = 0
        outputs = {}

        # Process Rear Motor
        motor_omega_r = wheel_speeds['rear'] * self.rear_trans.ratio
        p_mech_r, p_elec_r, heat_r, tau_r, redline_r = self.rear_motor.get_mechanical_output(
            motor_torque_demands['rear_motor'], motor_omega_r
        )
        # Transmission Loss
        trans_loss_r = abs(p_mech_r) * (1.0 - self.rear_trans.efficiency)
        total_elec_power += p_elec_r
        powertrain_losses += (heat_r + trans_loss_r)
        
        # Convert to wheel torque for VD
        wheel_tau_r = tau_r * self.rear_trans.ratio * self.rear_trans.efficiency
        outputs['rear_axle_torque'] = wheel_tau_r
        outputs['motor_redline'] = redline_r

        # Process Front Motors if AWD
        if self.front_left:
            # Front Left
            m_omega_fl = wheel_speeds['front_left'] * self.front_trans.ratio
            pm_fl, pe_fl, h_fl, t_fl, redline_fl = self.front_left.get_mechanical_output(
                motor_torque_demands['front_left_motor'], m_omega_fl
            )
            # Front Right
            m_omega_fr = wheel_speeds['front_right'] * self.front_trans.ratio
            pm_fr, pe_fr, h_fr, t_fr, redline_fr = self.front_right.get_mechanical_output(
                motor_torque_demands['front_right_motor'], m_omega_fr
            )
            
            trans_loss_f = (abs(pm_fl) + abs(pm_fr)) * (1.0 - self.front_trans.efficiency)
            total_elec_power += (pe_fl + pe_fr)
            powertrain_losses += (h_fl + h_fr + trans_loss_f)
            outputs['fl_wheel_torque'] = t_fl * self.front_trans.ratio * self.front_trans.efficiency
            outputs['fr_wheel_torque'] = t_fr * self.front_trans.ratio * self.front_trans.efficiency
            outputs['motor_redline'] = outputs['motor_redline'] or redline_fl or redline_fr

        # Add constant 1kW GLV system draw
        total_elec_power += 1000  # Add 1000W for GLV system
        powertrain_losses += 1000 # Considered a 'loss' relative to tractive work

        # Battery Simulation
        actual_v, batt_heat = self.battery.draw_power(total_elec_power, dt, battery_temp)
        # total_heat_gen includes powertrain + battery internal resistance
        total_heat_gen = powertrain_losses + batt_heat
        
        soe_val = (self.battery.energy_remaining_wh / self.battery.capacity_wh) * 100
        outputs.update({
            'p_net_w': total_elec_power,
            'voltage': actual_v,
            'total_heat_w': total_heat_gen,      # System-wide for energy breakdown
            'battery_heat_w': batt_heat,        # Battery-only for thermal model
            'soc_pct': soe_val,
            'soe_pct': soe_val
        })
        
        return outputs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import os

    # Resolve CSV path relative to this file's location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, 'emrax208_efficiency.csv')

    if not os.path.exists(csv_file):
        print(f"Error: Could not find efficiency file at {csv_file}")
    else:
        # Initialize Motor with the CSV
        motor = MotorInverter(efficiency_csv=csv_file)

        # Ensure interactive mode is handled correctly for the backend
        plt.ion() if plt.isinteractive() else None

        # Create a dense grid for plotting
        rpm_grid = np.linspace(0, 7500, 200)
        torque_grid = np.linspace(0, 160, 200)
        RPM, TORQUE = np.meshgrid(rpm_grid, torque_grid)
        
        # Interpolate across the grid
        EFF = motor.efficiency_map(RPM, TORQUE)

        # Setup the plot with 1 row, 2 columns for 2D and 3D views
        fig = plt.figure(figsize=(18, 8))
        
        # 1. 2D Contour Plot
        ax1 = fig.add_subplot(1, 2, 1)
        contour = ax1.contourf(RPM, TORQUE, EFF, levels=20, cmap='viridis')
        cbar = fig.colorbar(contour, ax=ax1)
        cbar.set_label('Efficiency (%)', rotation=270, labelpad=15)

        # Plot 80kW Power Limit Curve (Torque = Power / Omega)
        rpm_limit_x = np.linspace(500, 7500, 500)
        torque_limit_y = 80000 / (rpm_limit_x * 2 * np.pi / 60)
        ax1.plot(rpm_limit_x, torque_limit_y, 'r--', linewidth=2, label='80kW Power Limit')

        ax1.scatter(motor._raw_data[:, 0], motor._raw_data[:, 1], c=motor._raw_data[:, 2], 
                    cmap='viridis', edgecolors='white', linewidths=0.5, label='CSV Data Points')
        ax1.set_title(f'{motor.name} Efficiency Contour', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Motor Speed (RPM)')
        ax1.set_ylabel('Motor Torque (Nm)')
        ax1.set_ylim(0, 160) # Ensure hyperbola doesn't stretch Y-axis at low RPM
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        
        res_rpm, res_trq = motor._res_info
        ax1.text(500, 140, f"* Dynamic Downsampling: {res_rpm:.1f}RPM / {res_trq:.1f}Nm bins\n* Piecewise Linear Topology", 
                 bbox=dict(facecolor='white', alpha=0.8), fontsize=9)

        # 2. 3D Surface Plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # Plot the surface - antialiased=True helps with visual clarity during rotation
        surf = ax2.plot_surface(RPM, TORQUE, EFF, cmap='viridis', edgecolor='none', alpha=0.8, antialiased=True)

        # Add 3D scatter to show raw points on the surface
        ax2.scatter(motor._raw_data[:, 0], motor._raw_data[:, 1], motor._raw_data[:, 2], 
                   c=motor._raw_data[:, 2], cmap='viridis', edgecolors='black', s=10)
        
        ax2.set_title(f'{motor.name} 3D Surface (Click & Drag to Rotate)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('RPM')
        ax2.set_ylabel('Torque (Nm)')
        ax2.set_zlabel('Efficiency')
        ax2.view_init(elev=30, azim=225) # Standard angle to see the torque vs RPM curve clearly
        ax2.mouse_init() # Explicitly initialize mouse interaction

        plt.tight_layout()
        # Save first, then show. In some backends, plt.show() clears the figure event loop on close.
        fig.savefig('emrax208_efficiency_visual.png')
        print(f"Plot saved to emrax208_efficiency_visual.png. Opening interactive window...")
        plt.show(block=True)