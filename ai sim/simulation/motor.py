import numpy as np

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
        r_cell = self.params.get('r_cell', 0.0001)
        self.r_internal = self.n_series * r_cell

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
        voltage_drop = current * self.r_internal
        actual_v = v_oc - voltage_drop
        
        # Heat generation (P = I^2R)
        heat_gen_w = (current**2) * self.r_internal
        
        # Update state of charge
        self.energy_remaining_wh -= (power_w * (dt / 3600))
        
        return actual_v, heat_gen_w

class MotorInverter:
    """
    Combined Motor and Inverter model.
    Handles the Emrax 208 or custom front hub motors.
    """
    def __init__(self, name="Emrax 208", max_torque=150, peak_efficiency=0.96, max_rpm=7000, max_power_kw=100.0):
        self.name = name
        self.max_torque = max_torque
        self.peak_efficiency = peak_efficiency
        self.max_power_w = max_power_kw * 1000.0
        self.max_omega = max_rpm * (2 * np.pi / 60)
        self.temp = 25.0

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
        
        efficiency = self.peak_efficiency
        
        if p_mech < 0: # Regeneration
            p_elec = p_mech * efficiency
            # Power Limit: If regen power exceeds motor/inverter capacity, cap it
            if abs(p_elec) > self.max_power_w:
                p_elec = -self.max_power_w
                # Adjust mechanical output to match the electrical power limit
                p_mech = p_elec / efficiency
                torque_actual = p_mech / omega_rad_s if abs(omega_rad_s) > 0.01 else 0.0

        if p_mech >= 0:
            p_elec = p_mech / efficiency
        else:
            p_elec = p_mech * efficiency  # Regen
            
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
            max_power_kw=self.params.get('max_power_kw', 80.0)
        )
        self.rear_trans = Transmission(ratio=self.params.get('rear_gear_ratio', self.params.get('gear_ratio', 4.6))) # Chain
        
        self.front_left = None
        self.front_right = None
        
        if config == 'AWD':
            # Hub motors typically have integrated gearboxes
            f_pwr = self.params.get('front_motor_max_power_kw', 15.0) # Configurable for study
            f_trq = self.params.get('front_motor_max_torque', 7.0)
            f_rpm = self.params.get('front_motor_max_rpm', 20000.0)
            self.front_left = MotorInverter("Hub Motor", max_torque=f_trq, max_rpm=f_rpm, max_power_kw=f_pwr)
            self.front_right = MotorInverter("Hub Motor", max_torque=f_trq, max_rpm=f_rpm, max_power_kw=f_pwr)
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