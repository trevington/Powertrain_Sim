import numpy as np

class PowertrainController:
    """
    Powertrain Controller for CR30/CR31.
    Translates Driver inputs (throttle/steering) into motor torque requests
    while enforcing FSAE rules and car-specific motor configurations.
    """
    def __init__(self, motor_config='AWD', max_power_kw=80.0, params=None):
        """
        Initializes the controller with a specific motor configuration.
        
        :param motor_config: 'RWD' (1 Rear Motor - CR30) or 'AWD' (1 Rear, 2 Front Hub Motors - CR31)
        :param max_power_kw: FSAE power limit (default 80kW)
        """
        self.params = params if params is not None else {}
        self.motor_config = motor_config
        self.max_power_w = max_power_kw * 1000.0
        
        # Split ratios for AWD. These are candidates for OpenMDAO optimization.
        self.rear_bias = self.params.get('rear_bias', 0.85 if motor_config == 'AWD' else 1.0)
        self.front_bias = self.params.get('front_bias', 0.15 if motor_config == 'AWD' else 0.0)
        
        # Transmission parameters for torque scaling (Must match motor.py)
        self.wheel_radius = self.params.get('wheel_radius', 0.203)
        self.rear_ratio = self.params.get('rear_gear_ratio', self.params.get('gear_ratio', 4.6))
        self.front_ratio = self.params.get('front_gear_ratio', 15.0)
        self.transmission_eff = self.params.get('transmission_eff', 0.98)
        self.wheelbase = self.params.get('wheelbase', 1.53)
        self.tv_kp = self.params.get('tv_kp', 600.0)
        self.tv_kd = self.params.get('tv_kd', 25.0)
        self.mass = self.params.get('mass', 300.0)
        self.max_brake_accel_g = self.params.get('max_brake_accel_g', 1.5)

        self.prev_yaw_error = 0.0

    def calculate_demands(self, throttle_pct, steering_angle, vehicle_state, wheel_speeds=None, dt=0.01):
        """
        Calculates the electrical and mechanical demands for the powertrain.
        
        :param throttle_pct: Input from driver.py [-1.0 to 1.0]
        :param steering_angle: Input from driver.py (radians)
        :param vehicle_state: State dictionary containing 'v' (velocity in m/s)
        :return: Dictionary with battery power demand, motor torques, and brake signals
        """
        velocity = max(vehicle_state.get('v', 0.1), 0.1) # Prevent division by zero
        yaw_rate = vehicle_state.get('yaw_rate', 0.0)
        
        # 1. Handle Negative Throttle (Braking/Regen)
        if throttle_pct < 0:
            return self._handle_braking(abs(throttle_pct), velocity, steering_angle, yaw_rate, dt)

        # 2. Acceleration: Total Power Request (Capped by 80kW Rule)
        requested_power = throttle_pct * self.max_power_w
        actual_power = min(requested_power, self.max_power_w)

        # 3. Torque Distribution
        # For AWD, we split the torque and apply basic torque vectoring based on steering.
        # actual_power / velocity = Total Tractive Force (N)
        total_force_req = actual_power / velocity
        total_torque_req = total_force_req * self.wheel_radius # Total Wheel Torque (Nm)
        motor_requests = self._distribute_torque(total_torque_req, steering_angle, yaw_rate, velocity, dt)
        
        # 4. Traction Control System (TCS)
        if wheel_speeds and throttle_pct > 0:
            motor_requests = self._apply_tcs(motor_requests, velocity, wheel_speeds)

        return {
            'battery_power_demand_w': actual_power,
            'motor_torques': motor_requests,
            'mechanical_brake_pct': 0.0
        }

    def _distribute_torque(self, total_torque, steering_angle, yaw_rate, velocity, dt):
        """Allocates torque to motors based on CR30/CR31 configuration."""
        torques = {
            'rear_motor': 0.0,
            'front_left_motor': 0.0,
            'front_right_motor': 0.0
        }
        
        if self.motor_config == 'RWD':
            torques['rear_motor'] = total_torque
            # Convert wheel torque to motor shaft torque
            torques['rear_motor'] = (total_torque / self.rear_ratio) / self.transmission_eff
        else:
            # --- Dynamic Longitudinal Split ---
            # Shift more torque to front at high speed (0.15 @ 0m/s -> 0.45 @ 25m/s)
            # Shifting bias forward stabilizes high-speed oversteer.
            speed_factor = np.clip(velocity / 25.0, 0, 1)
            current_front_bias = self.front_bias + (0.30 * speed_factor)
            current_rear_bias = 1.0 - current_front_bias
            
            rear_torque = total_torque * current_rear_bias
            front_torque_total = total_torque * current_front_bias
            
            # 1. Calculate kinematic target yaw rate: r = v / radius = (v * tan(delta)) / L
            target_yaw_rate = (velocity * np.tan(steering_angle)) / self.wheelbase
            
            # 2. Calculate yaw error and error rate (Derivative)
            yaw_error = target_yaw_rate - yaw_rate
            yaw_error_dot = (yaw_error - self.prev_yaw_error) / dt if dt > 0 else 0
            self.prev_yaw_error = yaw_error
            
            # 3. Dynamic PD TV Gain
            # Higher gain at low speed for agility, lower gain at high speed for stability.
            k_p_low, k_p_high = self.tv_kp, 150.0
            k_d_low, k_d_high = self.tv_kd, 10.0 # Damping to catch the rear end
            
            v_ref_high = 25.0 # Speed (m/s) at which we reach minimum gain
            k_p = k_p_low - (k_p_low - k_p_high) * speed_factor
            k_d = k_d_low - (k_d_low - k_d_high) * speed_factor
            
            tv_offset = np.clip(k_p * yaw_error + k_d * yaw_error_dot, -abs(front_torque_total)*0.6, abs(front_torque_total)*0.6)

            # Convert wheel torques to motor shaft torques for motor.py inverter model
            torques['rear_motor'] = (rear_torque / self.rear_ratio) / self.transmission_eff
            
            torques['front_left_motor'] = (((front_torque_total / 2.0) - tv_offset) / self.front_ratio) / self.transmission_eff
            torques['front_right_motor'] = (((front_torque_total / 2.0) + tv_offset) / self.front_ratio) / self.transmission_eff
            
        return torques

    def _handle_braking(self, brake_pct, velocity, steering_angle, yaw_rate, dt):
        """Logic for blending regenerative and mechanical braking."""
        # Calculate target braking wheel torque based on driver input
        # Max force = mass * g * max_brake_g
        max_brake_force = self.mass * 9.81 * self.max_brake_accel_g
        target_wheel_torque = -brake_pct * max_brake_force * self.wheel_radius

        # Distribute negative torque to motors for regeneration
        # We reuse the AWD distribution logic to ensure stability during braking
        motor_requests = self._distribute_torque(target_wheel_torque, steering_angle, yaw_rate, velocity, dt)

        # Blend with mechanical brakes: If speed is low, or brake demand is extremely high, 
        # the mechanical system acts as the primary/backup.
        mech_brake_pct = brake_pct if velocity < 2.0 else max(0.0, brake_pct - 0.4)
             
        return {
            'battery_power_demand_w': 0.0, # Power will be generated by motor steps
            'motor_torques': motor_requests,
            'mechanical_brake_pct': mech_brake_pct
        }

    def _apply_tcs(self, motor_torques, velocity, wheel_speeds):
        """
        Reduces motor torque if the longitudinal slip ratio exceeds a target threshold.
        Longitudinal Slip (kappa) = (v_wheel - v_car) / v_car
        """
        slip_threshold = 0.12  # Target slip for maximum tractive force (typically 10-15%)
        tcs_kp = 8.0           # Proportional gain for torque reduction
        wheel_radius = 0.203   # Must match vehicle dynamics model
        
        # Map motors to their respective wheel speed feedback
        mapping = {'rear_motor': 'rear', 'front_left_motor': 'front_left', 'front_right_motor': 'front_right'}
        
        v_ref = max(velocity, 1.0) # Avoid division by zero and noise at standstill
        
        for motor, wheel in mapping.items():
            if motor not in motor_torques or motor_torques[motor] <= 0:
                continue
                
            v_wheel = wheel_speeds.get(wheel, 0.0) * wheel_radius
            slip = (v_wheel - velocity) / v_ref
            
            if slip > slip_threshold:
                # Reduce torque request: torque = torque * (1 - Kp * slip_error)
                reduction_factor = np.clip(1.0 - tcs_kp * (slip - slip_threshold), 0.1, 1.0)
                motor_torques[motor] *= reduction_factor
                
        return motor_torques

if __name__ == "__main__":
    # Quick validation test
    pc = PowertrainController(motor_config='AWD')
    test_state = {'v': 10.0}
    print(f"AWD @ 50% Throttle: {pc.calculate_demands(0.5, 0.2, test_state)}")
