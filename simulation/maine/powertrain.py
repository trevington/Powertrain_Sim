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
        self.regen_enabled = self.params.get('regen_enabled', True)
        self.dcir_safety_factor = self.params.get('dcir_safety_factor', 1.5)
        
        # Split ratios for AWD. These are candidates for OpenMDAO optimization.
        self.rear_bias = self.params.get('rear_bias', 0.85 if motor_config == 'AWD' else 1.0)
        self.front_bias = self.params.get('front_bias', 0.15 if motor_config == 'AWD' else 0.0)
        
        # Transmission parameters for torque scaling (Must match motor.py)
        self.wheel_radius = self.params.get('wheel_radius', 0.203)
        self.rear_ratio = self.params.get('rear_gear_ratio', self.params.get('gear_ratio', 4.6))
        self.front_ratio = self.params.get('front_gear_ratio', 15.0)
        self.transmission_eff = self.params.get('transmission_eff', 0.98)
        self.wheelbase = self.params.get('wheelbase', 1.53)
        self.h_cg = self.params.get('h_cg', 0.3302) # 13 inches in meters
        
        # Weight and Aero distributions (percentage on REAR)
        self.weight_dist_rear = self.params.get('weight_dist_rear', 0.53) 
        self.aero_dist_rear = self.params.get('aero_dist_rear', 0.55)

        self.tv_kp = self.params.get('tv_kp', 600.0)
        self.tv_kd = self.params.get('tv_kd', 25.0)
        self.mass = self.params.get('mass', 300.0)
        self.max_brake_accel_g = self.params.get('max_brake_accel_g', 1.5)
        self.max_grip_accel_g = self.params.get('max_grip_accel_g', 0.9656)
        self.prev_yaw_error = 0.0
        self.slip_integral = {'rear': 0.0, 'front_left': 0.0, 'front_right': 0.0}

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
        accel = vehicle_state.get('a', 0.0) # Assume acceleration is passed in state
        
        # 1. Handle Negative Throttle (Braking/Regen)
        if throttle_pct < 0:
            return self._handle_braking(abs(throttle_pct), velocity, accel, steering_angle, yaw_rate, dt)

        # 2. Acceleration: Total Power Request (Capped by 80kW Rule)
        requested_power = throttle_pct * self.max_power_w
        actual_power = min(requested_power, self.max_power_w)

        # 3. Torque Distribution
        # For AWD, we split the torque and apply basic torque vectoring based on steering.
        # actual_power / velocity = Total Tractive Force (N)
        total_force_req = actual_power / velocity
        total_torque_req = total_force_req * self.wheel_radius # Total Wheel Torque (Nm)
        motor_requests = self._distribute_torque(total_torque_req, steering_angle, yaw_rate, vehicle_state, dt)
        
        # 4. Traction Control System (TCS)
        if wheel_speeds and throttle_pct > 0:
            motor_requests = self._apply_tcs(motor_requests, velocity, wheel_speeds, vehicle_state, dt)

        return {
            'battery_power_demand_w': actual_power,
            'motor_torques': motor_requests,
            'mechanical_brake_pct': 0.0,
            'f_grip_f': 0.0,
            'f_grip_r': 0.0,
            'f_regen_f': 0.0,
            'f_regen_r': 0.0
        }

    def _distribute_torque(self, total_torque, steering_angle, yaw_rate, vehicle_state, dt):
        """Allocates torque to motors based on CR30/CR31 configuration."""
        torques = {
            'rear_motor': 0.0,
            'front_left_motor': 0.0,
            'front_right_motor': 0.0
        }
        
        velocity = vehicle_state.get('v', 0.1)

        if self.motor_config == 'RWD':
            torques['rear_motor'] = total_torque
            # Convert wheel torque to motor shaft torque
            torques['rear_motor'] = (total_torque / self.rear_ratio) / self.transmission_eff
        else:
            # --- Grip-Optimal Longitudinal Split ---
            # Calculate available grip on each axle based on dynamic load transfer
            f_grip_f, f_grip_r = self._calculate_dynamic_grip(vehicle_state)
            total_grip = f_grip_f + f_grip_r
            
            # Distribute torque proportional to the normal load (and thus grip)
            speed_factor = np.clip(velocity / 25.0, 0, 1)
            grip_bias_front = f_grip_f / total_grip if total_grip > 0 else self.front_bias
            
            # Stability Blend: At high speeds, we can still bias slightly forward 
            # beyond the grip-optimal split to stabilize the car.
            current_front_bias = np.clip(grip_bias_front + (0.1 * speed_factor), 0.0, 1.0)
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

    def _calculate_dynamic_grip(self, vehicle_state):
        """Calculates the maximum available longitudinal force per axle based on load transfer."""
        velocity = vehicle_state.get('v', 0.0)
        accel = vehicle_state.get('a', 0.0)
        f_downforce = vehicle_state.get('f_downforce', 0.0)

        # Load transfer term: m * a * h / L
        # positive a = load shifts to rear
        delta_fz = (self.mass * accel * self.h_cg) / self.wheelbase
        
        fz_rear = (self.mass * 9.81 * self.weight_dist_rear) + (f_downforce * self.aero_dist_rear) + delta_fz
        fz_front = (self.mass * 9.81 * (1 - self.weight_dist_rear)) + (f_downforce * (1 - self.aero_dist_rear)) - delta_fz
        
        # Traction limits
        mu = self.max_grip_accel_g
        return max(0.0, fz_front * mu), max(0.0, fz_rear * mu)

    def _handle_braking(self, brake_pct, velocity, accel, steering_angle, yaw_rate, dt):
        """Logic for blending regenerative and mechanical braking."""
        # 1. Total braking force requested by driver (N)
        max_brake_force = self.mass * 9.81 * self.max_brake_accel_g
        f_req_total = brake_pct * max_brake_force

        # 2. Use helper for dynamic grip limits
        vehicle_state = {'v': velocity, 'a': accel, 'f_downforce': 0.0} # Temp state for helper
        f_grip_front, f_grip_rear = self._calculate_dynamic_grip(vehicle_state)

        # 3. Allocation to Regen
        # Use speed-dependent bias for front/rear force allocation (matches _distribute_torque)
        speed_factor = np.clip(velocity / 25.0, 0, 1)
        current_front_bias = self.front_bias + (0.30 * speed_factor)
        current_rear_bias = 1.0 - current_front_bias
        
        if self.regen_enabled and velocity > 1.4: # 5km/hr minimum regen speed
            f_regen_req_front = f_req_total * current_front_bias
            f_regen_req_rear = f_req_total * current_rear_bias
        else:
            f_regen_req_front = 0.0
            f_regen_req_rear = 0.0

        # Limit regen force by the available traction (grip)
        f_regen_front = min(f_regen_req_front, f_grip_front)
        f_regen_rear = min(f_regen_req_rear, f_grip_rear)
        
        # 4. Convert regen force back to motor shaft torques (Nm)
        torques = {
            'rear_motor': 0.0,
            'front_left_motor': 0.0,
            'front_right_motor': 0.0
        }
        
        # Rear
        t_wheel_rear = -f_regen_rear * self.wheel_radius
        torques['rear_motor'] = (t_wheel_rear / self.rear_ratio) / self.transmission_eff
        
        # Front
        if self.motor_config == 'AWD' and f_regen_front > 0:
            t_wheel_front = -f_regen_front * self.wheel_radius
            # Split regen equally between front hubs
            torques['front_left_motor'] = ((t_wheel_front / 2.0) / self.front_ratio) / self.transmission_eff
            torques['front_right_motor'] = ((t_wheel_front / 2.0) / self.front_ratio) / self.transmission_eff

        # 5. Mechanical brakes handle the deficit
        f_regen_total = f_regen_front + f_regen_rear
        f_mech = max(0.0, f_req_total - f_regen_total)
        
        # Cutout at low speed (< 1.0 m/s) to ensure car stops reliably
        if velocity < 1.0:
            return {
                'battery_power_demand_w': 0.0,
                'motor_torques': {'rear_motor': 0.0, 'front_left_motor': 0.0, 'front_right_motor': 0.0},
                'mechanical_brake_pct': brake_pct,
                'f_grip_f': f_grip_front,
                'f_grip_r': f_grip_rear,
                'f_regen_f': 0.0,
                'f_regen_r': 0.0
            }

        mech_brake_pct = f_mech / max_brake_force
             
        return {
            'battery_power_demand_w': 0.0, # Power will be generated by motor steps
            'motor_torques': torques,
            'mechanical_brake_pct': mech_brake_pct,
            'f_grip_f': f_grip_front,
            'f_grip_r': f_grip_rear,
            'f_regen_f': f_regen_front,
            'f_regen_r': f_regen_rear
        }

    def _apply_tcs(self, motor_torques, velocity, wheel_speeds, vehicle_state, dt):
        """
        Advanced TCS using Feed-Forward Grip Limits and PI Slip Tracking.
        """
        target_slip = 0.12     # Optimal longitudinal slip for racing tires
        tcs_kp = 12.0          # Proportional gain
        tcs_ki = 5.0           # Integral gain for steady-state slip tracking
        
        # 1. Feed-Forward: Calculate physical grip limits based on current load
        f_grip_front, f_grip_rear = self._calculate_dynamic_grip(vehicle_state)
        
        # Convert Force limit (N) to Motor Torque limit (Nm)
        # For Front, we split the axle grip limit between the two hub motors
        max_t_motor_rear = ((f_grip_rear * self.wheel_radius) / self.rear_ratio) / self.transmission_eff
        max_t_motor_front = (((f_grip_front / 2.0) * self.wheel_radius) / self.front_ratio) / self.transmission_eff
        
        mapping = {'rear_motor': 'rear', 'front_left_motor': 'front_left', 'front_right_motor': 'front_right'}
        v_ref = max(velocity, 1.0) # Avoid division by zero and noise at standstill
        
        for motor, wheel in mapping.items():
            if motor not in motor_torques or motor_torques[motor] <= 0:
                self.slip_integral[wheel] = 0.0
                continue
                
            # 2. Feedback: Calculate Slip Error
            v_wheel = wheel_speeds.get(wheel, 0.0) * self.wheel_radius
            slip = (v_wheel - velocity) / v_ref
            slip_error = slip - target_slip
            
            # Only intervene if we are exceeding target slip
            if slip_error > 0:
                self.slip_integral[wheel] += slip_error * dt
                # Limit I-term to prevent windup
                self.slip_integral[wheel] = min(self.slip_integral[wheel], 0.5)
                
                # PI Reduction Factor (0.0 to 1.0)
                reduction = (tcs_kp * slip_error) + (tcs_ki * self.slip_integral[wheel])
                reduction_factor = np.clip(1.0 - reduction, 0.05, 1.0)
                motor_torques[motor] *= reduction_factor
            else:
                self.slip_integral[wheel] = 0.0

            # 3. Physical Ceiling: Never request more torque than the tire can physically ground
            # This is our Feed-Forward safety net.
            if 'rear' in motor:
                motor_torques[motor] = min(motor_torques[motor], max_t_motor_rear)
            else:
                motor_torques[motor] = min(motor_torques[motor], max_t_motor_front)
                
        return motor_torques

if __name__ == "__main__":
    # Quick validation test
    pc = PowertrainController(motor_config='AWD')
    test_state = {'v': 10.0}
    print(f"AWD @ 50% Throttle: {pc.calculate_demands(0.5, 0.2, test_state)}")
