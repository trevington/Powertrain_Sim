import numpy as np

class VehicleController:
    def __init__(self, layout="RWD", max_regen_power_kw=60.0):
        """
        Formula SAE Vehicle Control Unit (VCU) Simulator Block.
        
        Parameters:
        -----------
        layout : str
            "RWD" for single rear motor, "3-MOTOR" for 2 front hub + 1 rear inboard.
        max_regen_power_kw : float
            Maximum allowable regenerative braking power limit (Rule constraint or battery cell charge limit).
        """
        self.layout = layout.upper()
        self.max_regen_power = max_regen_power_kw * 1000.0  # Convert to Watts
        
        # Hardcoded vehicle metrics (Replace with your actual vehicle config object)
        self.r_eff = 0.2032          # 8-inch tire radius in meters
        self.brake_bias_front = 0.60 # 60% Front / 40% Rear mechanical brake distribution
        self.max_brake_torque = 1600 # Total vehicle mechanical braking capacity (Nm)
        
        # Motor Torque Limits (Realistic Emrax / Hub Motor constraints)
        self.max_torque_rear_peak = 210.0   # Nm
        self.max_torque_front_peak = 80.0   # Nm (Per hub motor)
        
        # Torque Vectoring Parameters (For 3-Motor Layout)
        self.tv_gain = 150.0  # Nm of torque delta per radian of steering angle

    def allocate_actuators(self, throttle, brake, steering_angle, v_car, wheel_speeds, thermal_derating=1.0):
        """
        Processes driver inputs and current vehicle states to command motor and brake torques.
        
        Parameters:
        -----------
        throttle : float (0.0 to 1.0)
        brake : float (0.0 to 1.0)
        steering_angle : float (radians, positive = steering left)
        v_car : float (Chassis forward velocity in m/s)
        wheel_speeds : dict
            Dictionary containing rad/s values for keys: 'FL', 'FR', 'RL', 'RR'
        thermal_derating : float (0.0 to 1.0)
            Multiplier from the thermal plant. 1.0 = healthy, 0.0 = critical thermal shutdown.
            
        Returns:
        --------
        motor_cmds : dict
            Target tractive/regen torques in Nm: {'FL': float, 'FR': float, 'Rear': float}
        brake_cmds : dict
            Target mechanical friction braking torques in Nm: {'FL': float, 'FR': float, 'RL': float, 'RR': float}
        """
        # 1. Initialize empty command dictionaries
        motor_cmds = {'FL': 0.0, 'FR': 0.0, 'Rear': 0.0}
        brake_cmds = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        
        # 2. Process Mechanical Braking Command
        total_friction_brake = brake * self.max_brake_torque
        
        brake_cmds['FL'] = (total_friction_brake * self.brake_bias_front) / 2.0
        brake_cmds['FR'] = (total_friction_brake * self.brake_bias_front) / 2.0
        brake_cmds['RL'] = (total_friction_brake * (1.0 - self.brake_bias_front)) / 2.0
        brake_cmds['RR'] = (total_friction_brake * (1.0 - self.brake_bias_front)) / 2.0

        # 3. Process Tractive Torque (Throttle)
        if throttle > 0.001 and brake < 0.05:  # Implements basic APPS/Brake pedal plausibility rule
            
            # Apply thermal system limits immediately to max torque ceiling
            current_max_rear = self.max_torque_rear_peak * thermal_derating
            current_max_front = self.max_torque_front_peak * thermal_derating
            
            if self.layout == "RWD":
                # Single rear motor receives all requested throttle
                motor_cmds['Rear'] = throttle * current_max_rear
                
            elif self.layout == "3-MOTOR":
                # Base torque split (e.g., 50% Rear, 50% Front total split evenly across hub motors)
                base_rear = throttle * current_max_rear * 0.50
                base_front_total = throttle * current_max_front * 2.0 * 0.50
                
                # Simple Feed-Forward Torque Vectoring based on steering angle
                # Steering Left (positive) -> Add torque to Outer Right wheel, subtract from Inner Left wheel
                tv_delta = steering_angle * self.tv_gain * throttle
                
                motor_cmds['Rear'] = base_rear
                motor_cmds['FL'] = np.clip((base_front_total / 2.0) - tv_delta, 0.0, current_max_front)
                motor_cmds['FR'] = np.clip((base_front_total / 2.0) + tv_delta, 0.0, current_max_front)
        
        # 4. Process Regenerative Braking (if driver is pressing the brake pedal)
        elif brake > 0.01 and v_car > 2.0:  # Disable regen below 2 m/s (~4.5 mph) to avoid solver instability
            motor_cmds = self._calculate_regen_torque(brake, v_car, wheel_speeds)
            
        return motor_cmds, brake_cmds

    def _calculate_regen_torque(self, brake, v_car, wheel_speeds):
        """Internal helper to calculate regenerative braking limits based on power boundaries."""
        regen_cmds = {'FL': 0.0, 'FR': 0.0, 'Rear': 0.0}
        
        # Simple proportional mapping: more brake pedal = more regen request up to power limits
        target_regen_power = brake * self.max_regen_power
        
        if self.layout == "RWD":
            omega_motor = wheel_speeds['RL']  # Assuming fixed gearing 1:1 or constant reduction ratio
            if omega_motor > 5.0:
                # Power = Torque * Omega -> Target Torque = Power / Omega
                torque_needed = -target_regen_power / omega_motor
                # Enforce physical peak motor constraints (negative for braking)
                regen_cmds['Rear'] = max(torque_needed, -self.max_torque_rear_peak)
                
        elif self.layout == "3-MOTOR":
            # Share regen load: 40% rear motor, 60% shared across front hub motors to exploit forward weight transfer
            omega_rear = wheel_speeds['RL']
            omega_fl = wheel_speeds['FL']
            omega_fr = wheel_speeds['FR']
            
            if omega_rear > 5.0:
                regen_cmds['Rear'] = max((-target_regen_power * 0.40) / omega_rear, -self.max_torque_rear_peak)
            if omega_fl > 5.0:
                regen_cmds['FL'] = max((-target_regen_power * 0.30) / omega_fl, -self.max_torque_front_peak)
            if omega_fr > 5.0:
                regen_cmds['FR'] = max((-target_regen_power * 0.30) / omega_fr, -self.max_torque_front_peak)
                
        return regen_cmds