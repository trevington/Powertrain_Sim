import numpy as np

class VehicleDynamics:
    """
    A Kinematic Bicycle Model for vehicle simulation.
    This model is highly stable and ideal for powertrain and thermal analysis
    where tire slip and complex suspension dynamics are not the primary focus.
    """
    def __init__(self, mass, params=None):
        self.params = params if params is not None else {}
        self.mass = mass
        self.L = self.params.get('wheelbase', 1.535)      # Distance between front and rear axles (m)
        self.R = self.params.get('wheel_radius', 0.2032)   # Wheel radius (m)

        # State variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.v = 0.0
        self.f_cent = 0.0

        # Physical constants (matching main.py)
        self.rho = self.params.get('rho', 1.225)         # Air density
        self.cd = self.params.get('cd', 1.2)            # Drag coefficient
        self.area = self.params.get('frontal_area', 1.0) # Frontal area
        self.crr = self.params.get('crr', 0.015)         # Rolling resistance
        self.max_brake_accel_g = self.params.get('max_brake_accel_g', 1.5)
        self.g = 9.81

    def step(self, physics_inputs, steering_angle, dt):
        """
        Updates the vehicle state using kinematic bicycle equations.
        
        Args:
            physics_inputs (dict): Contains wheel torques and brake percentage.
            steering_angle (float): Steering angle in radians.
            dt (float): Timestep in seconds.
        """
        # 1. Calculate Propulsion Force from Torques
        total_torque = (physics_inputs.get('rear_axle_torque', 0) +
                        physics_inputs.get('fl_wheel_torque', 0) +
                        physics_inputs.get('fr_wheel_torque', 0))
        
        f_propulsion = total_torque / self.R

        # 2. Calculate Braking Force
        max_brake_force = self.mass * self.g * self.max_brake_accel_g
        f_braking = physics_inputs.get('mechanical_brake_pct', 0) * max_brake_force

        # 3. Calculate Resistive Forces
        f_drag = 0.5 * self.rho * self.cd * self.area * self.v**2
        f_rolling = self.mass * self.g * self.crr if self.v > 0.1 else 0.0

        # 4. Longitudinal Dynamics (F = ma)
        net_force = f_propulsion - f_braking - f_drag - f_rolling
        accel = net_force / self.mass

        # Update velocity
        self.v += accel * dt
        if self.v < 0:
            self.v = 0.0

        # 5. Kinematic Bicycle Motion Equations
        # Position updates
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        
        # Yaw update (Steering)
        # Steering angle is limited to prevent mathematical singularity
        # 23.5 degree rule-limited maximum steering angle
        delta = np.clip(steering_angle, -np.radians(23.5), np.radians(23.5))
        self.yaw_rate = (self.v / self.L) * np.tan(delta)
        self.yaw += self.yaw_rate * dt
        self.yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw)) # Normalize

        # 6. Prepare Outputs
        state = {'x': self.x, 'y': self.y, 'yaw': self.yaw, 'v': self.v, 'yaw_rate': self.yaw_rate}
        
        # Wheel speeds (rad/s) - assumed no slip in kinematic model
        w_speed = self.v / self.R
        wheel_speeds = {
            'rear': w_speed,
            'front_left': w_speed,
            'front_right': w_speed
        }

        return state, wheel_speeds