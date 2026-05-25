import numpy as np

class DriverModel:
    def __init__(self, track_waypoints, wheelbase=1.530, steering_ratio=4.5):
        """
        Closed-loop Driver Simulation Block for FSAE Forward-Facing Model.
        
        Parameters:
        -----------
        track_waypoints : np.ndarray
            An Nx3 array representing the track centerline: [[X, Y, V_target], ...]
        wheelbase : float
            Distance between front and rear axles in meters (FSAE average ~1.53m).
        steering_ratio : float
            Mechanical ratio of the steering rack (e.g., 4.5 degrees of steering wheel per 1 degree at the tire).
        """
        self.waypoints = track_waypoints  # Shape: (N, 3)
        self.L = wheelbase
        self.steering_ratio = steering_ratio
        
        # --- Longitudinal Speed PID Gains ---
        self.kp_v = 0.4
        self.ki_v = 0.05
        self.kd_v = 0.01
        self.v_error_integral = 0.0
        self.v_error_prior = 0.0
        
        # --- Lateral Pure Pursuit Parameters ---
        self.l_min = 2.5       # Minimum lookahead distance at low speeds (meters)
        self.k_speed = 0.15    # Lookahead scaling factor with velocity (seconds)
        self.max_steer_tire = np.radians(25.0)  # Max tire steer angle constraint

    def get_controls(self, x, y, yaw, v, dt):
        """
        Calculates driver control demands based on the vehicle's instantaneous spatial position.
        
        Parameters:
        -----------
        x, y : float -> Current absolute position coordinates of the chassis (meters)
        yaw : float  -> Current heading angle in radians (0 = global positive X-axis)
        v : float    -> Current forward velocity of the vehicle (m/s)
        dt : float   -> Simulation integration step time (seconds)
        
        Returns:
        --------
        throttle : float (0.0 to 1.0)
        brake : float (0.0 to 1.0)
        steering_wheel_angle : float (radians)
        """
        # 1. Find the index of the closest waypoint on the track
        distances = np.hypot(self.waypoints[:, 0] - x, self.waypoints[:, 1] - y)
        closest_idx = np.argmin(distances)
        
        # 2. LONGITUDINAL CONTROLLER (Speed Tracker)
        v_target = self.waypoints[closest_idx, 2]
        v_error = v_target - v
        
        # Resolve numerical integration for PID terms
        self.v_error_integral += v_error * dt
        v_derivative = (v_error - self.v_error_prior) / dt if dt > 0 else 0.0
        self.v_error_prior = v_error
        
        # Calculate single raw longitudinal control output
        u_long = (self.kp_v * v_error) + (self.ki_v * self.v_error_integral) + (self.kd_v * v_derivative)
        
        # Split single control value into distinct throttle/brake percentages
        if u_long >= 0.0:
            throttle = np.clip(u_long, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-u_long, 0.0, 1.0)
            
        # 3. LATERAL CONTROLLER (Pure Pursuit Steering)
        # Dynamically scale lookahead distance with speed to prevent oscillations
        l_fd = self.l_min + (self.k_speed * v)
        
        # Search forward through waypoints to find the target intercept point
        target_idx = closest_idx
        for i in range(closest_idx, closest_idx + len(self.waypoints)):
            idx = i % len(self.waypoints)  # Wrap around for closed loop tracks
            if distances[idx] >= l_fd:
                target_idx = idx
                break
                
        target_pt = self.waypoints[target_idx]
        
        # Transform the target point into the vehicle's relative local coordinate frame
        dx = target_pt[0] - x
        dy = target_pt[1] - y
        
        # Coordinate transformation rotation matrix to align with vehicle forward vector
        x_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        y_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        
        # Pure Pursuit geometric equation calculation: delta = atan(2 * L * y_local / l_fd^2)
        if l_fd > 0.1:
            steer_tire = np.arctan2(2.0 * self.L * y_local, l_fd**2)
        else:
            steer_tire = 0.0
            
        # Constrain tire steering to physical limits of front suspension linkages
        steer_tire = np.clip(steer_tire, -self.max_steer_tire, self.max_steer_tire)
        
        # Scale to match driver steering wheel input based on physical cockpit gearing
        steering_wheel_angle = steer_tire * self.steering_ratio
        
        return throttle, brake, steering_wheel_angle