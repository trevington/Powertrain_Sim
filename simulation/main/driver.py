import numpy as np

class PathCommander:
    def __init__(self, track_waypoints=None, mode='Endurance', params=None):
        """
        Initializes the Path Commander (Driver).
        
        :param track_waypoints: List of (x, y, target_v, curvature) tuples
        :param mode: Simulation mode (Endurance, Acceleration, Autocross, Skidpad)
        :param params: Dictionary of tuning parameters
        """
        self.params = params if params is not None else {}
        self.waypoints = track_waypoints if track_waypoints is not None else []
        self.mode = mode
        self.current_waypoint_idx = 0
        
        # Look-ahead tuning
        self.look_ahead_indices = self.params.get('look_ahead_indices', 1)
        self.look_ahead_v_scale = self.params.get('look_ahead_v_scale', 0.25)
        
        # Longitudinal Control Gains (Shared Defaults)
        kp_default = self.params.get('kp_v', 0.187480)
        ki_default = self.params.get('ki_v', 0.022519)
        kd_default = self.params.get('kd_v', 0.014126)

        # Acceleration PID Gains
        self.kp_accel = self.params.get('kp_accel', kp_default)
        self.ki_accel = self.params.get('ki_accel', ki_default)
        self.kd_accel = self.params.get('kd_accel', kd_default)

        # Brake PID Gains
        self.kp_brake = self.params.get('kp_brake', kp_default)
        self.ki_brake = self.params.get('ki_brake', ki_default)
        self.kd_brake = self.params.get('kd_brake', kd_default)

        self.v_int_accel = 0.0
        self.prev_err_accel = 0.0
        self.v_int_brake = 0.0
        self.prev_err_brake = 0.0

        self.v_int_limit = self.params.get('v_int_limit', 0.5)
        
        self.v_deadband = self.params.get('v_deadband', 0.15)
        self.steer_deadband = self.params.get('steer_deadband', np.radians(0.1))
        self.throttle_alpha = self.params.get('throttle_alpha', 0.2)

        self.prev_throttle = 0.0  # For output smoothing
        self.prev_steering = 0.0  # For output smoothing

        # Stanley Controller Gain (Lateral Control)
        self.k_stanley = self.params.get('k_stanley', 1.4)
        self.wheelbase = self.params.get('wheelbase', 1.53)

    def get_commands(self, vehicle_state, dt):
        """
        Calculates throttle and steering based on vehicle state.
        
        :param vehicle_state: Dictionary containing x, y, yaw, v, etc.
        :param dt: Time step
        :return: (throttle_pct, steering_angle)
        """
        if self.mode == 'Acceleration':
            return self._handle_acceleration(vehicle_state)
        
        # 1. Update Waypoint Tracking
        self._update_waypoint(vehicle_state)
        
        # 2. Longitudinal Control (Speed)
        target_v = self.waypoints[min(self.current_waypoint_idx, len(self.waypoints)-1)][2]
        
        throttle_pct = self._calculate_longitudinal(vehicle_state['v'], target_v, dt)

        # 3. Lateral Control (Steering)
        steering_angle = self._calculate_steering(vehicle_state)
        
        return throttle_pct, steering_angle

    def reset(self):
        """Resets driver internal state variables for a clean simulation start."""
        self.current_waypoint_idx = 0
        self.v_int_accel = 0.0
        self.prev_err_accel = 0.0
        self.v_int_brake = 0.0
        self.prev_err_brake = 0.0
        self.prev_throttle = 0.0
        self.prev_steering = 0.0

    def _update_waypoint(self, state):
        """Finds the nearest/next waypoint to target."""
        if not self.waypoints:
            return
            
        # 1. Find the nearest waypoint to the current position
        curr_pos = np.array([state['x'], state['y']])
        search_range = int(state['v'] * .5) + 1 
        
        min_dist = float('inf')
        closest_idx = self.current_waypoint_idx

        # Localized search window to prevent jumping to distant parts of the track
        for i in range(search_range):
            idx = (self.current_waypoint_idx + i) % len(self.waypoints)
            wp_x, wp_y = self.waypoints[idx][0], self.waypoints[idx][1]
            dist = np.hypot(state['x'] - wp_x, state['y'] - wp_y)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        self.current_waypoint_idx = closest_idx

    def _calculate_longitudinal(self, current_v, target_v, dt):
        """Separate PIDs for acceleration and braking."""
        error = target_v - current_v

        # 1. Apply Deadband: Ignore small errors to stop the PID from "shivering"
        if abs(error) < self.v_deadband:
            error = 0.0

        if error >= 0:
            # Acceleration Controller
            self.v_int_accel += error * dt
            self.v_int_accel = np.clip(self.v_int_accel, -self.v_int_limit, self.v_int_limit)
            deriv = (error - self.prev_err_accel) / dt if dt > 0 else 0
            output = (self.kp_accel * error) + (self.ki_accel * self.v_int_accel) + (self.kd_accel * deriv)
            
            self.prev_err_accel = error
            # Reset Braking integrator to prevent integrator windup on transition
            self.v_int_brake = 0.0
            self.prev_err_brake = 0.0
        else:
            # Braking Controller
            self.v_int_brake += error * dt
            self.v_int_brake = np.clip(self.v_int_brake, -self.v_int_limit, self.v_int_limit)
            deriv = (error - self.prev_err_brake) / dt if dt > 0 else 0
            output = (self.kp_brake * error) + (self.ki_brake * self.v_int_brake) + (self.kd_brake * deriv)
            
            self.prev_err_brake = error
            # Reset Acceleration integrator to prevent integrator windup on transition
            self.v_int_accel = 0.0
            self.prev_err_accel = 0.0

        self.prev_v_error = error
        
        # Clip to -1.0 (full brake) to 1.0 (full throttle)
        raw_throttle = np.clip(output, -1.0, 1.0)
        
        # 2. Alpha Filter (Low-Pass): Smooths output to prevent rapid chatter.
        # This significantly reduces "chatter" between throttle and braking
        alpha = 0.1
        self.prev_throttle = (alpha * raw_throttle) + ((1.0 - alpha) * self.prev_throttle)
        
        return self.prev_throttle

    def _calculate_steering(self, state):
        """
        Stanley Controller implementation for lateral steering.
        """
        if not self.waypoints:
            return 0.0

        # Dynamic look-ahead: Look further ahead as velocity increases
        # Reduced scaling to prevent "orbiting" or cutting corners too early
        look_ahead = int(self.look_ahead_indices + (state['v'] * self.look_ahead_v_scale))
        target_idx = (self.current_waypoint_idx + look_ahead) % len(self.waypoints)
        
        # Current target for heading (Look-ahead point)
        target_wp = self.waypoints[target_idx]
        prev_target_wp = self.waypoints[(target_idx - 1) % len(self.waypoints)]
        
        # Point for cross-track error (Nearest point)
        closest_wp = self.waypoints[self.current_waypoint_idx]
        curvature = closest_wp[3] if len(closest_wp) > 3 else 0.0
        
        # Helper to get path heading between two points
        def get_path_yaw(idx1, idx2, fallback_yaw):
            p1 = self.waypoints[idx1]
            p2 = self.waypoints[idx2]
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            if np.hypot(dx, dy) < 1e-3:
                return fallback_yaw
            return np.arctan2(dy, dx)

        # Local path orientation at the car's current position for accurate CTE projection
        local_path_yaw = get_path_yaw(self.current_waypoint_idx, (self.current_waypoint_idx - 1) % len(self.waypoints), state['yaw'])
        path_yaw = get_path_yaw(target_idx, (target_idx - 1) % len(self.waypoints), state['yaw'])
        heading_error = (path_yaw - state['yaw'] + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Signed Cross-track error calculated using LOCAL path tangent
        dx = state['x'] - closest_wp[0]
        dy = state['y'] - closest_wp[1]
        cross_track_error = -dx * np.sin(local_path_yaw) + dy * np.cos(local_path_yaw)
        
        # 3. Feedback Steering (Stanley)
        v = state['v']
        k_s = self.k_stanley
            
        steering_fb = heading_error - np.arctan2(cross_track_error, k_s + v)
        
        # 4. Feed-Forward Steering (Ackermann geometry based)
        # delta_ff = atan(Wheelbase / Radius) = atan(Wheelbase * Curvature)
        steering_ff = np.arctan(self.wheelbase * curvature)
        
        # Dynamic Feed-Forward Weighting:
        # If the car is far off-track (e.g. > 1.5m), ignore curvature and focus 
        # entirely on the feedback recovery (CTE and Heading).
        ff_weight = np.clip(1.0 - abs(cross_track_error) / 1.5, 0.0, 1.0)

        # Total Steering = Proactive (FF) + Reactive (FB)
        combined_steering = steering_fb + (steering_ff * ff_weight)

        # Apply steering deadband to reduce "hunting" on long straights
        if abs(combined_steering) < self.steer_deadband:
            combined_steering = 0.0
        
        raw_steering = np.clip(combined_steering, -np.radians(25), np.radians(25))

        # More responsive filter (Alpha = 0.6) to reduce phase lag in corners
        alpha_steer = 0.6
        smoothed_steering = (alpha_steer * raw_steering) + ((1.0 - alpha_steer) * self.prev_steering)
        self.prev_steering = smoothed_steering
        
        return smoothed_steering

    def _handle_acceleration(self, state):
        """Logic for straight-line acceleration event."""
            
        return 1.0, 0.0

if __name__ == "__main__":
    # Example Usage / Unit Test
    driver = PathCommander(mode='Acceleration')
    dummy_state = {'x': 0, 'y': 0, 'yaw': 0, 'v': 10}
    print(f"Accel commands: {driver.get_commands(dummy_state, 0.1)}")