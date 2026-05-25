import numpy as np
from cooling_sim import CoolingLoop
from driver import DriverModel
from motor_sim import Motor
from vehicle_controller import VehicleController
from chassis_sim import ChassisLoadModel
from tire_sim import PacejkaTire

class VehicleSimulator:
    def __init__(self, track_waypoints, layout="RWD"):
        """
        Master Orchestration Class for a Forward-Facing EV Simulation.
        Coordinates driver inputs, control execution, and multi-body plant physics.
        """
        self.waypoints = track_waypoints
        self.layout = layout.upper()

        # 1. Instantiate the Control and Driver Blocks
        self.driver = DriverModel(track_waypoints=track_waypoints)
        self.vcu = VehicleController(layout=self.layout)
        
        # 2. Instantiate the Physical Plant Subsystems
        self.chassis = ChassisLoadModel()
        self.tire = PacejkaTire()
        self.cooling = CoolingLoop(radiator_area=0.12, fluid_mass=2.5)
        
        # 3. Dynamic Powertrain Allocation (Updated to match new Motor parameters)
        self.motors = {}
        if self.layout == "RWD":
            # Single inboard rear motor (e.g., Emrax 208 parameters)
            self.motors['Rear'] = Motor(inertia=0.045, thermal_mass=12.0, peak_torque=210.0, peak_power_kw=80.0)
        elif self.layout == "3-MOTOR":
            # Dual front outboard hub motors + single rear inboard motor
            self.motors['FL'] = Motor(inertia=0.012, thermal_mass=4.5, peak_torque=80.0, peak_power_kw=32.0)
            self.motors['FR'] = Motor(inertia=0.012, thermal_mass=4.5, peak_torque=80.0, peak_power_kw=32.0)
            self.motors['Rear'] = Motor(inertia=0.045, thermal_mass=12.0, peak_torque=210.0, peak_power_kw=80.0)

        # --- Master Vehicle State Vectors (Initialized at rest) ---
        self.x = 0.0          # Global X position (meters)
        self.y = 0.0          # Global Y position (meters)
        self.yaw = 0.0        # Heading angle (radians)
        self.v_x = 0.01       # Forward velocity (m/s) -> start slightly above zero to avoid divide-by-zero
        self.v_y = 0.0        # Lateral velocity (m/s)
        self.yaw_rate = 0.0   # Angular rotation velocity (rad/s)

        # --- Actuator & Control States ---
        self.steering_angle = 0.0  # Initialize steering angle in radians
        self.max_steer = np.radians(23.0)  # Typical FSAE rack limit (~23 degrees)

        # --- Wheel States ---
        self.omega = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        
        # Individual wheel speed velocities (rad/s)
        self.wheel_speeds = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        
        # Fixed Mass/Inertia Properties for 3-DOF Equations of Motion
        self.I_z = 120.0       # Vehicle polar moment of inertia around Z-axis (kg*m^2)
        self.J_wheel = 0.18    # Rotational inertia of a single tire/rim/hub assembly (kg*m^2)
        self.thermal_derating = 1.0

    def step(self, dt):
        """
        Executes a single forward integration step across all subsystems.
        """
        # 1. INTERNAL DRIVER MODEL (Pure Pursuit Path Following)
        lookahead_distance = max(2.0, self.v_x * 0.5)

        target_wp = None
        for wp in self.waypoints:
            dx = wp[0] - self.x
            dy = wp[1] - self.y
            dist = np.sqrt(dx**2 + dy**2)
            if dist >= lookahead_distance:
                target_wp = wp
                break
        
        if target_wp is None:
            target_wp = self.waypoints[-1]

        alpha_track = np.arctan2(target_wp[1] - self.y, target_wp[0] - self.x) - self.yaw
        alpha_track = (alpha_track + np.pi) % (2 * np.pi) - np.pi

        wheelbase = self.chassis.l_f + self.chassis.l_r
        target_steer = np.arctan2(2.0 * wheelbase * np.sin(alpha_track), lookahead_distance)

        steer_error = target_steer - self.steering_angle
        max_steer_rate = 5.0  
        self.steering_angle += np.clip(steer_error, -max_steer_rate * dt, max_steer_rate * dt)
        self.steering_angle = np.clip(self.steering_angle, -self.max_steer, self.max_steer)

        # --- 1. SOLVE ACCELERATIONS FROM PRIOR STATE ---
        a_x_estimated = (self.v_x / dt) if self.v_x < 0.5 else 0.0  
        a_y_estimated = self.v_x * self.yaw_rate
        
        # --- 2. EVALUATE PASSIVE CHASSIS LOADS ---
        Fz_dict = self.chassis.calculate_wheel_loads(a_x_estimated, a_y_estimated, self.v_x)

        # --- 3. EXECUTE CLOSED-LOOP DRIVER INPUTS ---
        throttle, brake, steer_wheel = self.driver.get_controls(
            self.x, self.y, self.yaw, self.v_x, dt
        )
        steer_tire = steer_wheel / self.driver.steering_ratio

        # --- 4. EXECUTE VCU ACTUATOR ALLOCATION ---
        motor_cmds, brake_cmds = self.vcu.allocate_actuators(
            throttle, brake, steer_tire, self.v_x, self.wheel_speeds, self.thermal_derating
        )

        # --- 5. SOLVE POWERTRAIN AND CALCULATE ENERGY LOSSES ---
        actual_motor_torques = {'FL': 0.0, 'FR': 0.0, 'Rear': 0.0}
        motor_losses = {'FL': 0.0, 'FR': 0.0, 'Rear': 0.0}
        total_p_loss = 0.0
        
        for key in self.motors:
            omega_shaft = self.wheel_speeds['RL'] if key == 'Rear' else self.wheel_speeds[key]
            actual_motor_torques[key] = self.motors[key].update_torque(motor_cmds[key], dt, omega=omega_shaft)
            p_loss_motor = self.motors[key].calculate_losses(actual_motor_torques[key], omega_shaft)
            motor_losses[key] = p_loss_motor
            total_p_loss += p_loss_motor

        # --- 6. CALCULATE CONTACT PATCH TIRE SLIPS & FORCES (WITH RELAXATION DYNAMICS) ---
        Fx_total = 0.0
        Fy_total = 0.0
        Mz_total = 0.0
        
        for wheel in ['FL', 'FR', 'RL', 'RR']:
            # A) Calculate the raw kinematic inputs from the chassis speeds
            kappa_kin = self.calculate_kinematic_long_slip(wheel)
            alpha_kin = self.calculate_kinematic_lat_slip(wheel)

            # B) Corrected: Fetch structural forces directly from the already calculated loads
            F_z = Fz_dict[wheel]
            v_x_wheel, _ = self._get_wheel_hub_velocities(wheel)

            # C) Pass through relaxation filter to get stable, non-stiff contact forces
            Fx_tire, Fy_tire = self.tire.step_relaxation_dynamics(
                wheel_id=wheel,
                kappa_kin=kappa_kin,
                alpha_kin=alpha_kin,
                v_x=v_x_wheel,
                F_z=F_z,
                dt=dt
            )
            
            # D) Rotate front tire forces into the chassis frame coordinate system
            is_front = 1 if wheel in ['FL', 'FR'] else -1
            dist_to_axle = self.chassis.a if is_front == 1 else self.chassis.b
            
            if is_front == 1:
                Fx_chassis = Fx_tire * np.cos(steer_tire) - Fy_tire * np.sin(steer_tire)
                Fy_chassis = Fx_tire * np.sin(steer_tire) + Fy_tire * np.cos(steer_tire)
            else:
                Fx_chassis = Fx_tire
                Fy_chassis = Fy_tire
                
            # --- 7. RESOLVE WHEEL ROTATIONAL SPEEDS & TORQUE BALANCE ---
            t_motor_hub = 0.0
            if self.layout == "RWD" and wheel in ['RL', 'RR']:
                t_motor_hub = actual_motor_torques['Rear'] / 2.0  
            elif self.layout == "3-MOTOR":
                t_motor_hub = actual_motor_torques['Rear'] / 2.0 if wheel in ['RL', 'RR'] else actual_motor_torques[wheel]
                
            T_net_wheel = t_motor_hub - brake_cmds[wheel] - (Fx_tire * self.chassis.r_eff)
            wheel_accel = T_net_wheel / self.J_wheel
            self.wheel_speeds[wheel] += wheel_accel * dt
            
            # Aggregate total forces acting on global body structure
            Fx_total += Fx_chassis
            Fy_total += Fy_chassis
            
            # Moment calculation mapping for yaw rotation changes
            dist_lat = (self.chassis.t_f / 2.0) if is_front == 1 else (self.chassis.t_r / 2.0)
            side_sign = 1 if wheel in ['FR', 'RR'] else -1
            Mz_total += (Fy_chassis * dist_to_axle * is_front) + (Fx_chassis * dist_lat * side_sign)

        # --- 8. 3-DOF PLANAR CHASSIS STATE INTEGRATION ---
        F_drag = 0.5 * self.chassis.rho * (self.v_x**2) * self.chassis.ClA * 0.35 
        
        dv_x = (Fx_total - F_drag) / self.chassis.m + self.v_y * self.yaw_rate
        dv_y = Fy_total / self.chassis.m - self.v_x * self.yaw_rate
        dyaw_rate = Mz_total / self.I_z
        
        self.v_x += dv_x * dt
        self.v_y += dv_y * dt
        self.yaw_rate += dyaw_rate * dt
        
        dx = self.v_x * np.cos(self.yaw) - self.v_y * np.sin(self.yaw)
        dy = self.v_x * np.sin(self.yaw) + self.v_y * np.cos(self.yaw)
        
        self.x += dx * dt
        self.y += dy * dt
        self.yaw += self.yaw_rate * dt

        # --- 9. TRANSIENT COOLING & MOTOR THERMAL NODE STEP ---
        coolant_temp = self.cooling.step_thermal_network(total_p_loss, 0.0, 0.0, self.v_x, dt)
        
        target_derating = 1.0
        for key in self.motors:
            motor_temp = self.motors[key].step_thermal_node(motor_losses[key], coolant_temp, dt)
            if motor_temp > 105.0:
                derate_factor = max(0.1, 1.0 - (motor_temp - 105.0) * 0.03)
                if derate_factor < target_derating:
                    target_derating = derate_factor
                    
        self.thermal_derating += (target_derating - self.thermal_derating) * (dt / 0.1)

    def _get_wheel_hub_velocities(self, wheel):
        """
        Helper method to transform vehicle CG velocities (v_x, v_y, yaw_rate)
        into the local longitudinal and lateral velocity vectors at the wheel contact patch.
        """
        # 1. Fetch geometric parameters from Chassis Load Model
        # lf = distance from CG to front axle (m)
        # lr = distance from CG to rear axle (m)
        # tf = front track width (m), tr = rear track width (m)
        lf = getattr(self.chassis, 'l_f', 1.0)
        lr = getattr(self.chassis, 'l_r', 0.6)
        tf = getattr(self.chassis, 't_f', 1.2)
        tr = getattr(self.chassis, 't_r', 1.2)
        
        # 2. Rigid body velocity translation from CG to the specific Wheel Hub (Chassis Frame)
        if wheel == 'FL':
            v_x_hub = self.v_x - self.yaw_rate * (tf / 2.0)
            v_y_hub = self.v_y + self.yaw_rate * lf
            delta = self.steering_angle  # Front wheel steers
        elif wheel == 'FR':
            v_x_hub = self.v_x + self.yaw_rate * (tf / 2.0)
            v_y_hub = self.v_y + self.yaw_rate * lf
            delta = self.steering_angle  # Front wheel steers
        elif wheel == 'RL':
            v_x_hub = self.v_x - self.yaw_rate * (tr / 2.0)
            v_y_hub = self.v_y - self.yaw_rate * lr
            delta = 0.0                  # Rear wheel does not steer
        elif wheel == 'RR':
            v_x_hub = self.v_x + self.yaw_rate * (tr / 2.0)
            v_y_hub = self.v_y - self.yaw_rate * lr
            delta = 0.0                  # Rear wheel does not steer
        else:
            raise ValueError(f"Unknown wheel identifier: {wheel}")

        # 3. Rotate chassis-frame velocities into the Wheel Orientation Frame
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        v_x_wheel = (v_x_hub * cos_d) + (v_y_hub * sin_d)
        v_y_wheel = (-v_x_hub * sin_d) + (v_y_hub * cos_d)
        
        return v_x_wheel, v_y_wheel

    def calculate_kinematic_long_slip(self, wheel):
        """
        Calculates raw kinematic longitudinal slip (kappa) for a given wheel.
        Positive for acceleration (wheel spinning faster than ground), 
        negative for braking (wheel locking up).
        """
        v_x_wheel, _ = self._get_wheel_hub_velocities(wheel)
        
        # Fetch current angular velocity (rad/s) and effective rolling radius
        omega_wheel = self.omega[wheel] 
        r_eff = self.chassis.r_eff
        
        # Small epsilon to safeguard against division by zero at a standstill
        epsilon = 0.1 
        
        # SAE J670/ISO definition normalized by forward velocity
        kappa_kin = (omega_wheel * r_eff - v_x_wheel) / (abs(v_x_wheel) + epsilon)
        
        # Cap slip ratio boundaries to prevent unphysical kinematic infinity
        return np.clip(kappa_kin, -2.0, 2.0)

    def calculate_kinematic_lat_slip(self, wheel):
        """
        Calculates raw kinematic slip angle (alpha) in radians for a given wheel.
        Represents the angle between the wheel's orientation vector and its 
        actual direction of travel.
        """
        v_x_wheel, v_y_wheel = self._get_wheel_hub_velocities(wheel)
        
        # Small epsilon to safeguard against division by zero at a standstill
        epsilon = 0.1
        
        # Alpha = arctan(v_y / |v_x|)
        alpha_kin = np.arctan2(v_y_wheel, abs(v_x_wheel) + epsilon)
        
        return alpha_kin