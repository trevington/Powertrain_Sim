import numpy as np

class PacejkaTire:
    def __init__(self):
        """
        Pacejka 'Magic Formula' Tire Model with Dynamic Relaxation Length.
        Coefficients represent typical lightweight Formula SAE racing tires.
        """
        # --- Longitudinal Coefficients (Pure Braking/Acceleration) ---
        self.B_x = 12.0
        self.C_x = 1.60
        self.E_x = -0.40
        self.mu_x = 1.75  # Peak longitudinal friction coefficient
        
        # --- Lateral Coefficients (Pure Cornering) ---
        self.B_y = 10.0
        self.C_y = 1.30
        self.E_y = -0.60
        self.mu_y = 1.55  # Peak lateral friction coefficient
        
        # --- Structural Relaxation Length Parameters ---
        # Typical 18-inch Hoosier FSAE tires distort over ~0.10 to 0.15 meters
        self.sigma_x = 0.12  # Longitudinal relaxation length (meters)
        self.sigma_y = 0.15  # Lateral relaxation length (meters)
        
        # Minimum transport speed to avoid zero-velocity freeze at a standstill
        self.v_min = 0.5     # m/s
        
        # Dynamic State Tracking Dictionaries (keyed by wheel identifier string)
        self.kappa_eff = {}
        self.alpha_eff = {}

    def step_relaxation_dynamics(self, wheel_id, kappa_kin, alpha_kin, v_x, F_z, dt):
        """
        Updates the structural tire carcass deformation states and calculates 
        stable, non-stiff combined patch forces.
        
        Parameters:
        -----------
        wheel_id : str       -> Unique string naming the wheel (e.g., 'FL', 'RL')
        kappa_kin : float    -> Kinematic slip ratio calculated from hub/chassis speeds
        alpha_kin : float    -> Kinematic slip angle in radians from steering/yaw geometry
        v_x : float          -> Linear forward velocity of the wheel hub (m/s)
        F_z : float          -> Vertical load acting on the tire patch (Newtons)
        dt : float           -> Integration time step (seconds)
        """
        # 1. Lazy-initialize states for this specific wheel if they don't exist yet
        if wheel_id not in self.kappa_eff:
            self.kappa_eff[wheel_id] = 0.0
            self.alpha_eff[wheel_id] = 0.0
            
        # 2. Compute Transport Velocity
        # The faster the tire rolls, the quicker it relaxes. At a complete stop,
        # we enforce a floor (v_min) so the tire can still relax statically.
        v_transport = max(abs(v_x), self.v_min)
        
        # 3. Exact Analytical Exponential Integration Step
        # This replaces: state += (dv/dt) * dt
        # It guarantees the state smoothly converges on target without ever overshooting.
        decay_x = np.exp(-v_transport / self.sigma_x * dt)
        self.kappa_eff[wheel_id] = kappa_kin + (self.kappa_eff[wheel_id] - kappa_kin) * decay_x
        
        decay_y = np.exp(-v_transport / self.sigma_y * dt)
        self.alpha_eff[wheel_id] = alpha_kin + (self.alpha_eff[wheel_id] - alpha_kin) * decay_y
        
        # 4. Feed the lagged, stable states into the standard Pacejka core
        return self.calculate_forces(self.kappa_eff[wheel_id], self.alpha_eff[wheel_id], F_z)

    def calculate_forces(self, kappa, alpha, F_z):
        """
        Calculates combined tire forces using effective (relaxed) slip states.
        """
        # Avoid division by zero issues or calculations if the wheel lifts off the ground
        if F_z <= 10.0:
            return 0.0, 0.0

        # 1. PURE LONGITUDINAL FORCE (Fx_pure)
        D_x = self.mu_x * F_z
        Fx_pure = D_x * np.sin(self.C_x * np.arctan(self.B_x * kappa - self.E_x * (self.B_x * kappa - np.arctan(self.B_x * kappa))))

        # 2. PURE LATERAL FORCE (Fy_pure)
        D_y = self.mu_y * F_z
        Fy_pure = D_y * np.sin(self.C_y * np.arctan(self.B_y * alpha - self.E_y * (self.B_y * alpha - np.arctan(self.B_y * alpha))))

        # 3. COMBINED SLIP TRANSLATION (The Friction Ellipse)
        max_Fx = D_x
        max_Fy = D_y

        if Fx_pure == 0.0 and Fy_pure == 0.0:
            return 0.0, 0.0
            
        combined_ratio = np.sqrt((Fx_pure / max_Fx)**2 + (Fy_pure / max_Fy)**2)

        if combined_ratio > 1.0:
            Fx_combined = Fx_pure / combined_ratio
            Fy_combined = Fy_pure / combined_ratio
        else:
            alpha_fraction = np.abs(Fy_pure / (max_Fy + 1e-6))
            kappa_fraction = np.abs(Fx_pure / (max_Fx + 1e-6))
            
            Fx_combined = Fx_pure * np.sqrt(max(0.0, 1.0 - alpha_fraction**2))
            Fy_combined = Fy_pure * np.sqrt(max(0.0, 1.0 - kappa_fraction**2))

        return Fx_combined, Fy_combined