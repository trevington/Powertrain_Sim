class ChassisLoadModel:
    def __init__(self, mass=292.0, wheelbase=1.53, track_front=1.22, track_rear=1.18, 
                 cg_height=0.28, weight_dist_front=0.45, roll_stiffness_front=0.55,
                 cla=2.8, cop_front=0.42):
        """
        Calculates dynamic 4-wheel vertical loads (Fz) for an FSAE car.
        """
        self.m = mass
        self.g = 9.81
        self.L = wheelbase
        self.t_f = track_front
        self.t_r = track_rear
        self.h_cg = cg_height
        self.r_eff = 0.2032  
        
        # Calculate CG position relative to axles from static weight distribution percentage
        self.b = weight_dist_front * self.L  # Distance from CG to rear axle
        self.a = self.L - self.b             # Distance from CG to front axle
        
        # --- FIX: Alias these variables to match the kinematics formulas ---
        self.l_f = self.a                    # Distance from CG to front axle
        self.l_r = self.b                    # Distance from CG to rear axle
        
        # Suspension Roll Distribution
        self.k_roll_f = roll_stiffness_front # Fraction of lateral load carried by front axle
        
        # Aero Properties
        self.rho = 1.225        # Air density at track level (kg/m^3)
        self.ClA = cla          # Lift coefficient * frontal area (negative lift)
        self.cop_f = cop_front  # Percentage of aero downforce acting on front axle

    def calculate_wheel_loads(self, a_x, a_y, v_car):
        """
        Computes the instantaneous vertical forces (Fz) acting on all four tires.
        """
        # 1. Base Static Weights (Newtons per wheel)
        static_f = (self.m * self.g * self.b) / (2.0 * self.L)
        static_r = (self.m * self.g * self.a) / (2.0 * self.L)
        
        # 2. Longitudinal Weight Transfer (Shared equally across axle pairs)
        delta_Fz_long = (self.m * a_x * self.h_cg) / self.L
        long_f = -delta_Fz_long / 2.0
        long_r =  delta_Fz_long / 2.0
        
        # 3. Lateral Weight Transfer (Calculated per axle via roll stiffness distribution)
        delta_Fz_lat_f = ((self.m * a_y * self.h_cg) / self.t_f) * self.k_roll_f
        delta_Fz_lat_r = ((self.m * a_y * self.h_cg) / self.t_r) * (1.0 - self.k_roll_f)
        
        # 4. Aerodynamic Downforce Generation
        f_down_total = 0.5 * self.rho * (v_car ** 2) * self.ClA
        aero_f = (f_down_total * self.cop_f) / 2.0
        aero_r = (f_down_total * (1.0 - self.cop_f)) / 2.0
        
        # 5. Aggregate All Component Vectors
        loads = {
            'FL': static_f + long_f - delta_Fz_lat_f + aero_f,
            'FR': static_f + long_f + delta_Fz_lat_f + aero_f,
            'RL': static_r + long_r - delta_Fz_lat_r + aero_r,
            'RR': static_r + long_r + delta_Fz_lat_r + aero_r
        }
        
        # Enforce physical ceiling constraint: a tire cannot pull upward on the asphalt
        for wheel in loads:
            if loads[wheel] < 0.0:
                loads[wheel] = 0.0
                
        return loads