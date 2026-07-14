import numpy as np
from scipy.interpolate import RegularGridInterpolator

class Motor:
    def __init__(self, thermal_mass=12.0, max_rpm=7000.0):
        """
        High-Fidelity Electric Motor Plant Model for FSAE Transient Simulation.
        Default values approximate an EMRAX 208 High Voltage Medium-Cooling motor.
        
        Parameters:
        -----------
        inertia : float -> Rotational inertia of the internal rotor (kg*m^2)
        thermal_mass : float -> Specific heat capacity lump of the active stator/core (kJ/K or mass*Cp)
        peak_torque : float -> Maximum continuous/peak stall torque (Nm)
        peak_power_kw : float -> Maximum continuous/peak electrical-mechanical power capacity (kW)
        max_rpm : float -> Mechanical redline ceiling of the rotor
        """
        # --- Mechanical Properties ---
        self.J = 0.01569 # Rotor inertia
        self.max_torque_peak = 150.0 # 150 Nm peak torque
        self.max_power = 80000.0  # 80 kw FSAE limit
        self.max_omega = (7000 * 2 * np.pi) / 60.0  # Convert 7,000 RPM to rad/s
        
        # --- Electrical/Loss Coefficients (Approximated for PMAC Motor) ---
        self.R_stator = 0.01227 * 2      # Internal phase-to-phase resistance approximation (Ohms) (Value at 25°C)
        self.kt = 0.62             # Torque constant (Nm / A_rms)
        
        # efficiency grid (speed, torque, efficiency)
        efficiency_map = np.array([[100,20,.86],[200,20,.86],[300,20,.86],[400,20,.86],[100,40,.90],[200,40,.94],[300]])
        
        # --- Thermal Properties ---
        # Thermal mass (C_p * mass) dictates how fast losses turn into temperature spikes
        # For ~12kg of copper/aluminum/iron mixture, C_p_effective is approx 450 J/kg*K -> ~5400 J/K
        self.C_p = thermal_mass * 450.0  
        self.temp = 25.0           # Instantaneous core temperature (°C)
        self.R_th_stator_to_case = 0.08  # Thermal resistance to coolant jacket (°C/W)
        
        # --- Transient Dynamic Properties ---
        self.T_actual = 0.0        # Current actual output torque (Nm)
        self.tau = 0.015           # Electrical torque response time constant (seconds, ~15ms)

    def get_max_torque_capacity(self, omega):
        """Calculates the physical maximum torque allowed at a given rotational speed."""
        abs_omega = max(0.1, abs(omega))
        
        if abs_omega >= self.max_omega:
            return 0.0
            
        # Constant Power Curve Limit: Torque = Power / Omega
        power_limit_torque = self.max_power / abs_omega
        
        # The true ceiling is the minimum of the peak physical structural torque or the power envelope
        return min(self.max_torque_peak, power_limit_torque)

    def update_torque(self, T_cmd, dt, omega=0.0):
        """
        Steps the internal inductive lag dynamics forward and clamps torque to the physical envelope.
        
        Parameters:
        -----------
        T_cmd : float -> Commanded torque from VCU (Nm)
        dt : float -> Simulation time step (seconds)
        omega : float -> Current rotor mechanical speed (rad/s)
        """
        # Determine the maximum available envelope at this precise RPM
        max_available_torque = self.get_max_torque_capacity(omega)
        
        # Handle directionality (T_cmd can be negative for regenerative braking)
        clamped_T_cmd = np.clip(T_cmd, -max_available_torque, max_available_torque)
        
        # First-order explicit integration filter representing inductive lag: 
        # dT/dt = (T_command - T_actual) / tau
        dT = ((clamped_T_cmd - self.T_actual) / self.tau) * dt
        self.T_actual += dT
        
        return self.T_actual

    def calculate_losses(self, torque, omega):
        """
        Calculates total power dissipation (heat losses) using a 2D efficiency map lookup.
        
        Returns:
        --------
        p_loss : float -> Total thermal energy generated in Watts
        """
        abs_omega = abs(omega)
        abs_torque = abs(torque)
        p_mech = abs_torque * abs_omega
        
        # Efficiency maps are numerically unstable at zero speed (efficiency = P_out / P_in)
        # We handle low-speed losses using the analytical copper loss as a fallback
        if abs_omega < 5.0:
            i_rms = abs_torque / self.kt
            return (i_rms**2) * self.R_stator
        
        # Perform 2D interpolation for motor efficiency
        # Query requires a point in the shape (1, 2)
        efficiency = float(self.eff_interp(np.array([abs_omega, abs_torque])))
        
        # Total input power is P_mech / efficiency; losses are the difference
        eff_clamped = np.clip(efficiency, 0.1, 0.99)
        p_loss = p_mech * (1.0 / eff_clamped - 1.0)
        
        return p_loss

    def step_thermal_node(self, p_loss, coolant_temp, dt):
        """
        Explicitly integrates motor core temperature based on internal losses 
        and heat rejection to the cooling loop fluid jacket.
        """
        # Heat transfer rate out of the motor: Q_out = (T_motor - T_coolant) / R_th
        q_rejected = (self.temp - coolant_temp) / self.R_th_stator_to_case
        
        # Net thermal energy trapped in the core mass
        q_net = p_loss - q_rejected
        
        # dT/dt = Q_net / Heat_Capacity
        self.temp += (q_net / self.C_p) * dt
        
        return self.temp