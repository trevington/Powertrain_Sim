import numpy as np

class CoolingLoop:
    def __init__(self, radiator_area=0.12, fluid_mass=2.5, t_ambient=25.0):
        """
        Lumped-Capacitance Thermal Cooling Loop Model for an FSAE EV.
        Tracks coolant core temperatures under dynamic ram-air heat rejection.
        
        Parameters:
        -----------
        radiator_area : float -> Total frontal surface area of the radiator core (m^2)
        fluid_mass : float     -> Total mass of coolant fluid in the system (kg)
        t_ambient : float      -> Ambient atmospheric temperature (°C)
        """
        self.A_rad = radiator_area
        self.m_fluid = fluid_mass
        self.t_amb = t_ambient
        
        # Specific heat capacity of 50/50 Water/Glycol mix (~3500 J/kg*K)
        self.cp_fluid = 3500.0  
        
        # State variables
        self.fluid_temp = t_ambient  # Initial coolant temp equals ambient air temp (°C)
        
        # Radiator core heat dissipation characteristics 
        # Base air heat transfer scaling factors (derived from core wind-tunnel data)
        self.h_base = 15.0          # Natural convection baseline/fan contribution (W/m^2*K)
        self.h_ram_slope = 45.0     # Forced convection scaling factor per m/s of vehicle speed

    def step_thermal_network(self, motor_losses, inverter_losses, auxiliary_losses, v_x, dt):
        """
        Updates the loop fluid temperature by balancing powertrain heat inputs 
        against radiator convective atmospheric dissipation.
        
        Parameters:
        -----------
        motor_losses : float     -> Instantaneous thermal energy from motors (Watts)
        inverter_losses : float  -> Instantaneous thermal energy from inverter stages (Watts)
        auxiliary_losses : float -> Pump/HV system overhead power losses (Watts)
        v_x : float              -> Vehicle forward linear velocity (m/s)
        dt : float               -> Simulation integration step interval (seconds)
        
        Returns:
        --------
        fluid_temp : float -> New calculated coolant bulk temperature (°C)
        """
        # 1. Calculate Total Heat Energy Entering the Coolant Network
        q_in = motor_losses + inverter_losses + auxiliary_losses
        
        # 2. Calculate Dynamic Convective Heat Transfer Coefficient (h)
        # As vehicle speed increases, mass airflow through the radiator scales up non-linearly
        abs_vx = abs(v_x)
        h_eff = self.h_base + (self.h_ram_slope * abs_vx)
        
        # 3. Calculate Heat Energy Rejected by the Radiator Core
        # Q_out = h * A * delta_T
        q_out = h_eff * self.A_rad * (self.fluid_temp - self.t_amb)
        
        # 4. Integrate Fluid Temperature via First-Order Thermodynamic Step
        # q_net = q_in - q_out (Watts or Joules/sec)
        q_net = q_in - q_out
        
        # dT = (Q_net * dt) / (m * Cp)
        delta_temp = (q_net / (self.m_fluid * self.cp_fluid)) * dt
        self.fluid_temp += delta_temp
        
        # 5. Apply Physical Boundary Condition Guardrail
        # Coolant cannot drop below the ambient air temperature acting on the vehicle
        if self.fluid_temp < self.t_amb:
            self.fluid_temp = self.t_amb
            
        return self.fluid_temp