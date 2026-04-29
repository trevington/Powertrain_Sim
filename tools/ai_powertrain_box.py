import math

class EVPowertrainSimulator:
    def __init__(self):
        # --- ESTIMATED CONSTANTS (To be updated with your FSAE data) ---
        
        # Motor & Drivetrain (Current Control Mode)
        self.MAX_PHASE_CURRENT_A = 199.0        # Max phase current from inverter (Amps)
        self.TORQUE_CONSTANT_KT = 0.62          # Motor Torque Constant (Nm / Amp)
        self.MOTOR_PHASE_RESISTANCE = 0.01227   # Stator resistance in Ohms (for stall losses)
        
        self.MOTOR_EFFICIENCY = 0.90            # Combined inverter & motor efficiency
        self.SYSTEM_INERTIA = 2.5               # Equivalent rotational inertia (kg*m^2)
        self.DRAG_TORQUE = 99*1.3558179483      # Constant estimated load/drag torque (Nm)
        
        # Accumulator (Battery)
        self.NOMINAL_VOLTAGE = 600.0            # Tractive system nominal voltage (V)
        self.INTERNAL_RESISTANCE = 0.15         # Pack internal resistance (Ohms)
        self.THERMAL_MASS = 15000.0             # Heat capacity * mass (J/K)
        
    def step(self, throttle_pct, initial_voltage, initial_temp_c, initial_rpm, dt):
        """
        Advances the powertrain simulation by one timestep (dt) using Current Control.
        """
        # Ensure inputs are within bounds
        throttle_pct = max(0.0, min(1.0, throttle_pct))
        
        # 1. MOTOR CURRENT & TORQUE CALCULATION
        # APPS requests a specific phase current from the inverter
        phase_current = throttle_pct * self.MAX_PHASE_CURRENT_A
        
        # Torque is directly proportional to phase current via Kt
        final_torque = phase_current * self.TORQUE_CONSTANT_KT
        
        # 2. MOTOR RPM CALCULATION
        # Net torque = Motor Torque - Load Torque
        net_torque = final_torque - self.DRAG_TORQUE
        alpha = net_torque / self.SYSTEM_INERTIA
        
        # Convert initial RPM to rad/s, apply acceleration, convert back
        initial_rad_s = initial_rpm * (2 * math.pi / 60.0)
        final_rad_s = initial_rad_s + (alpha * dt)
        
        # Prevent negative RPM for this simple forward-driving model
        final_rad_s = max(0.0, final_rad_s) 
        final_rpm = final_rad_s * (60.0 / (2 * math.pi))
        
        # 3. ELECTRICAL POWER & DC CURRENT CALCULATION
        # Mechanical Power (Watts) = Torque (Nm) * Angular Velocity (rad/s)
        mech_power = final_torque * final_rad_s
        
        if mech_power > 0:
            # While moving, power is mechanical work + efficiency losses
            elec_power = mech_power / self.MOTOR_EFFICIENCY
        else:
            # At 0 RPM (stall), mechanical power is 0, but creating torque requires current.
            # Power is consumed purely as copper losses (I^2 * R) in the motor windings.
            elec_power = (phase_current**2) * self.MOTOR_PHASE_RESISTANCE
            
        # DC Bus Current (Amps pulled from the accumulator)
        # P = V * I -> I = P / V
        if initial_voltage > 0:
            dc_bus_current = elec_power / initial_voltage
        else:
            dc_bus_current = 0.0
            
        # 4. BATTERY VOLTAGE CALCULATION
        # Voltage sag due to internal resistance
        final_voltage = self.NOMINAL_VOLTAGE - (dc_bus_current * self.INTERNAL_RESISTANCE)
        
        # 5. BATTERY TEMPERATURE CALCULATION
        # Joule heating in the accumulator: Q (Joules) = I^2 * R * t
        heat_generated_j = (dc_bus_current**2) * self.INTERNAL_RESISTANCE * dt
        
        temp_increase = heat_generated_j / self.THERMAL_MASS
        final_temp_c = initial_temp_c + temp_increase
        
        return {
            "final_voltage": final_voltage,
            "final_temp_c": final_temp_c,
            "final_rpm": final_rpm,
            "final_torque": final_torque,
            "phase_current": phase_current,
            "dc_bus_current": dc_bus_current 
        }

# --- Example Usage ---
if __name__ == "__main__":
    sim = EVPowertrainSimulator()
    
    dt = 0.1 # 100ms timestep
    current_voltage = 400.0
    current_temp = 25.0
    current_rpm = 0.0
    
    print("Time(s) | Throttle | RPM    | Phase(A) | DC Bus(A) | Torque(Nm) | Voltage(V)")
    print("-" * 80)
    
    for step in range(50):
        time = step * dt
        throttle = 1.0 # 100% throttle
        
        results = sim.step(throttle, current_voltage, current_temp, current_rpm, dt)
        
        current_voltage = results["final_voltage"]
        current_temp = results["final_temp_c"]
        current_rpm = results["final_rpm"]
        
        if step % 5 == 0: 
            print(f"{time:4.1f}    | {throttle * 100:3.0f}%     | {current_rpm:6.1f} | {results['phase_current']:8.1f} | {results['dc_bus_current']:9.1f} | {results['final_torque']:10.1f} | {current_voltage:10.1f}")