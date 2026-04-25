import math
import matplotlib.pyplot as plt
import numpy as np
from  scipy.constants import mph, lbf, g
from scipy.stats import linregress

class powertrain_sim:
    def __init__(self):
        '''
        CR30 powertrain properties
        '''
        # Aero data [mph, lbf]
        AERO_NUMBERS = np.array([[25, 30], [35,59], [45, 99], [55, 148], [65, 207], [75, 275]], dtype = float)

        # Rolling Resistance Data [m/s, NA]
        ROLLING_RESISTANCE = np.array([[5.56, 0.015], [11.11, 0.0165], [16.67, 0.018], [22.22, 0.021], [27.78, 0.025], [33.33, 0.028]], dtype = float)

        # Unit Conversions
        AERO_NUMBERS[:, 0] *= mph   # mph to m/s
        AERO_NUMBERS[:, 1] *= lbf   # lbf to N

        # Data lookup functions
        self.AERO_LOOKUP = linregress(AERO_NUMBERS[:,0], AERO_NUMBERS[:,1])
        self.ROLLING_RESISTANCE_LOOKUP = linregress(ROLLING_RESISTANCE[:,0], ROLLING_RESISTANCE[:,1])
        self.AERO_NUMBERS = AERO_NUMBERS
        self.ROLLING_RESISTANCE = ROLLING_RESISTANCE

        # Mass properties
        self.CAR_MASS = 212.0 # kg mass of car
        self.DRIVER_MASS = 80.0 # kg mass of driver
        self.TOTAL_MASS = self.DRIVER_MASS + self.CAR_MASS # Total car mass

        self.TIME_STEP = 0.05 # s time between data points

        self.BREAKING_DECELERATION_CUTOFF = 3 # m/s^2

        # Efficiency properties
        self.MOTOR_EFFICIENCY = 0.90 # %
        self.INVERTER_CABLE_LOSS = 0.98 # %
        self.FINAL_DRIVE_EFFICIENCY = 0.98 # %
        self.REGEN_TRANSFER_EFFICIENCY = 0.80 # %
        self.TOTAL_PT_EFFICIENCY = self.MOTOR_EFFICIENCY * self.INVERTER_CABLE_LOSS * self.FINAL_DRIVE_EFFICIENCY

        # GLV draw
        self.GLV_SYSTEM_DRAW = 1 * 1000 # W 



    def step(self, velocity_initial, acceleration):
        # N of aero drag
        drag = self.AERO_LOOKUP.intercept + self.AERO_LOOKUP.slope * velocity_initial
        drag = max(drag, 0)
        
        # m/s^2 of acceleration
        velocity_final = acceleration * self.TIME_STEP + velocity_initial

        # W power regenerated
        if acceleration <= -self.BREAKING_DECELERATION_CUTOFF:
            instant_aero_power = self.GLV_SYSTEM_DRAW
            instant_regen_power = self.TOTAL_MASS * self.REGEN_TRANSFER_EFFICIENCY * velocity_initial * acceleration
            instant_rolling_resistance_power = 0
        else:
            instant_aero_power = (drag * velocity_initial)/(self.TOTAL_PT_EFFICIENCY)
            instant_regen_power = 0 
            instant_rolling_resistance_power = self.TOTAL_MASS * g * (self.ROLLING_RESISTANCE_LOOKUP.slope * velocity_initial + self.ROLLING_RESISTANCE_LOOKUP.intercept) * velocity_initial / self.TOTAL_PT_EFFICIENCY

        if acceleration > 0:
            instant_acceleration_power = (velocity_initial * self.TOTAL_MASS * acceleration) / self.TOTAL_PT_EFFICIENCY
        else:
            instant_acceleration_power = 0

        # W*s of energy
        acceleration_energy = instant_acceleration_power * self.TIME_STEP
        aero_energy = instant_aero_power * self.TIME_STEP
        regen_energy = instant_regen_power * self.TIME_STEP # Ws energy regenerated 
        rolling_resistance_energy = instant_rolling_resistance_power * self.TIME_STEP 

        return {
            'final_velocity': velocity_final,
            'drag': drag,
            'instant_acceleration_power': instant_acceleration_power,
            'instant_aero_power': instant_aero_power,
            'instant_regen_power': instant_regen_power,
            'instant_rolling_resistance_power': instant_rolling_resistance_power,
            'acceleration_energy': acceleration_energy,
            'aero_energy': aero_energy,
            'regen_energy': regen_energy,
            'rolling_resistance_energy': rolling_resistance_energy,
            'tb_power_draw': instant_acceleration_power + instant_aero_power + instant_regen_power + instant_rolling_resistance_power,
            'energy_used': acceleration_energy + aero_energy + regen_energy + rolling_resistance_energy
        }