import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
from  scipy.constants import mph, lbf, g
from scipy.stats import linregress

class powertrain_sim:
    def __init__(self, 
                 CAR_MASS, DRIVER_MASS, 
                 TIME_STEP, 
                 BREAKING_DECELERATION_CUTOFF, 
                 MOTOR_EFFICIENCY, 
                 INVERTER_CABLE_LOSS, 
                 FINAL_DRIVE_EFFICIENCY, 
                 REGEN_TRANSFER_EFFICIENCY, 
                 GLV_SYSTEM_DRAW, 
                 RBJ, 
                 CELL: dict,  
                 TEMP_INITIAL, CURRENT_MAX, DCIR_SAFETY_FACTOR):
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
        self.CAR_MASS = CAR_MASS # kg mass of car
        self.DRIVER_MASS = DRIVER_MASS # kg mass of driver
        self.TOTAL_MASS = DRIVER_MASS + CAR_MASS # Total car mass

        self.TIME_STEP = TIME_STEP # s time between data points

        self.BREAKING_DECELERATION_CUTOFF = BREAKING_DECELERATION_CUTOFF # m/s^2

        # Efficiency properties
        self.MOTOR_EFFICIENCY = MOTOR_EFFICIENCY # %
        self.INVERTER_CABLE_LOSS = INVERTER_CABLE_LOSS # %
        self.FINAL_DRIVE_EFFICIENCY = FINAL_DRIVE_EFFICIENCY # %
        self.REGEN_TRANSFER_EFFICIENCY = REGEN_TRANSFER_EFFICIENCY # %
        self.TOTAL_PT_EFFICIENCY = MOTOR_EFFICIENCY * INVERTER_CABLE_LOSS * FINAL_DRIVE_EFFICIENCY

        # GLV draw
        self.GLV_SYSTEM_DRAW = GLV_SYSTEM_DRAW # W 

        '''
        BATTERY PROPERTIES
        '''
        self.DCIR_SAFETY_FACTOR = DCIR_SAFETY_FACTOR 
        self.RBJ = RBJ # Ohm
        self.CURRENT_MAX = CURRENT_MAX # A
        self.TEMP_INITIAL = TEMP_INITIAL # C 

        self.CELL_V_MAX = CELL['V_MAX'] # V
        self.CELL_V_NOM = CELL['V_NOM'] # V
        self.CELL_V_MIN = CELL['V_MIN'] # V
        self.CELL_MASS = CELL['MASS'] # kg
        self.CELL_SPECIFIC_HEAT = CELL['SPECIFIC_HEAT'] # J/(kg*K)
        self.CELL_COUNT = CELL['COUNT'] #
        self.CELL_CAPACITY = CELL['CAPACITY'] # Ah
        self.ENERGY = CELL['CAPACITY'] * CELL['V_NOM']* 3600 # J

        # VOLTAGE CURVE
        OCV_DATA = CELL['OCV_DATA']
        self.OCV_LOOKUP = linregress(OCV_DATA[:, 0], OCV_DATA[:, 1])

    def step(self, velocity_initial, acceleration, energy_initial, temp_initial):
        '''
        PERFORMANCE SIM
        '''
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

        '''
        BATTERY SIM
        '''
        tb_power_draw = instant_acceleration_power + instant_aero_power + instant_regen_power + instant_rolling_resistance_power
        if tb_power_draw > 80000:
            warnings.warn("Warning......> 80kw power used!!!")
        power = tb_power_draw / self.CELL_COUNT
        energy_consumed = power * self.TIME_STEP
        energy_final = energy_initial - energy_consumed
        soe = 100 * energy_final / self.ENERGY 
        ocv = self.OCV_LOOKUP.slope * soe + self.OCV_LOOKUP.intercept
        current = power / ocv

        peak_power_capability = self.CURRENT_MAX * ocv * self.CELL_COUNT

        resistance = self.RBJ + (0.2935676 + 
                                 -0.171888 * ocv + 
                                 -0.00373033 * temp_initial + 
                                 0.03292212 * (ocv**2) + 
                                 0.001604118 * ocv * temp_initial + 
                                 0.000009206928 * (temp_initial**2) + 
                                 -0.001941274 * (ocv**3) + 
                                 -0.0001845216 * (ocv**2) * temp_initial +
                                 -0.000001707152 * ocv * (temp_initial**2) + 
                                 -0.000000002083755 * (temp_initial**3)) * self.DCIR_SAFETY_FACTOR
        heat_generated = (current**2) * resistance * self.TIME_STEP
        temp_rise = heat_generated / (self.CELL_MASS * self.CELL_SPECIFIC_HEAT)

        temp_final = temp_initial + temp_rise

        return {
            'final_velocity': velocity_final,
            'energy_used': energy_consumed,
            'temp_final': temp_final,
            'power': power,
            'soe': soe
        }