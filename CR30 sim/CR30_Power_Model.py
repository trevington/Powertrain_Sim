import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
from  scipy.constants import mph, lbf, g
from scipy.stats import linregress

class PowertrainSim:
    def __init__(self,
                 car_mass=212.0,
                 driver_mass=80.0,
                 time_step=0.05,
                 braking_deceleration_cutoff=3.0,
                 motor_efficiency=0.90,
                 inverter_cable_loss=0.98,
                 final_drive_efficiency=0.98,
                 regen_transfer_efficiency=0.80,
                 glv_system_draw=1000.0,
                 rbj=0.0001,
                 cell_params: dict = None,
                 temp_initial=38.0,
                 current_max=199.0,
                 dcir_safety_factor=1.5):
        """
        Initializes the CR30 powertrain properties and lookup models.
        """
        if cell_params is None:
            cell_params = {
                'V_MAX': 4.2, 'V_NOM': 3.5, 'V_MIN': 3.2,
                'MASS': 0.2667, 'SPECIFIC_HEAT': 954, 'COUNT': 140, 'CAPACITY': 13,
                'OCV_DATA': np.array([
                    [100, 4.26], [98.03922, 3.892], [96.07843, 3.868], [94.11765, 3.845],
                    [92.15686, 3.8005], [90.19608, 3.792], [88.2353, 3.77], [86.27451, 3.758],
                    [84.31373, 3.737], [82.35294, 3.714], [80.39216, 3.699], [78.43138, 3.681],
                    [76.47059, 3.668], [74.50981, 3.65], [72.54902, 3.632], [70.58824, 3.614],
                    [68.62746, 3.594], [66.66667, 3.566], [64.70589, 3.552], [62.7451, 3.54],
                    [60.78432, 3.528], [58.82354, 3.518], [56.86275, 3.505], [54.90197, 3.5],
                    [52.94118, 3.489], [50.9804, 3.484], [49.01962, 3.474], [47.05883, 3.47],
                    [45.09805, 3.466], [43.13726, 3.454], [41.17648, 3.447], [39.2157, 3.44],
                    [37.25491, 3.435], [35.29413, 3.428], [33.33334, 3.419], [31.37256, 3.414],
                    [29.41178, 3.404], [27.45099, 3.395], [25.49021, 3.391], [23.52942, 3.378],
                    [21.56864, 3.362], [19.60786, 3.353], [17.64707, 3.342], [15.68629, 3.33],
                    [13.7255, 3.313], [11.76472, 3.3], [9.803936, 3.296], [7.843152, 3.276],
                    [5.882368, 3.246], [3.921584, 3.204], [1.9608, 3.119], [0, 3]
                ])
            }

        # Aero data [mph, lbf]
        AERO_NUMBERS = np.array([[25, 30], [35,59], [45, 99], [55, 148], [65, 207], [75, 275]], dtype = float)

        # Rolling Resistance Data [m/s, NA]
        ROLLING_RESISTANCE = np.array([[5.56, 0.015], [11.11, 0.0165], [16.67, 0.018], [22.22, 0.021], [27.78, 0.025], [33.33, 0.028]], dtype = float)

        # Unit Conversions
        AERO_NUMBERS[:, 0] *= mph   # mph to m/s
        AERO_NUMBERS[:, 1] *= lbf   # lbf to N

        # Data lookup models (Interpolation is more accurate for physics curves)
        self.aero_data = AERO_NUMBERS
        self.rolling_resistance = ROLLING_RESISTANCE

        # Mass properties
        self.car_mass = car_mass
        self.driver_mass = driver_mass
        self.total_mass = driver_mass + car_mass

        self.time_step = time_step

        self.braking_cutoff = braking_deceleration_cutoff

        # Efficiency properties
        self.motor_eff = motor_efficiency
        self.cable_loss = inverter_cable_loss
        self.drive_eff = final_drive_efficiency
        self.regen_eff = regen_transfer_efficiency
        self.total_eff = motor_efficiency * inverter_cable_loss * final_drive_efficiency

        # GLV draw
        self.glv_draw = glv_system_draw # W 

        # Battery Properties
        self.dcir_safety_factor = dcir_safety_factor 
        self.rbj = rbj # Ohm
        self.current_max = current_max # A
        self.temp_initial = temp_initial # C 

        self.cell_v_max = cell_params['V_MAX']
        self.cell_v_nom = cell_params['V_NOM']
        self.cell_v_min = cell_params['V_MIN']
        self.cell_mass = cell_params['MASS']
        self.cell_specific_heat = cell_params['SPECIFIC_HEAT']
        self.cell_count = cell_params['COUNT']
        self.cell_capacity = cell_params['CAPACITY']
        self.total_energy_j = cell_params['CAPACITY'] * cell_params['V_NOM'] * 3600

        # OCV Curve (Ensure x is increasing for np.interp)
        ocv_data = cell_params['OCV_DATA']
        sorted_ocv = ocv_data[ocv_data[:, 0].argsort()]
        self.ocv_soe_pts = sorted_ocv[:, 0]
        self.ocv_voltage_pts = sorted_ocv[:, 1]

    def step(self, velocity_initial, acceleration, energy_initial, temp_initial):
        """Advances the simulation by one time step."""
        drag = np.interp(velocity_initial, self.aero_data[:, 0], self.aero_data[:, 1])
        drag = max(drag, 0)
        
        # m/s^2 of acceleration
        velocity_final = acceleration * self.time_step + velocity_initial

        # W power regenerated
        if acceleration <= -self.braking_cutoff:
            instant_aero_power = self.glv_draw
            instant_regen_power = self.total_mass * self.regen_eff * velocity_initial * acceleration
            instant_rolling_resistance_power = 0
        else:
            instant_aero_power = (drag * velocity_initial) / self.total_eff
            instant_regen_power = 0 
            rr_coeff = np.interp(velocity_initial, self.rolling_resistance[:, 0], self.rolling_resistance[:, 1])
            instant_rolling_resistance_power = (self.total_mass * g * rr_coeff * velocity_initial) / self.total_eff

        if acceleration > 0:
            instant_acceleration_power = (velocity_initial * self.total_mass * acceleration) / self.total_eff
        else:
            instant_acceleration_power = 0

        # W*s of energy
        tb_power_draw = instant_acceleration_power + instant_aero_power + instant_regen_power + instant_rolling_resistance_power
        if tb_power_draw > 80000:
            warnings.warn("Warning......> 80kw power used!!!")
            
        power = tb_power_draw / self.cell_count
        energy_consumed = power * self.time_step
        energy_final = energy_initial - energy_consumed
        soe = 100 * energy_final / self.total_energy_j 
        ocv = np.interp(soe, self.ocv_soe_pts, self.ocv_voltage_pts)
        current = power / ocv

        resistance = self._calculate_dcir(ocv, temp_initial)
        heat_generated = (current**2) * resistance * self.time_step
        temp_rise = heat_generated / (self.cell_mass * self.cell_specific_heat)
        temp_final = temp_initial + temp_rise

        return {
            'final_velocity': velocity_final,
            'energy_used': energy_consumed,
            'temp_final': temp_final,
            'power': power,
            'soe': soe
        }

    def _calculate_dcir(self, ocv, temp):
        """Calculates internal resistance using a 3rd-order polynomial fit."""
        poly_val = (0.2935676 + 
                    -0.171888 * ocv + 
                    -0.00373033 * temp + 
                    0.03292212 * (ocv**2) + 
                    0.001604118 * ocv * temp + 
                    0.000009206928 * (temp**2) + 
                    -0.001941274 * (ocv**3) + 
                    -0.0001845216 * (ocv**2) * temp +
                    -0.000001707152 * ocv * (temp**2) + 
                    -0.000000002083755 * (temp**3))
        
        return self.rbj + (poly_val * self.dcir_safety_factor)