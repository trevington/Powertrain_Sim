from datetime import datetime
import numpy as np
import pandas as pd
from CR30_Power_Model import powertrain_sim

def status(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def simulate(sim, input_file, output_file):
    energy_initial = sim.CELL_CAPACITY * sim.CELL_V_NOM * 3600 # J
    temp_initial = sim.TEMP_INITIAL # C
    
    operating_points = pd.read_csv('{}.csv'.format(input_file))

    data_array = operating_points.values

    results_list = []
    lap = 1
    starting_point = data_array[0, 1:3]  # Starting X and Y coordinates

    for i in range(len(data_array)):
        velocity_initial = data_array[i, 3]
        acceleration = data_array[i, 4]
        
        # Check if the car has completed a lap
        if np.linalg.norm(data_array[i, 1:3] - starting_point) < 1.0 and i != 0 and np.linalg.norm(data_array[i-1, 1:3] - starting_point) >= 1.0:
            lap += 1
            status(f"Lap {lap} completed.")

        step_output = sim.step(velocity_initial, acceleration, energy_initial, temp_initial)
        results_list.append({'time (s)': data_array[i,0], 'lap': lap, 'x coordinate (m)': data_array[i,1], 'y coordinate (m)': data_array[i,2]} | step_output)

        energy_initial -= step_output['energy_used (Ws)']
        temp_initial = step_output['temp_final (C)']

    status("Simulation complete.")

    final_results_df = pd.DataFrame(results_list)
    final_results_df.to_csv('outputs\\{}.csv'.format(output_file), index=False)
    status("Simulation saved")

'''
CR30 Properties
'''
CR30_CAR_MASS = 212.0 # kg mass of car
CR30_DRIVER_MASS = 80.0 # kg mass of driver

CR30_TIME_STEP = 0.05 # s time between data points

CR30_BREAKING_DECELERATION_CUTOFF = 3 # m/s^2

CR30_MOTOR_EFFICIENCY = 0.90 # %
CR30_INVERTER_CABLE_LOSS = 0.98 # %
CR30_FINAL_DRIVE_EFFICIENCY = 0.98 # %
CR30_REGEN_TRANSFER_EFFICIENCY = 0.80 # % NOTE: 0.8 FOR TESTING

# GLV draw
CR30_GLV_SYSTEM_DRAW = 1000 # W 

# Battery
CR30_RBJ = 0.0001 # Ohm

CR30_CELL = {'V_MAX': 4.2, # V
             'V_NOM': 3.5, # V
             'V_MIN': 3.2, # V
             'MASS': 0.2667, # kg
             'SPECIFIC_HEAT': 954, # J/(kg*K)
             'COUNT': 140,
             'CAPACITY': 13, # Ah
             'OCV_DATA': np.array([[100, 4.26],[98.03922, 3.892],[96.07843, 3.868],[94.11765, 3.845],[92.15686, 3.8005],[90.19608, 3.792],[88.2353, 3.77],[86.27451, 3.758],[84.31373, 3.737],[82.35294, 3.714],[80.39216, 3.699],[78.43138, 3.681],[76.47059, 3.668],[74.50981, 3.65],[72.54902, 3.632],[70.58824, 3.614],[68.62746, 3.594],[66.66667, 3.566],[64.70589, 3.552],[62.7451, 3.54],[60.78432, 3.528],[58.82354, 3.518],[56.86275, 3.505],[54.90197, 3.5],[52.94118, 3.489],[50.9804, 3.484],[49.01962, 3.474],[47.05883, 3.47],[45.09805, 3.466],[43.13726, 3.454],[41.17648, 3.447],[39.2157, 3.44],[37.25491, 3.435],[35.29413, 3.428],[33.33334, 3.419],[31.37256, 3.414],[29.41178, 3.404],[27.45099, 3.395],[25.49021, 3.391],[23.52942, 3.378],[21.56864, 3.362],[19.60786, 3.353],[17.64707, 3.342],[15.68629, 3.33],[13.7255, 3.313],[11.76472, 3.3],[9.803936, 3.296],[7.843152, 3.276],[5.882368, 3.246],[3.921584, 3.204],[1.9608, 3.119],[0, 3]])
             }

CR30_TEMP_INITIAL = 38 # C
CR30_DCIR_SAFETY_FACTOR = 1.5
CR30_CURRENT_MAX = 199.0 # Amps

CR30_sim = powertrain_sim(CR30_CAR_MASS, 
                     CR30_DRIVER_MASS, 
                     CR30_TIME_STEP, 
                     CR30_BREAKING_DECELERATION_CUTOFF, 
                     CR30_MOTOR_EFFICIENCY, 
                     CR30_INVERTER_CABLE_LOSS, 
                     CR30_FINAL_DRIVE_EFFICIENCY, 
                     0.0, # REGEN
                     CR30_GLV_SYSTEM_DRAW, 
                     CR30_RBJ, 
                     CR30_CELL, 
                     CR30_TEMP_INITIAL, 
                     CR30_CURRENT_MAX, 
                     CR30_DCIR_SAFETY_FACTOR)

simulate(CR30_sim, 'tracks\\endurance_track_timed', 'CR30_Michican_EV_2026')

simulate(CR30_sim, 'tracks\\cr28_endurance_track_timed', 'CR30_Michican_IC_2025')