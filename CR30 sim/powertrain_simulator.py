import os
from datetime import datetime
import numpy as np
import pandas as pd
from CR30_Power_Model import PowertrainSim

def status(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def simulate(sim, input_file, output_file):
    energy_initial = sim.total_energy_j # J
    temp_initial = sim.temp_initial # C
    
    operating_points = pd.read_csv(f'{input_file}.csv')

    data_array = operating_points.values

    results_list = []

    for i in range(len(data_array)):
        velocity_initial = data_array[i, 0]
        acceleration = data_array[i, 1]
        
        step_output = sim.step(velocity_initial, acceleration, energy_initial, temp_initial)
        results_list.append(step_output)
        status("Operating point {} simulated.".format(i))

        energy_initial -= step_output['energy_used']
        temp_initial = step_output['temp_final']

    status("Simulation complete.")

    output_dir = 'sim_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_results_df = pd.DataFrame(results_list)
    final_results_df.to_csv(f'{output_dir}/{output_file}.csv', index=False)
    status("Simulation saved")

CR30_sim = PowertrainSim(regen_transfer_efficiency=0.0)

CR31_sim = PowertrainSim(driver_mass=75.0, regen_transfer_efficiency=0.0)

simulate(CR30_sim, 'CR28_Endurance_Sim_Data', 'CR30_endurance_output')
simulate(CR31_sim, 'CR28_Endurance_Sim_Data', 'CR31_endurance_output')