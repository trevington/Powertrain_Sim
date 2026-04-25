import pandas as pd
from CR30_Power_Model import powertrain_sim

sim = powertrain_sim()

operating_points = pd.read_csv('CR28_Endurance_Sim_Data.csv')

data_array = operating_points.values

results_list = []

for i in range(len(data_array)):
    velocity_initial = data_array[i, 0]
    acceleration = data_array[i, 1]
    
    step_output = sim.step(velocity_initial, acceleration)
    
    results_list.append(step_output)

final_results_df = pd.DataFrame(results_list)
final_results_df.to_csv('sim_output.csv', index=False)