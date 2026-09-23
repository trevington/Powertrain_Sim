import openmdao.api as om
import numpy as np
import sys
import os

# Ensure we can import the simulation engine
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
from main import run_simulation

class RatioComp(om.ExplicitComponent):
    """
    OpenMDAO component wrapping the simulation for front gear ratio optimization.
    """
    def setup(self):
        # Design variable input
        self.add_input('gear_ratio', val=4.65)
        
        # Metrics we want to track or optimize
        self.add_output('best_lap_time', val=0.0)
        self.add_output('energy_used_wh', val=0.0)
        
        # Declare that inputs affect outputs to satisfy connectivity checks.
        # method='fd' tells OpenMDAO to use Finite Difference for gradients.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Access the scalar value by indexing the input array
        ratio = inputs['gear_ratio'][0]
        
        params = {
            'mode': 'Acceleration',
            'laps_to_run': 1,
            'motor_config': 'RWD',
            'gear_ratio': ratio,
        }
        
        # Run simulation in headless mode
        # Wrap in try-except to prevent the optimizer from crashing on invalid physics steps
        try:
            results = run_simulation(params, verbose=False, plot=False)
            outputs['best_lap_time'] = results['best_lap_time']
            outputs['energy_used_wh'] = results['energy_used_wh']
        except Exception:
            outputs['best_lap_time'] = 999.0 
            outputs['energy_used_wh'] = 10000.0

def run_optimization():
    # Create the OpenMDAO problem
    prob = om.Problem()
    model = prob.model

    # Add our simulation component
    model.add_subsystem('ratio_sim', RatioComp(), promotes=['*'])

    # Setup the optimizer (COBYLA is derivative-free and handles constraints)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'COBYLA'
    prob.driver.options['maxiter'] = 50
    prob.driver.options['tol'] = 1e-3

    # Define Design Variables and Objective
    # Using a search range around the previous study values (13.4 - 17.0)
    model.add_design_var('gear_ratio', lower=2.0, upper=4.75)
    model.add_objective('best_lap_time', scaler=1.0) # Minimize energy use

    prob.setup()
    
    print(f"Starting Front Hub Gear Ratio Optimization...")
    print("-" * 70)
    
    prob.run_driver()

    # Extract Results
    opt_ratio = prob.get_val('gear_ratio')[0]
    best_time = prob.get_val('best_lap_time')[0]
    best_regen = prob.get_val('energy_used_wh')[0]

    print("\n" + "="*40)
    print("OPENMDAO OPTIMIZATION RESULTS")
    print("="*40)
    print(f"Optimal Gear Ratio:        {opt_ratio:.4f}")
    print(f"Minimized Energy Use:      {best_regen:.4f} Wh")
    print(f"Corresponding Lap Time:    {best_time:.4f} s")

if __name__ == "__main__":
    run_optimization()