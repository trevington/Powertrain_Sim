import numpy as np

class FSAEScorer:
    """
    Calculates points based on FSAE Electric rules for the Endurance event.
    Note: These formulas typically require the time/energy of the fastest 
    completing car (T_min, E_min) and the slowest (T_max).
    """
    def __init__(self, t_min_limit=1.45, e_min_limit=1.0):
        self.t_min_limit = t_min_limit
        self.total_laps = 22

    def calculate_endurance_score(self, sim_time, energy_wh, laps_completed, 
                                 t_min_reference=1308.281,
                                 e_min_reference=0.0840):
        """
        Calculates points for the Endurance and Efficiency events.
        :param sim_time: Total time taken for the session (s)
        :param energy_wh: Total energy consumed (Wh)
        :param laps_completed: Number of laps finished
        :param t_min_reference: Reference fastest time for 250pt scale
        :param e_min_reference: Reference lowest energy for efficiency
        """
        if laps_completed < self.total_laps:
            # DNF logic: Points based on laps completed
            # Typically 25pts + (laps/total * points)
            return {'endurance_points': (laps_completed / self.total_laps) * 25.0, 
                    'efficiency_points': 0.0, 
                    'status': 'DNF'}

        # 1. Endurance Points (out of 275)
        # Formula: 25 + 250 * ((T_max/T_your) - 1) / ((T_max/T_min) - 1)
        # Using a simplified version comparing against a target time
        t_max = t_min_reference * self.t_min_limit
        if sim_time > t_max:
            end_pts = 25.0 # Minimum points for finishing
        elif sim_time < t_max:
            end_pts = 275.0 # Maximum points for finishing
        else:
            end_pts = 25 + 250.0 * ((t_max / sim_time) - 1) / ((t_max / t_min_reference) - 1) + 25.0

        # 2. Efficiency Points (out of 100)
        efficiency_factor = (t_min_reference/22)/(sim_time/laps_completed) * (e_min_reference)/(energy_wh/laps_completed)
        
        min_efficiency_factor = 0.433
        max_efficiency_factor = 0.797
        
        eff_pts = 100.0 * (efficiency_factor - min_efficiency_factor) / (max_efficiency_factor - min_efficiency_factor)

        if eff_pts > 100.0: eff_pts = 100.0
        elif eff_pts < 0.0: eff_pts = 0.0

        return {
            'endurance_points': float(np.round(end_pts, 2)),
            'efficiency_points': float(np.round(eff_pts, 2)),
            'total_points': float(np.round(end_pts + eff_pts, 2)),
            'energy_per_lap': energy_wh / self.total_laps,
            'status': 'Finished'
        }