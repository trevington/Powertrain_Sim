from track_processor import process_track_coordinates
from physics_profiler import add_velocity_targets

'''if __name__ == "__main__":
    raw_csv = 'endurance.csv'
    geometry_csv = 'endurance_processed.csv'
    final_csv = 'endurance_meters.csv'

    print("--- Phase 1: Coordinate Processing ---")
    success = process_track_coordinates(raw_csv, geometry_csv)
    
    if success:
        print("\n--- Phase 2: Physics Profiling ---")
        add_velocity_targets(
            input_csv=geometry_csv, 
            output_csv=final_csv,
            max_lat_g=1.5,
            max_long_g=0.8
        )'''

if __name__ == "__main__":
    add_velocity_targets(
        input_csv='tracks\\endurance_track.csv',
        output_csv='tracks\\endurance_track.csv',
        max_lat_g=2.0,
        max_long_g=0.6
    )