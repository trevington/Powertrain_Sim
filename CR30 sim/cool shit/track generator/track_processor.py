import pandas as pd
import numpy as np

def process_track_coordinates(input_csv, output_csv):
    """
    Converts track coordinates from feet to meters, zeros them at the 
    upper-left corner, and ensures counter-clockwise winding.
    """
    try:
        # 1. Load the data
        df = pd.read_csv(input_csv, skipinitialspace=True)
        
        # Extract raw values
        x_ft = df['X coord (ft)'].values
        y_ft = df['Y coord (ft)'].values
        
        # 2. Convert Feet to Meters
        x_m = x_ft * 0.3048
        y_m = y_ft * 0.3048
        
        # 3. Zero data at the Upper Left Corner (Min X, Max Y)
        x_origin = np.min(x_m)
        y_origin = np.max(y_m)
        
        x_zeroed = x_m - x_origin
        y_zeroed = y_m - y_origin
        
        # 4. Order points Counter-Clockwise (CCW) via Shoelace Formula
        area = 0.0
        n = len(x_zeroed)
        for i in range(n):
            j = (i + 1) % n
            area += x_zeroed[i] * y_zeroed[j]
            area -= x_zeroed[j] * y_zeroed[i]
            
        if area < 0:
            print("Detected Clockwise order. Reversing to CCW...")
            x_final = x_zeroed[::-1]
            y_final = y_zeroed[::-1]
        else:
            x_final = x_zeroed
            y_final = y_zeroed
            
        # 5. Export Geometric Data
        output_df = pd.DataFrame({
            'X coord (m)': x_final,
            'Y coord (m)': y_final
        })
        
        output_df.to_csv(output_csv, index=False)
        print(f"Successfully processed {n} waypoints to: {output_csv}")
        return True

    except Exception as e:
        print(f"Error in coordinate processing: {e}")
        return False