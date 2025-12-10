import os
from tools_plus import *
from scipy.signal import savgol_filter
from scipy import signal
import pandas as pd


sample_rate = 100  # 100 Hz
data = []

output_dir = '_cycle_start_end_Excel'
# Read data
files_name = os.listdir('../dataset/')
for i in range(0, len(files_name)):
    print(i)
    ppg = Read_raw_ppg_signal('../dataset/' + files_name[i])
    name = files_name[i].split('.')[0]

    # Filter data
    filter_pgg = savgol_filter(ppg, 201, 2)

    # Find valleys
    valleys, _ = signal.find_peaks(
        -filter_pgg,
        distance=int(sample_rate * 0.8),  # At least 1 second apart (60 BPM)
        prominence=30,  # Valley prominence
    )

    # Extract cycles and record data
    if len(valleys) >= 2:
        cycle_start, cycle_end = valleys[0], valleys[1]
        data.append({
            "Name": name,  # First column: filename (without extension)
            "cycle_start": cycle_start,  # Second column: start index
            "cycle_end": cycle_end  # Third column: end index
        })
    else:
        print(f"Skipped {name}: Not enough valleys detected")

# Sort by Name (first by number before _ in ascending order, then by number after _ in ascending order)
data_sorted = sorted(data, key=lambda x: (
    int(x["Name"].split('_')[0]),  # Number before _ ascending
    int(x["Name"].split('_')[1])  # Number after _ ascending
))

if data_sorted:
    df = pd.DataFrame(data_sorted)
    df.to_excel(os.path.join(output_dir, "cycle_data.xlsx"), index=False)
    print("Data saved to cycle_data.xlsx")
else:
    print("No valid cycle data to save")