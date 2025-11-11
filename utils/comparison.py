import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
'''
FlamMap MTT Arrival Times - time unit is minutes
ABM arrival times are simulated in SECONDS, so convert to minutes for apples-to-apples.
'''

script_dir = os.path.dirname(os.path.abspath(__file__))
abm_path = os.path.join(script_dir, "fire_arrival_times.csv")  # ABM output (seconds)
# abm_path = os.path.join(script_dir, "resampled_fire_arrival_times.csv")
flammap_path = os.path.join(script_dir, "flammap_arrival_times.csv")  # FlamMap output (minutes)
# flammap_path = os.path.join(script_dir, "flammap_resampled_arrival_times.csv")

# Load data
abm_data_min = pd.read_csv(abm_path, header=None).to_numpy()
flammap_data_min = pd.read_csv(flammap_path, header=None).to_numpy()

# Convert ABM seconds -> minutes; round
abm_data = np.round(abm_data_min, 4)
# abm_data = np.flipud(abm_data)
flammap_data = np.round(flammap_data_min, 4)

# Pretty printing (no scientific notation)
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

# Centers
r1, c1 = abm_data.shape[0] // 2, abm_data.shape[1] // 2
r2, c2 = flammap_data.shape[0] // 2 - 1, flammap_data.shape[1] // 2

print("ABM data shape:", abm_data.shape)
print("ABM center cell indices:", (r1, c1))
print("ABM center cell arrival time (min):", abm_data[r1, c1])

print("FlamMap data shape:", flammap_data.shape)
print("FlamMap center cell indices:", (r2, c2))
print("FlamMap center cell arrival time (min):", flammap_data[r2, c2])

neighbors_abm = abm_data[r1-3:r1+5, c1-3:c1+5]
neighbors_flammap = flammap_data[r2-3:r2+5, c2-3:c2+5]

print("\nABM neighbor arrival times (minutes, rounded to 4 decimals):\n", neighbors_abm)
print("\nFlamMap neighbor arrival times (minutes, rounded to 4 decimals):\n", neighbors_flammap)

plt.imshow(abm_data, cmap="hot", origin="upper")
plt.title("ABM Fire Arrival Times (Minutes)")
plt.colorbar(label="Minutes")
plt.show()
