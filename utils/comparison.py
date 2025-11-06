import pandas as pd
import os

'''
FlamMap MTT Arrival Times - time unit is minutes
'''

script_dir = os.path.dirname(os.path.abspath(__file__))
# abm_path = os.path.join(script_dir, "fire_arrival_times.csv")
abm_path = os.path.join(script_dir, "resampled_fire_arrival_times.csv")
flammap_path = os.path.join(script_dir, "flammap_arrival_times.csv")

abm_data = pd.read_csv(abm_path, header=None).to_numpy()
flammap_data = pd.read_csv(flammap_path, header=None).to_numpy()

r1, c1 = abm_data.shape[0] // 2, abm_data.shape[1] // 2
r2, c2 = flammap_data.shape[0] // 2 - 1, flammap_data.shape[1] // 2

print("ABM data shape:", abm_data.shape)
print("ABM center cell indices:", (r1, c1))
print("ABM center cell arrival time:", abm_data[r1, c1])

print("FlamMap data shape:", flammap_data.shape)
print("FlamMap center cell indices:", (r2, c2))
print("FlamMap center cell arrival time:", flammap_data[r2, c2])

neighbors_abm = abm_data[r1-1:r1+2, c1-1:c1+2]
neighbors_flammap = flammap_data[r2-1:r2+2, c2-1:c2+2]

print("ABM neighbor arrival times:\n", neighbors_abm)
print("FlamMap neighbor arrival times:\n", neighbors_flammap)


#TODO: Need to ensure I save the ABM arrival times in the same format as FlamMap for proper comparison - this should be done now.
'''
Once proper format for arrival times compare neighbors of center cell and use a buffer to compare overall differences. Ex: if flammap has 100 min arrival for next cell and abm has 40 then need to add a 60 minute buffer to abm neighbors and can then compare how the rest of the grid looks.
'''