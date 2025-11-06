import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
abm_path = os.path.join(script_dir, "fire_arrival_times.csv")
flammap_path = os.path.join(script_dir, "flammap_arrival_times.csv")

abm_data = pd.read_csv(abm_path, header=None).to_numpy()
flammap_data = pd.read_csv(flammap_path, header=None).to_numpy()

r, c = abm_data.shape[0] // 2, abm_data.shape[1] // 2

print("ABM center cell arrival time:", abm_data[r, c])
print("FlamMap center cell arrival time:", flammap_data[r, c])

neighbors_abm = abm_data[r-1:r+2, c-1:c+2]
neighbors_flammap = flammap_data[r-1:r+2, c-1:c+2]

print("ABM neighbor arrival times:\n", neighbors_abm)
print("FlamMap neighbor arrival times:\n", neighbors_flammap)


#TODO: Need to ensure I save the ABM arrival times in the same format as FlamMap for proper comparison
'''
Once proper format for arrival times compare neighbors of center cell and use a buffer to compare overall differences. Ex: if flammap has 100 min arrival for next cell and abm has 40 then need to add a 60 minute buffer to abm neighbors and can then compare how the rest of the grid looks.
'''