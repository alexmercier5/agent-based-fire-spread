'''
Basic framework for mesa model
agent-based-fire-spread/
├─ .env/                 # 
├─ main_resampled.tif
├─ main.py               # entry point for your Mesa model
├─ model/
│   ├─ __init__.py
│   ├─ fire_model.py     # Mesa Model class
│   ├─ cell_agent.py     # Individual grid cell agent
│   └─ fire_agent.py     # Fire agent (optional)
└─ utils/
    └─ raster_utils.py   # functions for reading/resampling TIFF
'''
import contextlib
import os
import numpy as np
from model.fire_model import FireSpreadModel
from model.fire_agent import FireAgent
from model.cell_agent import CellAgent

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tif_path = os.path.join(script_dir, "resampled_main.tif")
    model = FireSpreadModel(tif_path)

    # Ignite center cell
    center_row = model.rows // 2
    center_col = model.cols // 2
    center_agent = model.grid.get_cell_list_contents([(center_row, center_col)])[0]
    center_agent.burning = True

    for i in range(10):
        model.step()
        print(f"Step {i+1} completed")

        # Print a random cell agent and its properties
        rand_row = np.random.randint(0, model.rows)
        rand_col = np.random.randint(0, model.cols)
        sample_agent = model.grid.get_cell_list_contents([(rand_col, rand_row)])[0]
        if isinstance(sample_agent, FireAgent):
            pass
        else:
            print(f"Sample Agent at ({rand_row}, {rand_col}) - Burning: {sample_agent.burning}, Burned: {sample_agent.burned}, Elevation: {sample_agent.elevation}, Slope: {sample_agent.slope}, Aspect: {sample_agent.aspect},Fuel: {sample_agent.fuel}, Canopy Cover: {sample_agent.canopy_cover}")
