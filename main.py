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
    └─ setup.py   # functions for reading/resampling TIFF
    └─ visualization.py # functions for setting up server based visualization: IN PROGRESS
    └─ fuel_cmap.csv  # fuel colormap data
'''
import contextlib
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from model.fire_model import FireSpreadModel
from model.fire_agent import FireAgent
from model.cell_agent import CellAgent

def load_fuel_cmap(csv_path):
    """
    Load fuel colormap from CSV.
    Returns:
        fuel_values: array of fuel values
        colors_rgb: array of RGB colors (0-1)
    """
    # Skip the header row
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=None, encoding=None)

    # VALUE column = index 0, R = 2, G = 3, B = 4
    fuel_values = np.array([float(row[0]) for row in data])
    colors_rgb = np.array([[int(row[2])/255, int(row[3])/255, int(row[4])/255] for row in data])

    return fuel_values, colors_rgb
    
def fuel_to_color(fuel, fuel_values, colors_rgb):
    """
    Map a fuel value to the nearest color in the colormap.
    """
    idx = np.abs(fuel_values - fuel).argmin()
    return colors_rgb[idx]

def plot_fire_grid(model, fuel_cmap_path="fuel_cmap.csv"):
    fuel_values, colors_rgb = load_fuel_cmap(fuel_cmap_path)

    # Base image: fuel colors (RGB)
    grid_rgb = np.zeros((model.rows, model.cols, 3))
    for row in range(model.rows):
        for col in range(model.cols):
            agents = model.grid.get_cell_list_contents([(col, row)])
            if not agents:
                grid_rgb[row, col] = np.array([1.0, 1.0, 1.0])  # white
                continue
            agent = agents[0]
            if not agent.burning and not agent.burned:
                grid_rgb[row, col] = fuel_to_color(agent.fuel, fuel_values, colors_rgb)
            else:
                grid_rgb[row, col] = fuel_to_color(agent.fuel, fuel_values, colors_rgb)

    # Overlay burned cells with semi-transparent red
    alpha = 0.7 # 0 is fully transparent, 1 is fully black
    for row in range(model.rows):
        for col in range(model.cols):
            agents = model.grid.get_cell_list_contents([(col, row)])
            if not agents:
                continue
            agent = agents[0]
            if getattr(agent, "burned", False):
                # Blend fuel color with black
                grid_rgb[row, col] = (1-alpha)*grid_rgb[row, col] + alpha*np.array([0.0, 0.0, 0.0])
            elif getattr(agent, "burning", False):
                # Make burning cells fully red
                grid_rgb[row, col] = np.array([1.0, 0.0, 0.0])

    plt.figure(figsize=(10, 8))
    plt.imshow(grid_rgb, origin='upper', interpolation='nearest')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tif_path = os.path.join(script_dir, "utils/resampled_main.tif")
    cmap_path = os.path.join(script_dir, "utils/fuel_cmap.csv")
    model = FireSpreadModel(tif_path)

    
    plot_fire_grid(model, fuel_cmap_path=cmap_path)
    for step in range(900):
        model.step()
        #print(f"Step {step + 1} completed")

        if step % 10 == 0:
            print(f"Plotting fire grid at step {step}")
            plot_fire_grid(model, fuel_cmap_path=cmap_path)

        # Print a random cell agent and its properties
        # rand_row = np.random.randint(0, model.rows)
        # rand_col = np.random.randint(0, model.cols)
        # test_row = 94
        # test_col = 94
        # sample_agent = model.grid.get_cell_list_contents([(test_col, test_row)])[0]
        # if sample_agent.__class__.__name__ == "FireAgent":
        #     pass
        # else:
        #     print(f"Sample Agent at ({test_row}, {test_col}) - Burning: {sample_agent.burning}, Burned: {sample_agent.burned}, Elevation: {sample_agent.elevation}, Slope: {sample_agent.slope}, Aspect: {sample_agent.aspect},Fuel: {sample_agent.fuel}, Canopy Cover: {sample_agent.canopy_cover}")
    