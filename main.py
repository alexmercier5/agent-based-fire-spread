'''
Basic framework for mesa model
Directory structure:
fire-spread/ # root directory
    ├─ .env/                 # virtual environment
    └─ .vscode/              # VSCode settings
    └─ agent-based-fire-spread/ # main model directory - GitHub repo
        ├─ main.py               # Entry point for running the model and visualizing results -- contains plotting functions and running model
        ├─ model/
        │   ├─ __init__.py
        │   ├─ fire_model.py     # Mesa Model class - instantiates the cell and fire agents and handles data collector/stepping of the model
        │   ├─ cell_agent.py     # Individual grid cell agent - includes tif layer properties, transitions cells to burned state, keeps track of arrival times
        │   └─ fire_agent.py     # Fire agent - handles fire spread logic, rate of spread calculations, tells cell agent when to start burning
        └─ utils/
            └─ comparison.py # used for analysis comparing model output to flammap output
            └─ setup.py   # functions for reading/resampling TIFF
            └─ output_utils.py  # functions for exporting model output to CSV
            └─ visualization.py # functions for setting up server based visualization: IN PROGRESS
            └─ fuel_cmap.csv  # fuel colormap data
            └─ flammap_arrival_times.csv # flammap output for comparison
            └─ original_flammap_mtts.csv # flammap output for comparison with first line labeled for columns
            └─ fire_arrival_times.csv # output file arrival times from this model
            └─ 8900main.tif # original tif file - currently using this because resampled_main.tif not running with flammap
            └─ resampled_main.tif # resampled tif file - scaled down to 100 x 100 m cell size
'''
import contextlib
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from model.fire_model import FireSpreadModel
from model.fire_agent import FireAgent
from model.cell_agent import CellAgent
from utils.output_utils import export_fire_arrival_times

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
    tif_path = os.path.join(script_dir, "utils/8900main.tif")
    # tif_path = os.path.join(script_dir, "utils/resampled_main.tif")
    cmap_path = os.path.join(script_dir, "utils/fuel_cmap.csv")
    model = FireSpreadModel(tif_path)
    saveMTTS = True

    
    #plot_fire_grid(model, fuel_cmap_path=cmap_path)
    for step in range(1500):
        model.step()
        #print(f"Step {step + 1} completed")

        if step % 100 == 0:
            print(f"Step {step + 1} complete.")
        #     print(f"Plotting fire grid at step {step}")
        #     plot_fire_grid(model, fuel_cmap_path=cmap_path)

    plot_fire_grid(model, fuel_cmap_path=cmap_path)
    if saveMTTS:
        if tif_path.endswith("resampled_main.tif"):
            output_csv = "./utils/resampled_fire_arrival_times.csv"
        else:
            output_csv = "./utils/fire_arrival_times.csv"
        export_fire_arrival_times(model, output_path=output_csv)
    print("Simulation complete.")
