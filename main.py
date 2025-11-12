'''
Basic framework for mesa model
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from model.fire_model import FireSpreadModel
from utils.output_utils import export_fire_arrival_times

def load_fuel_cmap(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=None, encoding=None)
    fuel_values = np.array([float(row[0]) for row in data])
    colors_rgb = np.array([[int(row[2])/255, int(row[3])/255, int(row[4])/255] for row in data])
    return fuel_values, colors_rgb
    
def fuel_to_color(fuel, fuel_values, colors_rgb):
    idx = np.abs(fuel_values - fuel).argmin()
    return colors_rgb[idx]

def plot_fire_grid(model, fuel_cmap_path="fuel_cmap.csv"):
    fuel_values, colors_rgb = load_fuel_cmap(fuel_cmap_path)
    grid_rgb = np.zeros((model.rows, model.cols, 3))
    
    for row in range(model.rows):
        for col in range(model.cols):
            agents = model.grid.get_cell_list_contents([(col, row)])
            if not agents:
                grid_rgb[row, col] = np.array([1.0, 1.0, 1.0])
                continue
            agent = agents[0]
            grid_rgb[row, col] = fuel_to_color(agent.fuel, fuel_values, colors_rgb)

    alpha = 0.7
    for row in range(model.rows):
        for col in range(model.cols):
            agents = model.grid.get_cell_list_contents([(col, row)])
            if not agents:
                continue
            agent = agents[0]
            if getattr(agent, "burned", False):
                grid_rgb[row, col] = (1-alpha)*grid_rgb[row, col] + alpha*np.array([0.0, 0.0, 0.0])
            elif getattr(agent, "burning", False):
                grid_rgb[row, col] = np.array([1.0, 0.0, 0.0])

    grid_rgb = np.clip(grid_rgb, 0.0, 1.0)
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

    print(f"Starting simulation with {model.rows}x{model.cols} = {model.rows * model.cols} total cells")
    print("Running until fire extinguishes naturally...")
    
    step = 0
    max_steps = 20000
    
    while step < max_steps:
        cont = model.step()
        
        if step % 100 == 0:
            burned = model.burned_count
            burning = int(np.sum(model.fire_agent.burning_mask)) if hasattr(model, 'fire_agent') else 0
            print(f"Step {step}: t={model.time:.2f} min | Burning: {burning} | Burned: {burned}")
        
        if cont is False:
            print(f"\n Fire extinguished at step {step}, time {model.time:.2f} min")
            break
        
        step += 1
    
    if step >= max_steps:
        print(f"\n⚠️  Reached maximum step limit ({max_steps})")
    
    final_burned = model.burned_count
    final_burning = int(np.sum(model.fire_agent.burning_mask)) if hasattr(model, 'fire_agent') else 0
    total_affected = final_burned + final_burning
    percent_burned = (total_affected / (model.rows * model.cols)) * 100
    
    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total steps: {step}")
    print(f"Simulation time: {model.time:.2f} minutes ({model.time/60:.2f} hours)")
    print(f"Cells burned: {final_burned}")
    print(f"Cells burning: {final_burning}")
    print(f"Total affected: {total_affected} ({percent_burned:.2f}%)")
    print(f"{'='*60}\n")

    if hasattr(model, 'fire_agent'):
        print("Syncing FireAgent data to CellAgent objects for visualization...")
        model.fire_agent.sync_to_cell_agents()
    
    plot_fire_grid(model, fuel_cmap_path=cmap_path)
    
    if saveMTTS:
        if tif_path.endswith("resampled_main.tif"):
            output_csv = "./utils/resampled_fire_arrival_times.csv"
        else:
            output_csv = "./utils/fire_arrival_times.csv"
        export_fire_arrival_times(model, output_path=output_csv)
    print("Simulation complete.")