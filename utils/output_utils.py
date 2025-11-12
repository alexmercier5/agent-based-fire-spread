import numpy as np
import csv

def export_fire_arrival_times(model, output_path="fire_arrival_times.csv"):
    """
    Exports fire arrival times for each cell in the model grid.
    Burned cells show their arrival time; unburned cells are blank (for FlamMap compatibility).
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        for row in range(model.rows):
            row_values = []
            for col in range(model.cols):
                agents = model.grid.get_cell_list_contents([(col, row)])
                if not agents:
                    row_values.append("")  # blank for no data
                    continue

                agent = agents[0]
                if hasattr(agent, "arrival_time") and np.isfinite(agent.arrival_time):
                    row_values.append(f"{agent.arrival_time:.4f}")
                else:
                    row_values.append("")  # blank for unburned
            writer.writerow(row_values)

    print(f"Fire arrival times exported to {output_path}")