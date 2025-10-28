import solara
import numpy as np
from model.fire_model import FireSpreadModel

tif_path = "./resampled_main.tif"
model = FireSpreadModel(tif_path)

# Ignite center cell
center_row = model.rows // 2
center_col = model.cols // 2
center_agent = model.grid.get_cell_list_contents([(center_row, center_col)])[0]
center_agent.burning = True

def grid_to_rgb(model):
    colors = np.zeros((model.rows, model.cols, 3), dtype=np.uint8)  # RGB
    for row in range(model.rows):
        for col in range(model.cols):
            agents = model.grid.get_cell_list_contents([(col, row)])
            if agents:
                agent = agents[0]
                if agent.burning:
                    colors[row, col] = [255, 0, 0]  # red
                elif agent.burned:
                    colors[row, col] = [0, 0, 0]    # black
                else:
                    colors[row, col] = [0, 255, 0]  # green
            else:
                colors[row, col] = [255, 255, 255]  # white
    return colors

@solara.component
def FireGrid():
    colors_array = grid_to_rgb(model)
    return solara.Image(colors_array)

@solara.component
def StepButton():
    if solara.Button("Step Fire"):
        model.step()
    return solara.Text("Step completed")

@solara.component
def FireVisualizationApp():
    return solara.VBox([
        FireGrid(),
        StepButton()
    ])


# THIS LINE fixes the Page error
Page = FireVisualizationApp
