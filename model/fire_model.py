import contextlib
import rasterio
import os
from mesa import Model
from mesa.space import MultiGrid
from model.cell_agent import CellAgent
from mesa import DataCollector
    


class FireSpreadModel(Model):
    def __init__(self, tif_path):
        super().__init__()
        self._agent_id_counter = 0
        print("Current working directory:", os.getcwd())
        print("Absolute path to TIFF:", os.path.abspath(tif_path))
        # Read raster
        with rasterio.open(tif_path) as src:
            bands = [src.read(i).astype(float) for i in range(1, src.count + 1)]

        # Assign to variables
        elevation    = bands[0]
        slope        = bands[1]
        aspect       = bands[2]
        fuel         = bands[3]
        canopy_cover = bands[4]
        self.rows, self.cols = fuel.shape


        # Create grid
        self.grid = MultiGrid(self.cols, self.rows, torus=False)

        # Create agents
        for row in range(self.rows):
            for col in range(self.cols):
                elevation_value = float(elevation[row, col])
                slope_value = float(slope[row, col])
                aspect_value = float(aspect[row, col])
                fuel_value = float(fuel[row, col])
                canopy_value = float(canopy_cover[row, col])

                agent_id = self._agent_id_counter
                agent = CellAgent(
                    self, 
                    agent_id, 
                    row, col, 
                    elevation=elevation_value, 
                    slope=slope_value, 
                    aspect=aspect_value, 
                    fuel=fuel_value, 
                    canopy_cover=canopy_value
                )
                self._agent_id_counter += 1
                self.grid.place_agent(agent, (col, row))  # (x, y) order in Mesa grids

        self.datacollector = DataCollector(
            model_reporters={"BurnedCells": lambda m: sum(a.burned for a in m.agents)},
            agent_reporters={"Burning": "burning", "Fuel": "fuel"}
        )

    def step(self):
        self.datacollector.collect(self)
        # Perform all agent steps
        self.agents.do("step")
        self.agents.do("advance")
