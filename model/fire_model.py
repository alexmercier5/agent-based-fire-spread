import rasterio
import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from model.cell_agent import CellAgent
from model.fire_agent import FireAgent

class FireSpreadModel(Model):
    def __init__(self, tif_path):
        super().__init__()
        self._agent_id_counter = 0
        self.time = 0
        self.fire_queue = []  # priority queue of (arrival_time, cell)

        # Read raster bands
        with rasterio.open(tif_path) as src:
            bands = [src.read(i).astype(float) for i in range(1, src.count + 1)]

        elevation    = bands[0]
        slope        = bands[1]
        aspect       = bands[2]
        fuel         = bands[3]
        canopy_cover = bands[4]
        self.rows, self.cols = fuel.shape

        # Create grid
        self.grid = MultiGrid(self.cols, self.rows, torus=False)

        # Create CellAgents
        self.cell_agents = []
        for row in range(self.rows):
            for col in range(self.cols):
                agent_id = self._agent_id_counter
                agent = CellAgent(
                    self,
                    agent_id,
                    row, col,
                    elevation=float(elevation[row, col]),
                    slope=float(slope[row, col]),
                    aspect=float(aspect[row, col]),
                    fuel=float(fuel[row, col]),
                    canopy_cover=float(canopy_cover[row, col])
                )
                self._agent_id_counter += 1
                self.cell_agents.append(agent)
                self.grid.place_agent(agent, (col, row))

        # Create FireAgent
        self.fire_agent = FireAgent(
            model=self,
            unique_id=self._agent_id_counter,
            fuel_load=0.5,
            fuel_density=32.0,
            heat_content=8000.0,
            wind_speed=0.0,
            slope_deg=0.0,
            moisture_content=0.08
        )
        self._agent_id_counter += 1

        # Ignite center cell
        center_col = self.cols // 2
        center_row = self.rows // 2
        center_cell = self.grid.get_cell_list_contents([(center_col, center_row)])[0]
        center_cell.burning = True
        center_cell.arrival_time = 0.0
        self.grid.place_agent(self.fire_agent, (center_col, center_row))

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "BurnedCells": lambda m: sum(
                    1 for a in m.cell_agents if a.burned
                )
            },
            agent_reporters={
                "Burning": lambda a: getattr(a, "burning", False),
                "Fuel": lambda a: getattr(a, "fuel", 0.0)
            }
        )

    def step(self):
        self.time += 1
        self.datacollector.collect(self)

        # Step FireAgent
        self.fire_agent.step()

        # Step all CellAgents
        for agent in self.cell_agents:
            agent.step()
