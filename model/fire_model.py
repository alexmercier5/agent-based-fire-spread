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
            self.cell_size = src.res[0]  # assuming square cells, in METERS

        elevation    = bands[0]
        slope        = bands[1]
        aspect       = bands[2]
        fuel         = bands[3]
        canopy_cover = bands[4]
        tree_height  = bands[5]
        crown_base_height = bands[6]
        crown_bulk_density = bands[7]
        FCCS = bands[8]
        self.rows, self.cols = fuel.shape


        self.burned_count = 0
        # Clarify slope units for downstream logic (degrees here)
        self.slope_is_percent = False

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
                    slope=float(slope[row, col]),         # DEGREES
                    aspect=float(aspect[row, col]),
                    fuel=float(fuel[row, col]),           # FBFM40 integer code
                    canopy_cover=float(canopy_cover[row, col]),
                    tree_height=float(tree_height[row, col]),
                    crown_base_height=float(crown_base_height[row, col]),
                    crown_bulk_density=float(crown_bulk_density[row, col]),
                    FCCS=float(FCCS[row, col]),
                )
                self._agent_id_counter += 1
                self.cell_agents.append(agent)
                self.grid.place_agent(agent, (col, row))

        self.fire_agent = FireAgent(
            model=self,
            unique_id=self._agent_id_counter,
            fuel_load=0.5,
            fuel_density=32.0,
            heat_content=18600.0,
            wind_speed=10.0,    # 20-ft wind in mph - default in flammap
            slope_deg=0.0,
            moisture_content=0.10
        )
        self.fire_agent.wind_is_mps = False     # we are passing mph
        self.fire_agent.waf = 0.40              # adjust per cover; try 0.25 in timber
        self.fire_agent.wind_direction = 0.0    # 0 = North to South wind
        self._agent_id_counter += 1

        # Ignite center cell
        center_col = self.cols // 2
        center_row = self.rows // 2
        center_cell = self.grid.get_cell_list_contents([(center_col, center_row)])[0]
        center_cell.burning = True
        center_cell.arrival_time = 0.0
        center_cell.is_ignition = True
        self.grid.place_agent(self.fire_agent, (center_col, center_row))

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "BurnedCells": lambda m: m.burned_count
            },
            agent_reporters={
                "Burning": lambda a: getattr(a, "burning", False),
                "Fuel": lambda a: getattr(a, "fuel", 0.0)
            }
        )

    def step(self):
        # Initialize FireAgent arrays if not already done
        if not getattr(self.fire_agent, "_initialized", False):
            self.fire_agent._ensure_arrays()

        # Collect data every 50 steps (optional throttle)
        if (getattr(self, "_tick", 0) % 50) == 0:
            self.datacollector.collect(self)
        self._tick = getattr(self, "_tick", 0) + 1

        # 🔥 Run one fire step first
        self.fire_agent.step()

        frontier = getattr(self.fire_agent, "_frontier", None)
        if frontier is not None and len(frontier) == 0:
            print(f"🔥 Fire extinguished at {self.fire_agent.model.time:.2f} min, "
                f"after {self._tick} steps.")
            return False

        return True