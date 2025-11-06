import contextlib
from mesa import Agent
import numpy as np

class CellAgent(Agent):
    def __init__(self, 
                 model, 
                 unique_id, 
                 row, 
                 col, 
                 elevation=0.0, 
                 slope=0.0, 
                 aspect=0.0, 
                 fuel=0.0, 
                 canopy_cover=0.0,
                 tree_height=0.0,
                 crown_base_height=0.0,
                 crown_bulk_density=0.0,
                 FCCS=0.0):
        super().__init__(model)
        self.unique_id = unique_id
        self.row = row
        self.col = col
        self.elevation = elevation                      # Layer 1
        self.slope = slope                              # Layer 2
        self.aspect = aspect                            # Layer 3
        self.fuel = fuel                                # Layer 4
        self.canopy_cover = canopy_cover                # Layer 5
        self.tree_height = tree_height                  # Layer 6
        self.crown_base_height = crown_base_height      # Layer 7
        self.crown_bulk_density = crown_bulk_density    # Layer 8
        self.FCCS = FCCS                                # Layer 9

        self.burning = False
        self.burned = False
        self.arrival_time = np.inf
        self.burn_time = None
        self.rate_of_spread = 0.0
        self.is_ignition = False
        '''
        https://owfflammaphelp62.firenet.gov/FileTypes/PU_Landscape_File.htm
        https://owfflammaphelp62.firenet.gov/AnalysisCMDs/Get_Landscape.htm
        '''

    def step(self):
        # Just update burned state if currently burning
        if self.burning and not self.burned:
            if self.burn_time is None:
                self.burn_time = self.model.time
            elif self.model.time > self.burn_time:
                self.burned = True
                self.burning = False
        elif not self.burning and not self.burned and self.model.time >= self.arrival_time:
            self.burning = True
