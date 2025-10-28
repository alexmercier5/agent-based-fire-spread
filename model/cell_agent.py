import contextlib
from mesa import Agent
import numpy as np

class CellAgent(Agent):
    def __init__(self, model, unique_id, row, col, elevation=0.0, slope=0.0, aspect=0.0, fuel=0.0, canopy_cover=0.0):
        super().__init__(model)
        self.unique_id = unique_id
        self.row = row
        self.col = col
        #self.pos = (col, row)  # Mesa grid uses (x, y)
        self.elevation = elevation
        self.slope = slope
        self.aspect = aspect
        self.fuel = fuel
        self.canopy_cover = canopy_cover

        self.burning = False
        self.burned = False
        self.arrival_time = np.inf
        self.burn_time = None
        self.rate_of_spread = 0.0
        '''
        https://owfflammaphelp62.firenet.gov/FileTypes/PU_Landscape_File.htm
        https://owfflammaphelp62.firenet.gov/AnalysisCMDs/Get_Landscape.htm

        Starting with just the necessary attributes for fire spread. Additional are below.

        self.tree_height = tree_height                  # Layer 6
        self.crown_base_height = crown_base_height      # Layer 7
        self.crown_bulk_density = crown_bulk_density    # Layer 8
        self.FCCS = FCCS                                # Layer 9
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
