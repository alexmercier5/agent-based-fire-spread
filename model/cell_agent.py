import contextlib
from mesa import Agent

class CellAgent(Agent):
    def __init__(self, model, unique_id, row, col, elevation=0.0, slope=0.0, aspect=0.0, fuel=0.0, canopy_cover=0.0):
        super().__init__(model)
        self.unique_id = unique_id
        self.row = row
        self.col = col
        self.elevation = elevation  # Layer 1
        self.slope = slope      # Layer 2
        self.aspect = aspect    # Layer 3
        self.fuel = fuel        # Layer 4
        self.canopy_cover = canopy_cover    # Layer 5
        self.burning = False    
        self.burned = False     
        self.unburned = True   
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
        if self.burning and not self.burned:
            self.burned = True
            # Ignite neighbors
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            for n in neighbors:
                if not n.burning and not n.burned and n.fuel > 0:
                    n.burning = True