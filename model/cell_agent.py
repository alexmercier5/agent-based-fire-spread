from mesa import Agent
import numpy as np
from utils.fuel_models import sb40_by_landfire

class CellAgent(Agent):
    """
    Represents a single landscape cell for fire spread.
    Each cell stores terrain, canopy, and fuel parameters (Scott & Burgan 40).
    """

    def __init__(self, model, unique_id, row, col,
                 elevation=0.0,
                 slope=0.0,
                 aspect=0.0,
                 fuel=0.0,               # Layer 4 = LANDFIRE FBFM40 integer code
                 canopy_cover=0.0,
                 tree_height=0.0,
                 crown_base_height=0.0,
                 crown_bulk_density=0.0,
                 FCCS=0.0):
        super().__init__(model)
        self.unique_id = unique_id
        self.row = row
        self.col = col

        self.arrival_locked = False

        # --- Base landscape attributes ---
        self.elevation = float(elevation)
        self.slope = float(slope)
        self.aspect = float(aspect)
        self.fuel = float(fuel)
        self.canopy_cover = float(canopy_cover)
        self.tree_height = float(tree_height)
        self.crown_base_height = float(crown_base_height)
        self.crown_bulk_density = float(crown_bulk_density)
        self.FCCS = float(FCCS)

        # --- Fuel mapping (Scott & Burgan 40) ---
        if not np.isnan(self.fuel):
            self.fuel_code = str(int(self.fuel))
        else:
            self.fuel_code = str(91)
        
        # For non-burnable fuel types, use GR2 (moderate grassland) as default
        # This allows fire to spread over water, roads, urban areas, etc.
        if self.fuel_code not in sb40_by_landfire:
            # Replace with GR2 (LANDFIRE code 92)
            self.fuel_code = "92"
            self.original_fuel_code = str(int(self.fuel)) if not np.isnan(self.fuel) else "0"
        
        if self.fuel_code in sb40_by_landfire:
            f = sb40_by_landfire[self.fuel_code]

            # Primary SI fields
            self.fuel_load = float(f["fuel_load_kg_m2"])             # kg/m²
            self.fuel_bed_depth = float(f["fuel_bed_depth_m"])       # m
            self.heat_content = float(f["heat_content_kJ_kg"])       # kJ/kg
            self.moisture_content = f["extinction_moisture_dead_fraction"] # percent (10% = 0.1)
            self.sav_dead_1h_per_ft = float(f["sav_1h_ft_inv"])      # ft⁻¹
            self.dead_1h = float(f["dead_1h_ton_ac"])       # ton/acre
            self.live_herb = float(f["total_live_ton_ac"])  # ton/acre

            # Derived bulk density (kg/m³)
            self.fuel_density = self.fuel_load / max(self.fuel_bed_depth, 1e-3)

        else:
            # This should never happen now, but keep as safety fallback
            self.fuel_load = 0.5                 # kg/m²
            self.fuel_bed_depth = 0.5            # m
            self.heat_content = 18600.0          # kJ/kg
            self.moisture_content = 0.10         # fraction
            self.sav_dead_1h_per_ft = 2000.0     # ft⁻¹
            self.fuel_density = 30.0             # kg/m³
            self.fuel_code = "92"  # GR2
            self.dead_1h = 0.1                   # ton/acre (GR2 value)
            self.live_herb = 1.0                 # ton/acre (GR2 value)

        # --- Fire state variables ---
        self.burning = False
        self.burned = False
        self.arrival_time = np.inf
        self.burn_time = None
        self.rate_of_spread = 0.0
        self.curing_fraction = 0.4 # live herb curing fraction - 0.3-0.6
        self.is_ignition = False
        # All cells are now burnable since we replace non-burnable fuels with GR2
        self.is_burnable = True

    def step(self):
        """
        Update burning/burned state each simulation tick.
        """
        if self.burning and not self.burned:
            if self.burn_time is None:
                self.burn_time = self.model.time
            elif self.model.time > self.burn_time:
                self.burned = True
                self.burning = False
        elif not self.burning and not self.burned:
            if not np.isinf(self.arrival_time) and not self.arrival_locked:
                self.arrival_locked = True

            # Ignite only when it's time
            if self.arrival_locked and self.model.time >= self.arrival_time:
                self.burning = True