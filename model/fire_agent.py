import math
import mesa
import numpy as np
from mesa import Agent
from model.cell_agent import CellAgent

'''
TODO: Include the actual fuel physical parameters from the standard FBFM data tables for each fuel type instead of generic constants.
This will require mapping FCCS or fuel values to standard parameters like fuel load, density, heat content, etc.
'''

class FireAgent(Agent):
    def __init__(self, model, unique_id, fuel_load, fuel_density, heat_content,
                 wind_speed, slope_deg, moisture_content):
        super().__init__(model)
        self.fuel_load = fuel_load
        self.fuel_density = fuel_density
        self.heat_content = heat_content
        self.wind_speed = wind_speed
        self.slope_deg = slope_deg
        self.moisture_content = moisture_content

        self.burning = True  # FireAgent is “active” by default
        self.rate_of_spread = 0.0
        self.wind_direction = 90.0  # degrees from Wind origin

        # Constants
        self.xi = 0.3 # Propagating flux ratio 0.25-0.35 for surface fuels
        self.epsilon = 0.9 # Effective heating number 0.8-1.0 for fine fuels
        self.Q_ig = 2500.0 # kJ/kg Heat of preignition 2000-3000 depending on fuel
        self.I_R = 3.0 # kJ/m^2/s Reaction intensity 10-20 for 1 kg/m^2 of fine fuel
        self.rho_b = 30.0 # 30.0 kg/m^3 Bulk density: mass of fuel per unit volume of fuel bed
        self.C = 0.045 # Wind factor constant
        self.B = 2.0 # Wind factor exponent constant
        self.E = 0.715 # Packing ratio exponent

    def compute_rate_of_spread(self, cell, neighbor):
        """
        Compute the effective rate of spread (m/s) toward a neighbor cell
        using available per-cell structure attributes (FCCS, crown height, etc.)
        and model-level parameters (heat content, moisture, etc.).
        """

        # --- Fuel and global properties ---
        fuel_load = max(1e-6, neighbor.fuel)                # kg/m²
        fuel_density = max(1e-6, self.fuel_density)          # kg/m³
        heat_content = self.heat_content                      # kJ/kg (global constant)
        moisture = min(max(self.moisture_content, 0.0), 1.0)  # fraction 0–1
        rho_b = fuel_density

        # --- Terrain & geometry ---
        dz = neighbor.elevation - cell.elevation
        dx = neighbor.col - cell.col
        dy = neighbor.row - cell.row
        dist = np.hypot(dx, dy)
        slope_angle = math.atan2(dz, dist)  # radians

        # --- Packing ratio (Rothermel) ---
        beta = rho_b / fuel_density
        beta_op = 3.348 * (beta ** 0.8189)

        # --- Wind & slope factors ---
        phi_w = self.C * (self.wind_speed ** self.B) * ((beta / beta_op) ** -self.E)
        phi_s = 5.275 * (beta ** -0.3) * (math.tan(math.radians(neighbor.slope)) ** 2)
        phi_elev = 5.275 * (beta ** -0.3) * (math.tan(slope_angle) ** 2)

        # --- Crown structure effects ---
        crown_density = max(0.01, neighbor.crown_bulk_density)  # kg/m³
        crown_base = max(0.1, neighbor.crown_base_height)       # m
        tree_height = max(1.0, neighbor.tree_height)            # m

        # Empirical: dense + low crowns promote spread
        crown_factor = (crown_density / 0.2) * (1.0 - crown_base / tree_height)
        crown_factor = max(0.0, min(crown_factor, 3.0))  # limit multiplier to 0–3×

        # --- FCCS fuel-type multiplier ---
        # FCCS typically ranges 1–150; normalize to ~0.1–3×
        FCCS_factor = max(0.1, min(neighbor.FCCS / 50.0, 3.0))

        # --- Reaction intensity (scaled by fuel and heat content) ---
        ref_fuel = 0.5       # kg/m²
        ref_heat = 18000.0   # kJ/kg
        k_m = 0.6            # moisture damping factor

        I_R_base = self.I_R
        I_R = I_R_base * (fuel_load / ref_fuel) * (heat_content / ref_heat) * (1 - k_m * moisture)
        I_R *= crown_factor * FCCS_factor
        I_R = max(0.0, I_R)

        # --- Directional factor due to wind alignment ---
        angle_to_neighbor = math.atan2(dy, dx)
        wind_angle = math.radians(self.wind_direction)
        direction_factor = max(0.1, math.cos(angle_to_neighbor - wind_angle))

        # --- Effective rate of spread (m/s) ---
        numerator = I_R * self.xi * (1 + phi_w + phi_s + phi_elev)
        denominator = rho_b * self.epsilon * self.Q_ig
        R_eff = direction_factor * numerator / denominator

        # --- Clamp to realistic bounds ---
        R_eff = max(1e-4, min(R_eff, 5.0))  # typical range for wildland fire
        return R_eff




    def step(self):
        """
        Event-driven fire spread: ignite the next cell based on earliest arrival time.
        """
        cell_size = self.model.cell_size  # in meters

        # Collect all burning cells
        burning_cells = [a for a in self.model.agents 
                        if isinstance(a, CellAgent) and a.burning]

        # Track candidate neighbor cells to ignite
        candidate_cells = []

        for cell in burning_cells:
            # Get neighbors
            neighbors = self.model.grid.get_neighbors(
                (cell.col, cell.row), moore=True, include_center=False
            )
            for n in neighbors:
                if (isinstance(n, CellAgent) and 
                    not n.burning and 
                    not n.burned and 
                    n.fuel > 0):
                    # Compute rate of spread
                    R_eff = self.compute_rate_of_spread(cell, n)
                    dx = n.col - cell.col
                    dy = n.row - cell.row
                    dist = np.hypot(dx, dy) * cell_size  # meters
                    dt_sec = dist / max(R_eff, 1e-6)      # seconds to ignite neighbor

                    # Apply boost for first-neighbor ignition if needed
                    if getattr(cell, "is_ignition", False):
                        dt_sec *= 0.5  # example: fire spreads faster to immediate neighbors

                    arrival_time = self.model.time + dt_sec
                    # Keep the earliest arrival time if multiple neighbors try to ignite the same cell
                    n.arrival_time = min(n.arrival_time, arrival_time)

                    candidate_cells.append((arrival_time, n))

        if not candidate_cells:
            # No more cells to ignite; fire is done
            return

        # Find the next cell to ignite (earliest arrival)
        next_arrival_time, next_cell = min(candidate_cells, key=lambda x: x[0])

        # Advance model time to next event
        self.model.time = next_arrival_time
        next_cell.burning = True

