import math
import mesa
import numpy as np
from mesa import Agent
from model.cell_agent import CellAgent

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
        self.wind_direction = 0.0  # No wind as default

        # Constants
        self.xi = 0.3
        self.epsilon = 0.9
        self.Q_ig = 250.0
        self.I_R = 100.0
        self.rho_b = 0.02
        self.C = 0.045
        self.B = 2.0
        self.E = 0.715

    # def compute_rate_of_spread(self):
    #     beta = self.rho_b / self.fuel_density
    #     beta_op = 3.348 * (beta ** 0.8189)
    #     phi_w = self.C * (self.wind_speed ** self.B) * ((beta / beta_op) ** -self.E)
    #     phi_s = 5.275 * (beta ** -0.3) * (math.tan(math.radians(self.slope_deg)) ** 2)
    #     numerator = self.I_R * self.xi * (1 + phi_w + phi_s)
    #     denominator = self.rho_b * self.epsilon * self.Q_ig
    #     self.rate_of_spread = numerator / denominator
    #     return self.rate_of_spread

    def compute_rate_of_spread(self, cell, neighbor):
        # Incorporate fuel and slope factors
        fuel_load = neighbor.fuel
        fuel_density = self.fuel_density  # should get these values from neighboring cells instead
        heat_content = self.heat_content
        moisture = self.moisture_content

        rho_b = fuel_density  # or another cell-specific bulk density
        beta = rho_b / fuel_density
        beta_op = 3.348 * (beta ** 0.8189)

        # Wind and slope
        phi_w = self.C * (self.wind_speed ** self.B) * ((beta / beta_op) ** -self.E)
        phi_s = 5.275 * (beta ** -0.3) * (math.tan(math.radians(neighbor.slope)) ** 2)

        # Reaction intensity can also scale with fuel load
        I_R = self.I_R * (fuel_load / 0.5)  # normalize to your reference fuel load

        # Direction relative to slope factor
        dx = neighbor.col - cell.col
        dy = neighbor.row - cell.row
        angle_to_neighbor = math.atan2(dy, dx)
        wind_angle = math.radians(self.wind_direction)
        direction_factor = max(0.1, math.cos(angle_to_neighbor - wind_angle))


        # Compute R
        numerator = I_R * self.xi * (1 + phi_w + phi_s)
        denominator = rho_b * self.epsilon * self.Q_ig
        R_eff = direction_factor * numerator / denominator

        return R_eff

    def step(self):
        """
        Spread fire from burning cells to neighbors, update arrival times.
        """
        # Loop over all grid cells
        for item in self.model.grid.coord_iter():
            # Handle both possible return formats
            if len(item) == 2:
                cell_contents, (x, y) = item
            elif len(item) == 3:
                cell_contents, x, y = item
            else:
                raise ValueError(f"Unexpected coord_iter format: {item}")

            for agent in cell_contents:
                if isinstance(agent, CellAgent) and agent.burning:
                    neighbors = self.model.grid.get_neighbors((x, y), moore=True, include_center=False)
                    for n in neighbors:
                        if isinstance(n, CellAgent) and not n.burning and not n.burned and n.fuel > 0:
                            # Compute spread based on rate_of_spread
                            R_eff = self.compute_rate_of_spread(agent, n)
                            dx = abs(n.col - agent.col)
                            dy = abs(n.row - agent.row)
                            dist = np.hypot(dx, dy)
                            dt = dist / R_eff if R_eff > 0 else float('inf')
                            arrival_time = self.model.time + dt
                            n.arrival_time = min(n.arrival_time, arrival_time)
                    

                elif isinstance(agent, CellAgent) and not agent.burning and not agent.burned and self.model.time >= agent.arrival_time:
                    agent.burning = True


                            # if arrival_time < n.arrival_time:
                            #     n.arrival_time = arrival_time
                            # if self.model.time >= n.arrival_time:
                            #     n.burning = True
