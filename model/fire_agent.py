import math
import mesa

class FireAgent(mesa.Agent):
    """
    Fire agent using the Rothermel (1972) surface fire spread model
    with Albini (1976) wind and slope corrections.
    """
    def __init__(self, model, unique_id, fuel_load, fuel_density, heat_content, 
                 wind_speed, slope_deg, moisture_content):
        super().__init__(model)
        self.fuel_load = fuel_load          # w0, lb/ft^2
        self.fuel_density = fuel_density    # ρp, lb/ft^3
        self.heat_content = heat_content    # h, Btu/lb
        self.wind_speed = wind_speed        # ft/min
        self.slope_deg = slope_deg          # degrees
        self.moisture_content = moisture_content
        self.burning = False
        self.rate_of_spread = 0.0           # ft/min

        # Constants (can be refined per fuel model)
        self.xi = 0.3            # Propagating flux ratio (empirical)
        self.epsilon = 0.9       # Effective heating number
        self.Q_ig = 250.0        # Heat of preignition (Btu/lb)
        self.I_R = 100.0         # Reaction intensity (Btu/ft²/min)
        self.rho_b = 0.02        # Bulk density (lb/ft³)

        # Albini constants
        self.C = 0.045
        self.B = 2.0
        self.E = 0.715

    def compute_packing_ratio(self):
        """
        Compute actual and optimal packing ratios.
        """
        beta = self.rho_b / self.fuel_density
        beta_op = 3.348 * (beta ** 0.8189)  # approximate relation
        return beta, beta_op

    def compute_wind_factor(self, beta, beta_op):
        """
        Compute Albini (1976) wind factor φw.
        """
        phi_w = self.C * (self.wind_speed ** self.B) * ((beta / beta_op) ** -self.E)
        return phi_w

    def compute_slope_factor(self, beta):
        """
        Compute Albini (1976) slope factor φs.
        """
        slope_rad = math.radians(self.slope_deg)
        phi_s = 5.275 * (beta ** -0.3) * (math.tan(slope_rad) ** 2)
        return phi_s

    def compute_rate_of_spread(self):
        """
        Compute the final rate of spread (ft/min) using Rothermel.
        """
        beta, beta_op = self.compute_packing_ratio()
        phi_w = self.compute_wind_factor(beta, beta_op)
        phi_s = self.compute_slope_factor(beta)

        numerator = self.I_R * self.xi * (1 + phi_w + phi_s)
        denominator = self.rho_b * self.epsilon * self.Q_ig

        self.rate_of_spread = numerator / denominator

        return self.rate_of_spread, phi_w, phi_s

    def step(self):
        """
        Fire behavior update step.
        """
        if self.burning:
            R, phi_w, phi_s = self.compute_rate_of_spread()
            print(f"🔥 FireAgent {self.unique_id}: R={R:.3f} ft/min, φw={phi_w:.3f}, φs={phi_s:.3f}")
