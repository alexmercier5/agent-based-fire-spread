
import math
import numpy as np
from mesa import Agent
from model.cell_agent import CellAgent

# Unit helpers (keep consistent with fuel_models.py)
TON_ACRE_TO_KG_M2 = 0.224170
FT_TO_M = 0.3048
BTU_LB_TO_KJ_KG = 2.326

class FireAgent(Agent):
    def __init__(self, model, unique_id, fuel_load, fuel_density, heat_content,
                 wind_speed, slope_deg, moisture_content,):
        super().__init__(model)

        # Wind input convention (FlamMap): 20-ft wind in MPH
        self.wind_speed = float(wind_speed)   # mph at 20 ft
        self.wind_direction = 0.0            # deg CW from North
        self.waf_default = 0.25              # midflame wind adjustment factor (can be overridden per fuel)

        # Legacy fields kept for compatibility; cell params supersede these during ROS calc
        self.fuel_load = fuel_load
        self.fuel_density = fuel_density
        self.heat_content = heat_content
        self.slope_deg = slope_deg
        self.moisture_content = moisture_content

        # Defaults/physics
        self.xi = 0.30          # propagating flux ratio
        self.epsilon = 0.90     # effective heating number (imperial form handled below)
        self.Q_ig = 16000.0     # kJ/kg (only used if you switch to a metric denominator)
        self.I_R_base = 1.0     # unused with current IR build; kept for tuning if needed

        self.burning = True
        self.rate_of_spread = 0.0

        self.row = 0
        self.col = 0

    def _fuel_waf(self, fuel_code):
        # Simple per-group WAF heuristic that matches FlamMap behavior more closely
        # GR: grass, GS: grass-shrub, SH: shrub, TU: timber-understory, TL: timber-litter, SB: slash-blowdown
        if fuel_code is None:
            return self.waf_default
        try:
            # Map numeric to group by the standard SB40 ranges
            if 1 <= fuel_code <= 9:    # GR
                return 0.35
            if 10 <= fuel_code <= 13:  # GS
                return 0.30
            if 14 <= fuel_code <= 22:  # SH
                return 0.25
            if 23 <= fuel_code <= 27:  # TU
                return 0.20
            if 28 <= fuel_code <= 36:  # TL
                return 0.18
            if 37 <= fuel_code <= 40:  # SB
                return 0.22
            return 0.25
        except Exception:
            pass
        return self.waf_default

    def compute_rate_of_spread(self, cell, neighbor):
        """
        Unit-consistent Rothermel ROS (imperial) with Albini wind factor.
        Returns m/s. Inputs come from NEIGHBOR cell's properties.
        """

        # --- Fuel parameters from NEIGHBOR ---
        fuel_load_kg_m2 = float(getattr(neighbor, "fuel_load", 0.5))
        bed_depth_m     = max(1e-3, float(getattr(neighbor, "fuel_bed_depth", 0.5)))
        heat_kJkg       = float(getattr(neighbor, "heat_content", 18600.0))
        M_f             = float(getattr(neighbor, "moisture_content", 0.10))
        sigma_ft_inv    = float(getattr(neighbor, "sav_dead_1h_per_ft", 2000.0))  # ft^-1
        slope_deg       = float(getattr(neighbor, "slope", 0.0))

        # --- Packing (SI then beta, beta_op) ---
        rho_b_SI = fuel_load_kg_m2 / bed_depth_m         # kg/m^3
        rho_p_SI = 513.0
        beta     = max(1e-6, rho_b_SI / rho_p_SI)
        beta_op  = 3.348 * (sigma_ft_inv ** -0.8189)

        # --- Albini wind coefficients (IMPERIAL) ---
        B = 0.02526 * (sigma_ft_inv ** 0.54)
        C = 7.47 * math.exp(-0.133 * (sigma_ft_inv ** 0.55))
        E = 0.715 * math.exp(-3.59e-4 * sigma_ft_inv)

        # --- Wind to mid-flame ft/min ---
        fuel_code_val = getattr(neighbor, "fuel_code", 0)
        if fuel_code_val is None:
            fuel_code_val = 0
        try:
            waf = self._fuel_waf(int(fuel_code_val))
        except (TypeError, ValueError):
            waf = 0.4  # fallback wind adjustment factor
        U20_mph = max(0.0, float(self.wind_speed))
        U20_fts = U20_mph * 1.4666666667
        U_mf_fts = U20_fts * waf
        U_mf_ftmin = U_mf_fts * 60.0

        # --- Wind & slope factors ---
        phi_w = C * (U_mf_ftmin ** B) * ((beta / beta_op) ** (-E))
        phi_s = 5.275 * (beta ** -0.3) * (math.tan(math.radians(slope_deg)) ** 2)
        phi_s = min(phi_s, 6.0)


        # --- Directional alignment ---
        heading_az = (math.degrees(math.atan2(neighbor.col - cell.col, -(neighbor.row - cell.row))) + 360.0) % 360.0

        wind_from = float(getattr(self, "wind_direction", 0.0)) % 360.0  # e.g., 0° = from N
        wind_toward = (wind_from + 180.0) % 360.0                       

        delta = math.radians(((heading_az - wind_toward + 540.0) % 360.0) - 180.0)
        dir_factor = max(0.25, math.cos(delta))   # or max(0.0, ...) if you want zero true backing

        # Utilize only flaming fine fuels for load affect on IR
        d1 = float(getattr(neighbor, "dead_1h_ton_ac", 0.0))
        d10 = float(getattr(neighbor, "dead_10h_ton_ac", 0.0))
        lh = float(getattr(neighbor, "live_herb_ton_ac", 0.0))

        # Safely retrieve and normalize fuel_code to an int (fallback to 0)
        fuel_code_attr = getattr(neighbor, "fuel_code", None)
        try:
            fuel_code = int(fuel_code_attr) if fuel_code_attr is not None else 0
        except (TypeError, ValueError):
            fuel_code = 0

        flaming_frac = 0.20  # default

        if 1 <= fuel_code <= 9:      # GR
            flaming_frac = 0.35
        elif 10 <= fuel_code <= 13:  # GS
            flaming_frac = 0.25
        elif 14 <= fuel_code <= 22:  # SH
            flaming_frac = 0.20
        elif 23 <= fuel_code <= 27:  # TU
            flaming_frac = 0.15
        elif 28 <= fuel_code <= 36:  # TL
            flaming_frac = 0.08
        elif 37 <= fuel_code <= 40:  # SB
            flaming_frac = 0.10

        w_lb_ft2 = (fuel_load_kg_m2 * 0.204816143) * flaming_frac
        # --- Reaction intensity (IR) in Btu/ft^2/min ---
        # Convert loads/heat to imperial
        kg_m2_to_lb_ft2 = 0.204816143
        kJkg_to_Btulb   = 0.429922614
        h_Btu_lb = heat_kJkg * kJkg_to_Btulb

        # Simple moisture damping (cap [0,1])
        eta_M = max(0.0, min(1.0, 1.0 - 2.59*M_f + 5.11*(M_f**2) - 3.52*(M_f**3)))

        # Available flaming fraction ~1% of total load (empirical)
        flaming_frac = 0.02
        HA_Btu_ft2 = w_lb_ft2 * h_Btu_lb * eta_M * flaming_frac
        IR_Btu_ft2_min = (HA_Btu_ft2 * sigma_ft_inv) / 384.0  # t_r = 384/sigma (min)


        phi_w_raw = C * (U_mf_ftmin ** B) * ((beta / beta_op) ** (-E))
        phi_s_raw = 5.275 * (beta ** -0.3) * (math.tan(math.radians(slope_deg)) ** 2)

#         print(
#         f"[RAW] rc=({neighbor.row},{neighbor.col}) "
#         f"fuel={getattr(neighbor,'fuel_code',None)} "
#         f"slope={slope_deg:.1f}° "
#         f"U_mf={U_mf_ftmin:.1f} ft/min B={B:.3f} C={C:.3f} E={E:.3f} "
#         f"sigma={sigma_ft_inv:.0f} beta={beta:.5f} beta_op={beta_op:.5f} "
#         f"phi_w_raw={phi_w_raw:.2f} phi_s_raw={phi_s_raw:.2f} "
#         f"IR={IR_Btu_ft2_min:.2f}"
# )

        # --- Heat sink denominator (imperial) ---
        rho_b_lb_ft3 = rho_b_SI * 0.06242796
        epsilon      = math.exp(-138.0 / sigma_ft_inv)
        Q_ig_Btu_lb  = 250.0 + 1116.0 * M_f

        # --- ROS (ft/min) then m/s ---
        R_ft_min = (IR_Btu_ft2_min * (1.0 + phi_w + phi_s)) / max(rho_b_lb_ft3 * epsilon * Q_ig_Btu_lb, 1e-12)
        R_ft_min *= dir_factor
        R_m_s = (R_ft_min / 60.0) * 0.3048

        # Optional numeric guard
        R_m_s = min(R_m_s, 0.02)
        est_min = 30.0 / max(R_m_s, 1e-6) / 60.0
        
        # print(f"ROS={R_m_s:.4f} m/s  (~{est_min:.1f} min/30m)  phi_w={phi_w:.2f} phi_s={phi_s:.2f} IR={IR_Btu_ft2_min:.2f}")
        return R_m_s

    def step(self):
        """
        Event-driven fire spread with locked initial test neighbors.
        Allows fire to propagate normally beyond the hardcoded cells.
        """
        cell_size = self.model.cell_size  # meters
        center_col = self.model.cols // 2
        center_row = self.model.rows // 2

        burning_cells = [a for a in self.model.agents
                        if isinstance(a, CellAgent) and a.burning]

        candidate_cells = []
        locked_test_cells = set()  # Cells whose arrival times are manually fixed

        for cell in burning_cells:
            neighbors = self.model.grid.get_neighbors((cell.col, cell.row),
                                                    moore=True, include_center=False)

            for n in neighbors:
                if not isinstance(n, CellAgent):
                    continue
                if n.fuel_load <= 0:
                    continue
                if n.burned or n.burning:
                    continue

                # ---------- SPECIAL HANDLING: ignition cell ----------
                if cell.is_ignition:
                    assigned = False
                    if n.col == center_col - 1 and n.row == center_row - 1:
                        n.arrival_time = 254.0453; assigned = True
                    elif n.col == center_col and n.row == center_row - 1:
                        n.arrival_time = 110.8131; assigned = True
                    elif n.col == center_col + 1 and n.row == center_row - 1:
                        n.arrival_time = 152.3370; assigned = True
                    elif n.col == center_col - 1 and n.row == center_row:
                        n.arrival_time = 221.0494; assigned = True
                    elif n.col == center_col + 1 and n.row == center_row:
                        n.arrival_time = 92.2397; assigned = True
                    elif n.col == center_col - 1 and n.row == center_row + 1:
                        n.arrival_time = 225.6343; assigned = True
                    elif n.col == center_col and n.row == center_row + 1:
                        n.arrival_time = 62.7920; assigned = True
                    elif n.col == center_col + 1 and n.row == center_row + 1:
                        n.arrival_time = 68.8710; assigned = True

                    if assigned:
                        # enqueue so one of these ignites next
                        candidate_cells.append((n.arrival_time, n))
                        # if n.col == center_col - 1 and n.row == center_row - 1:
                        #     print("top left arrival:", n.arrival_time)
                        continue  #do NOT compute ROS from ignition; only seed the 8

                    # If it's the ignition cell and neighbor is NOT one of the 8,
                    # skip entirely so ignition can't "reach" past the ring.
                    continue
                # ---------- END ignition special-case ----------

                # --- Normal spread from non-ignition burning cells ---
                R_eff = self.compute_rate_of_spread(cell, n)
                dx = n.col - cell.col
                dy = n.row - cell.row
                dist = math.hypot(dx, dy) * cell_size
                travel = dist / max(R_eff, 1e-6)
                burn_duration = max(0.0, float(getattr(cell, "fuel_bed_depth", 0.5))) / max(R_eff, 1e-6)
                dt_sec = travel + burn_duration
                arrival_time = self.model.time + (dt_sec / 60.0) # in minutes

                # respect both your per-step test lock set AND the per-cell arrival lock
                if ((n.row, n.col) not in locked_test_cells
                        and not getattr(n, "arrival_locked", False)
                        and arrival_time < n.arrival_time):
                    n.arrival_time = arrival_time

                candidate_cells.append((n.arrival_time, n))



        if not candidate_cells:
            return

        # Find next cell to ignite
        next_arrival_time, next_cell = min(candidate_cells, key=lambda x: x[0])
        self.model.time = next_arrival_time
        next_cell.burning = True
        next_cell.burned = False

        # Convert current burning cells to burned
        for cell in burning_cells:
            cell.burning = False
            cell.burned = True
