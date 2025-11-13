import math
import numpy as np
import heapq
from mesa import Agent

try:
    from numba import njit
    
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

from model.cell_agent import CellAgent

# Unit helpers (keep consistent with fuel_models.py)
TON_ACRE_TO_KG_M2 = 0.224170
FT_TO_M = 0.3048
BTU_LB_TO_KJ_KG = 2.326

# -----------------------------
# Numba-safe helpers
# -----------------------------
if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _fuel_waf_numba(fuel_code: int) -> float:
        # GR: 1-9, GS:10-13, SH:14-22, TU:23-27, TL:28-36, SB:37-40
        if 1 <= fuel_code <= 9:
            return 0.35
        if 10 <= fuel_code <= 13:
            return 0.30
        if 14 <= fuel_code <= 22:
            return 0.25
        if 23 <= fuel_code <= 27:
            return 0.20
        if 28 <= fuel_code <= 36:
            return 0.18
        if 37 <= fuel_code <= 40:
            return 0.22
        return 0.25

    @njit(cache=True, fastmath=True)
    def _compute_rate_of_spread_numba(
        fuel_load_kg_m2: float,
        bed_depth_m: float,
        heat_kJkg: float,
        M_f: float,
        sigma_ft_inv: float,
        slope_deg: float,
        fuel_code: int,
        wind_speed_mph_20ft: float,
        dir_factor: float,
        dead_1h: float,
        live_herb: float,
        curing_fraction: float,
    ) -> float:
        # --- Packing (SI then beta, beta_op) ---
        cured_live = curing_fraction * live_herb
        sigma_live_proxy = 1500 # ft^-1 assumed 1500-1800
        sigma_eff = (dead_1h*sigma_ft_inv + cured_live*sigma_live_proxy) / max(dead_1h + cured_live, 1e-9)
        if sigma_eff <= 0 or math.isnan(sigma_eff):
            sigma_eff = 1e-6
        rho_b_SI = fuel_load_kg_m2 / max(1e-3, bed_depth_m)
        rho_p_SI = 513.0
        beta = max(1e-6, rho_b_SI / rho_p_SI)
        beta_op = 3.348 * (sigma_eff ** -0.8189)

        # --- Albini wind coefficients (IMPERIAL) ---
        B = 0.02526 * (sigma_eff ** 0.54)
        C = 7.47 * math.exp(-0.133 * (sigma_eff ** 0.55))
        E = 0.715 * math.exp(-3.59e-4 * sigma_eff)

        # --- Wind to mid-flame ft/min ---
        waf_base = _fuel_waf_numba(fuel_code)
        canopy_cover_frac = 0.5 # in range 0 - 1 from crude Scott & Reinhardt
        canopy_cover = max(0.0, min(1.0, canopy_cover_frac))
        waf = waf_base * (1.0 - 0.6*canopy_cover)
        U20_fts = wind_speed_mph_20ft * 1.4666666667
        U_mf_fts = U20_fts * waf
        U_mf_ftmin = U_mf_fts * 60.0

        # --- Wind & slope factors ---
        ratio = max(beta / beta_op, 1e-9)
        phi_w = C * (U_mf_ftmin ** B) * (ratio ** (-E))
        phi_s = 5.275 * (beta ** -0.3) * (math.tan(math.radians(slope_deg)) ** 2)
        # print("Phi_w and Phi_s", phi_w, phi_s)
        if phi_s > 6.0:
            phi_s = 6.0

        # --- Reaction intensity (IR) in Btu/ft^2/min ---
        kg_m2_to_lb_ft2 = 0.204816143
        kJkg_to_Btulb = 0.429922614
        h_Btu_lb = heat_kJkg * kJkg_to_Btulb

        # Simple moisture damping (cap [0,1])
        x = M_f
        eta_M = 1.0 - 2.59 * x + 5.11 * (x * x) - 3.52 * (x * x * x)
        if eta_M < 0.0:
            eta_M = 0.0
        elif eta_M > 1.0:
            eta_M = 1.0

        # Available flaming fraction - use 20% of total load
        flaming_frac_avail = 0.20

        # Convert ton/acre to kg/m2 and apply flaming fraction
        # Simplified: Use fuel_load directly
        flaming_frac_avail = 0.20
        w_n_kg_m2 = fuel_load_kg_m2 * flaming_frac_avail
        w_lb_ft2 = w_n_kg_m2 * kg_m2_to_lb_ft2
        HA_Btu_ft2 = w_lb_ft2 * h_Btu_lb * eta_M
        IR_Btu_ft2_min = (HA_Btu_ft2 * sigma_ft_inv) / 384.0

        # --- Heat sink denominator (imperial) ---
        rho_b_lb_ft3 = rho_b_SI * 0.06242796
        epsilon = math.exp(-138.0 / sigma_eff)
        Q_ig_Btu_lb = 250.0 + 1116.0 * M_f

        # --- ROS (ft/min) then m/s ---
        denom = rho_b_lb_ft3 * epsilon * Q_ig_Btu_lb
        if denom < 1e-12:
            denom = 1e-12
        R_ft_min = (IR_Btu_ft2_min * (1.0 + phi_w + phi_s)) / denom
        
        # Convert to m/s and apply minimum/maximum
        R_m_s = (R_ft_min / 60.0) * 0.3048
        R_m_s *= dir_factor
        
        # Apply minimum FIRST - increased to 0.01 to prevent very late arrivals
        if R_m_s > 0 and R_m_s < 0.01:
            R_m_s = 0.01  # 10x faster than before - prevents holes
        # Then apply maximum cap
        if R_m_s > 1:
            R_m_s = 1
        return R_m_s
else:
    # Fallback pure-Python version
    def _fuel_waf_numba(fuel_code: int) -> float:
        if 1 <= fuel_code <= 9:
            return 0.35
        if 10 <= fuel_code <= 13:
            return 0.30
        if 14 <= fuel_code <= 22:
            return 0.25
        if 23 <= fuel_code <= 27:
            return 0.20
        if 28 <= fuel_code <= 36:
            return 0.18
        if 37 <= fuel_code <= 40:
            return 0.22
        return 0.25

    def _compute_rate_of_spread_numba(
        fuel_load_kg_m2: float,
        bed_depth_m: float,
        heat_kJkg: float,
        M_f: float,
        sigma_ft_inv: float,
        slope_deg: float,
        fuel_code: int,
        wind_speed_mph_20ft: float,
        dir_factor: float,
        dead_1h: float,
        live_herb: float,
        curing_fraction: float,
    ) -> float:
        # --- Packing (SI then beta, beta_op) ---
        cured_live = curing_fraction * live_herb
        sigma_live_proxy = 1500 # ft^-1 assumed 1500-1800
        sigma_eff = (dead_1h*sigma_ft_inv + cured_live*sigma_live_proxy) / max(dead_1h + cured_live, 1e-9)
        rho_b_SI = fuel_load_kg_m2 / max(1e-3, bed_depth_m)
        rho_p_SI = 513.0
        beta = max(1e-6, rho_b_SI / rho_p_SI)
        beta_op = 3.348 * (sigma_eff ** -0.8189)

        # --- Albini wind coefficients (IMPERIAL) ---
        B = 0.02526 * (sigma_eff ** 0.54)
        C = 7.47 * math.exp(-0.133 * (sigma_eff ** 0.55))
        E = 0.715 * math.exp(-3.59e-4 * sigma_eff)

        # --- Wind to mid-flame ft/min ---
        waf_base = _fuel_waf_numba(fuel_code)
        canopy_cover_frac = 0.5 # in range 0 - 1 from crude Scott & Reinhardt
        canopy_cover = max(0.0, min(1.0, canopy_cover_frac))
        waf = waf_base * (1.0 - 0.6*canopy_cover)
        U20_fts = wind_speed_mph_20ft * 1.4666666667
        U_mf_fts = U20_fts * waf
        U_mf_ftmin = U_mf_fts * 60.0

        # --- Wind & slope factors ---
        ratio = max(beta / beta_op, 1e-9)
        phi_w = C * (U_mf_ftmin ** B) * (ratio ** (-E))
        phi_s = 5.275 * (beta ** -0.3) * (math.tan(math.radians(slope_deg)) ** 2)
        # print("Phi_w and Phi_s", phi_w, phi_s)
        if phi_s > 6.0:
            phi_s = 6.0

        # --- Reaction intensity (IR) in Btu/ft^2/min ---
        kg_m2_to_lb_ft2 = 0.204816143
        kJkg_to_Btulb = 0.429922614
        h_Btu_lb = heat_kJkg * kJkg_to_Btulb

        # Simple moisture damping (cap [0,1])
        x = M_f
        eta_M = 1.0 - 2.59 * x + 5.11 * (x * x) - 3.52 * (x * x * x)
        if eta_M < 0.0:
            eta_M = 0.0
        elif eta_M > 1.0:
            eta_M = 1.0

        # Available flaming fraction - use 20% of total load
        flaming_frac_avail = 0.20

        # Convert ton/acre to kg/m2 and apply flaming fraction
        # Simplified: Use fuel_load directly
        flaming_frac_avail = 0.20
        w_n_kg_m2 = fuel_load_kg_m2 * flaming_frac_avail
        w_lb_ft2 = w_n_kg_m2 * kg_m2_to_lb_ft2
        HA_Btu_ft2 = w_lb_ft2 * h_Btu_lb * eta_M
        IR_Btu_ft2_min = (HA_Btu_ft2 * sigma_ft_inv) / 384.0

        # --- Heat sink denominator (imperial) ---
        rho_b_lb_ft3 = rho_b_SI * 0.06242796
        epsilon = math.exp(-138.0 / sigma_eff)
        Q_ig_Btu_lb = 250.0 + 1116.0 * M_f

        # --- ROS (ft/min) then m/s ---
        denom = rho_b_lb_ft3 * epsilon * Q_ig_Btu_lb
        if denom < 1e-12:
            denom = 1e-12
        R_ft_min = (IR_Btu_ft2_min * (1.0 + phi_w + phi_s)) / denom
        
        # Convert to m/s and apply minimum/maximum
        R_m_s = (R_ft_min / 60.0) * 0.3048
        R_m_s *= dir_factor
        
        # Apply minimum FIRST - increased to 0.01 to prevent very late arrivals
        if R_m_s > 0 and R_m_s < 0.01:
            R_m_s = 0.01  # 10x faster than before - prevents holes
        # Then apply maximum cap
        if R_m_s > 1:
            R_m_s = 1

        if math.isclose(beta_op, 0.0) or math.isclose(beta, 0.0):
            print(f"DEBUG: beta={beta}, beta_op={beta_op}, sigma_eff={sigma_eff}")
        if U_mf_ftmin == 0:
            print(f"DEBUG: U_mf_ftmin=0 for sigma_eff={sigma_eff}, wind_speed={wind_speed_mph_20ft}")

        return R_m_s


# -----------------------------
# Optimized FireAgent
# -----------------------------
class FireAgent(Agent):
    """
    Optimized version with proper wind direction handling (stepped factors: 1.0, 0.5, 0.25)
    """
    _OFFSETS = [
        (-1, -1),  # NW
        (-1,  0),  # N
        (-1,  1),  # NE
        ( 0, -1),  # W
        ( 0,  1),  # E
        ( 1, -1),  # SW
        ( 1,  0),  # S
        ( 1,  1),  # SE
    ]

    def __init__(self, model, unique_id,
                 fuel_load=0.5,
                 fuel_density=32.0,
                 heat_content=18600.0,
                 wind_speed=10.0,
                 slope_deg=0.0,
                 moisture_content=0.10):
        super().__init__(model)
        self.unique_id = unique_id

        # Basic properties
        self.fuel_load = float(fuel_load)
        self.fuel_density = float(fuel_density)
        self.heat_content = float(heat_content)
        self.wind_speed = float(wind_speed)
        self.slope = float(slope_deg)
        self.moisture_content = float(moisture_content)

        # Control flags
        self.wind_is_mps = False
        self.waf = 0.4  # .4 default
        self.wind_direction = 0.0 # 0 North to South wind -- based on observations
 
        # Internal flags
        self._initialized = False

        # Offset headings for 8-neighborhood (in degrees, heading = where fire is going)
        self._offset_headings = []

        for (drow, dcol) in self._OFFSETS:
            angle_rad = math.atan2(dcol, -drow)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360.0
            self._offset_headings.append(angle_deg)

        # Precompute offset distances once (in meters)
        cell_size = float(self.model.cell_size)
        self._offset_dists = np.array([
            cell_size * math.hypot(dc, dr) for (dr, dc) in self._OFFSETS
        ], dtype=np.float32)

    def _dir_factor_for_heading(self, heading_deg: float) -> float:
        """
        Return directional factor for a given heading using STEPPED factors (1.0, 0.5, 0.25).
        This matches FlamMap behavior better than cosine-based smooth gradients.
        
        wind_direction is the direction FROM which wind blows (0=North).
        heading is the direction fire spreads TO (0=North).
        
        If wind is FROM North (0°), it blows TO South (180°).
        Fire spreads fastest TO South (heading=180°), which should get factor ~1.0
        """
        wind_from = self.wind_direction
        # Convert wind FROM to wind TO direction
        wind_to = (wind_from + 180.0) % 360.0
        
        # Calculate angular difference between fire heading and wind direction
        diff = abs((heading_deg - wind_to + 180.0) % 360.0 - 180.0)
        
        # STEPPED factors (FlamMap style)
        # Head fire: within 90° of wind direction
        if diff < 90.0:
            return 1.0
        # Flank fire: 90-135° from wind
        elif diff < 135.0:
            return 0.6  # Increased from 0.5
        # Backing fire: opposite wind direction
        else:
            return 0.35  # Increased from 0.1 - allows fire to spread against wind

    def _ensure_arrays(self):
        """Initialize NumPy arrays exactly once; populates from model.cell_agents."""
        if self._initialized:
            return

        rows = self.model.rows
        cols = self.model.cols
        self._cell_matrix = np.empty((rows, cols), dtype=object)

        # Allocate arrays
        self.fuel_loads = np.zeros((rows, cols), dtype=np.float32)
        self.bed_depths = np.full((rows, cols), 0.5, dtype=np.float32)
        self.heat_contents = np.full((rows, cols), 18600.0, dtype=np.float32)
        self.moistures = np.full((rows, cols), 0.10, dtype=np.float32)
        self.slopes = np.zeros((rows, cols), dtype=np.float32)
        self.sigmas = np.full((rows, cols), 2000.0, dtype=np.float32)
        self.dead_1hs = np.zeros((rows, cols), dtype=np.float32)
        self.live_herbs = np.zeros((rows, cols), dtype=np.float32)
        self.curing_fraction = np.full((rows, cols), 0.4, dtype=np.float32)
        self.fuel_codes = np.zeros((rows, cols), dtype=np.int16)
        self.arrival_times = np.full((rows, cols), np.inf, dtype=np.float64)
        self.arrival_locked = np.zeros((rows, cols), dtype=bool)
        self.is_ignition = np.zeros((rows, cols), dtype=bool)
        self.burning_mask = np.zeros((rows, cols), dtype=bool)
        self.burned_mask = np.zeros((rows, cols), dtype=bool)

        # Populate from CellAgent instances once
        for a in self.model.cell_agents:
            if not isinstance(a, CellAgent):
                continue
            r, c = int(a.row), int(a.col)
            self._cell_matrix[r, c] = a

            # Static properties
            self.fuel_loads[r, c] = float(getattr(a, 'fuel_load', 0.5))
            self.bed_depths[r, c] = max(1e-3, float(getattr(a, 'fuel_bed_depth', 0.5)))
            self.heat_contents[r, c] = float(getattr(a, 'heat_content', 18600.0))
            self.moistures[r, c] = float(getattr(a, 'moisture_content', 0.10))
            self.slopes[r, c] = float(getattr(a, 'slope', 0.0))
            self.sigmas[r, c] = float(getattr(a, 'sav_dead_1h_per_ft', 2000.0))
            self.dead_1hs[r, c] = float(getattr(a, 'dead_1h', 0.2))
            self.live_herbs[r, c] = float(getattr(a, 'live_herb', 0))
            self.curing_fraction[r, c] = float(getattr(a, 'curing_fraction', 0.4))
            fc = getattr(a, 'fuel_code', 0)
            try:
                self.fuel_codes[r, c] = int(fc) if fc is not None else 0
            except Exception:
                self.fuel_codes[r, c] = 0

            # Dynamic state
            self.arrival_times[r, c] = float(getattr(a, 'arrival_time', np.inf))
            self.arrival_locked[r, c] = bool(getattr(a, 'arrival_locked', False))
            self.is_ignition[r, c] = bool(getattr(a, 'is_ignition', False))
            self.burning_mask[r, c] = bool(getattr(a, 'burning', False))
            self.burned_mask[r, c] = bool(getattr(a, 'burned', False))

        # Precompute dir-factors per offset
        self._dir_factors = np.array([
            self._dir_factor_for_heading(h) for h in self._offset_headings
        ], dtype=np.float32)
        
        # DEBUG: Print direction factors
        print(f"\n{'='*60}")
        print(f"Wind FROM {self.wind_direction}° at {self.wind_speed} mph")
        print(f"Direction factors:")
        offset_names = ['NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE']
        offset_names = ['NW','N','NE','W','E','SW','S','SE']
        for name, heading, factor in zip(offset_names, self._offset_headings, self._dir_factors):
            print(f"  {name:2s}: heading={heading:6.1f}°, factor={factor:.2f}")
        print(f"{'='*60}\n")

        self._initialized = True
        # Frontier & timing arrays
        self._finalized = np.zeros((rows, cols), dtype=bool)
        self._burn_end  = np.full((rows, cols), np.inf, dtype=np.float64)
        self._frontier  = []

        # Seed ignition cells
        burning_seed = np.argwhere(self.burning_mask)
        for r0, c0 in burning_seed:
            if self.is_ignition[r0, c0]:
                # This is the ignition point - set arrival time to 0 and add to frontier
                self.arrival_times[r0, c0] = 0.0
                heapq.heappush(self._frontier, (0.0, int(r0), int(c0)))
                # Seed its 8 neighbors with hardcoded times (they'll be added to frontier too)
                # self._seed_ignition_ring(int(r0), int(c0))
            else:
                # Regular burning cell
                t0 = float(self.arrival_times[r0, c0]) if np.isfinite(self.arrival_times[r0, c0]) else 0.0
                if not np.isfinite(self.arrival_times[r0, c0]):
                    self.arrival_times[r0, c0] = 0.0
                    t0 = 0.0
                heapq.heappush(self._frontier, (t0, int(r0), int(c0)))

    def compute_rate_of_spread(self, r, c, dir_k) -> float:
        """Return ROS (m/s) for neighbor cell (r_n, c_n) along direction index dir_k."""
        try:
            ros = _compute_rate_of_spread_numba(
                self.fuel_loads[r, c],
                self.bed_depths[r, c],
                self.heat_contents[r, c],
                self.moistures[r, c],
                self.sigmas[r, c],
                self.slopes[r, c],
                int(self.fuel_codes[r, c]),
                float(self.wind_speed),
                float(self._dir_factors[dir_k]),
                self.dead_1hs[r, c],
                self.live_herbs[r, c],
                self.curing_fraction[r, c],
            )
        except Exception as e:
            print(f"Numba error at cell {r},{c} dir {dir_k}: {e}")
            raise
        return ros
    def _seed_ignition_ring(self, center_row: int, center_col: int):
        """Apply the fixed arrival-time seeding around ignition cell and add to frontier."""
        fixed = {
            (-1, -1): 254.0453,
            (0, -1): 110.8131,
            (1, -1): 152.3370,
            (-1, 0): 221.0494,
            (1, 0): 92.2397,
            (-1, 1): 225.6343,
            (0, 1): 62.7920,
            (1, 1): 68.8710,
        }
        rows, cols = self.arrival_times.shape
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = center_row + dr, center_col + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    if not self.burned_mask[rr, cc] and not self.burning_mask[rr, cc] and self.fuel_loads[rr, cc] > 0:
                        tval = fixed[(dc, dr)]
                        # Force set the arrival time and lock it (override any previous value)
                        self.arrival_times[rr, cc] = tval
                        self.arrival_locked[rr, cc] = True  # Lock these initial neighbors
                        # Add to frontier
                        heapq.heappush(self._frontier, (tval, rr, cc))

    def set_wind_direction(self, wind_from_degrees: float):
        self.wind_direction = float(wind_from_degrees)
        if self._initialized:
            self._dir_factors = np.array([
                self._dir_factor_for_heading(h) for h in self._offset_headings
            ], dtype=np.float32)

    def sync_to_cell_agents(self):
        """
        Sync the numpy array state back to individual CellAgent objects.
        Call this before exporting data if you need CellAgent objects updated.
        """
        if not self._initialized:
            return
        
        for r in range(self.arrival_times.shape[0]):
            for c in range(self.arrival_times.shape[1]):
                cell = self._cell_matrix[r, c]
                if cell is not None:
                    cell.arrival_time = float(self.arrival_times[r, c])
                    cell.burning = bool(self.burning_mask[r, c])
                    cell.burned = bool(self.burned_mask[r, c])
                    cell.arrival_locked = bool(self.arrival_locked[r, c])

    def step(self):
        """
        Event-driven fire spread via a frontier (min-heap).
        """
        self._ensure_arrays()

        if not self._frontier:
            return

        # Pop earliest ignition time
        t0, r0, c0 = self._frontier[0]
        self.model.time = float(t0)

        # Process all events at time t0 (monotonic, batch)
        batch = []
        while self._frontier and abs(self._frontier[0][0] - t0) < 1e-6:
            t, r, c = heapq.heappop(self._frontier)
            if self._finalized[r, c]:
                continue
            batch.append((t, r, c))

        # Finalize arrivals for the batch and compute burn_end
        for t, r, c in batch:
            self.arrival_times[r, c] = float(t)
            self._finalized[r, c] = True
            self.burning_mask[r, c] = True
            self.burned_mask[r, c] = False

            # Cell flaming duration
            max_ros = 1e-5
            for k, (dr, dc) in enumerate(self._OFFSETS):
                rr = r + int(dr); cc = c + int(dc)
                if 0 <= rr < self.arrival_times.shape[0] and 0 <= cc < self.arrival_times.shape[1]:
                    if self.fuel_loads[rr, cc] > 0 and not self._finalized[rr, cc]:
                        ros_k = self.compute_rate_of_spread(r, c, k)
                        if ros_k > max_ros:
                            max_ros = ros_k
            tau_burn_min = (float(self.bed_depths[r, c]) / max_ros) / 60.0
            self._burn_end[r, c] = t + tau_burn_min

            # Relax neighbors from this cell (FIXED: use source cell r,c)
            for k, (dr, dc) in enumerate(self._OFFSETS):
                rr = r + int(dr); cc = c + int(dc)
                if rr < 0 or rr >= self.arrival_times.shape[0] or cc < 0 or cc >= self.arrival_times.shape[1]:
                    continue
                if self._finalized[rr, cc] or self.fuel_loads[rr, cc] <= 0 or self.burned_mask[rr, cc]:
                    continue

                R_eff = self.compute_rate_of_spread(r, c, k)
                
                # DEBUG: Track why cells don't get added to frontier
                if R_eff > 0 and self.fuel_loads[rr, cc] > 0:
                    if not hasattr(self, '_skip_reasons'):
                        self._skip_reasons = {'ros_zero': 0, 'locked': 0, 'slower': 0, 'added': 0}
                    
                    if R_eff <= 0:
                        self._skip_reasons['ros_zero'] += 1
                    elif self.arrival_locked[rr, cc]:
                        self._skip_reasons['locked'] += 1
                    else:
                        dist_m = float(self._offset_dists[k])
                        travel_s = dist_m / R_eff
                        cand_t = t + (travel_s / 60.0)
                        if cand_t >= self.arrival_times[rr, cc]:
                            self._skip_reasons['slower'] += 1
                        else:
                            self._skip_reasons['added'] += 1
                    
                    # Print summary every 10000 evaluations
                    total = sum(self._skip_reasons.values())
                    if total > 0 and total % 10000 == 0:
                        print(f"\nSpread evaluation stats after {total} attempts:")
                        print(f"  Added to frontier: {self._skip_reasons['added']}")
                        print(f"  Skipped (ROS=0): {self._skip_reasons['ros_zero']}")
                        print(f"  Skipped (locked): {self._skip_reasons['locked']}")
                        print(f"  Skipped (slower path): {self._skip_reasons['slower']}")
                
                # DEBUG: Print ROS for center cell's first spread
                if not hasattr(self, '_debug_printed') and r == self.model.rows//2 and c == self.model.cols//2:
                    offset_names = ['NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE']
                    if k == 0:
                        print(f"\n{'='*60}")
                        print(f"DEBUG: ROS from center cell ({r},{c}) at t={t:.2f}")
                    print(f"  {offset_names[k]:2s} (k={k}): ROS={R_eff:.6f} m/s, dir_factor={self._dir_factors[k]:.2f}")
                    if k == 7:
                        print(f"{'='*60}\n")
                        self._debug_printed = True
                
                if R_eff <= 0:  # Only skip truly zero ROS
                    # Defensive check: if cell is burnable, force minimum ROS
                    if self.fuel_loads[rr, cc] > 0:
                        R_eff = 0.001  # Force minimum spread rate
                        
                        # Debug: Log this case
                        if not hasattr(self, '_forced_ros_count'):
                            self._forced_ros_count = 0
                        self._forced_ros_count += 1
                        if self._forced_ros_count <= 5:
                            print(f"Cell ({rr},{cc}): ROS=0 but fuel_load={self.fuel_loads[rr,cc]:.4f}, "
                                  f"forcing to 0.001 m/s (case #{self._forced_ros_count})")
                    else:
                        continue

                dist_m   = float(self._offset_dists[k])
                travel_s = dist_m / R_eff
                cand_t = t + (travel_s / 60.0)

                # Only update if not locked (for hardcoded ignition neighbors) and if faster
                if not self.arrival_locked[rr, cc] and cand_t < self.arrival_times[rr, cc]:
                    self.arrival_times[rr, cc] = cand_t
                    heapq.heappush(self._frontier, (cand_t, int(rr), int(cc)))

        # Advance time and cool off cells whose burn_end <= t0
        # BUT only if ALL 8 neighbors are either burning or burned
        done_mask = (self._burn_end <= self.model.time) & self.burning_mask
        
        if np.any(done_mask):
            # Check each burning cell to see if all neighbors are burning/burned
            rows, cols = self.arrival_times.shape
            ready_to_burn = []
            
            for r in range(rows):
                for c in range(cols):
                    if not done_mask[r, c]:
                        continue
                    
                    # Check all 8 neighbors
                    all_neighbors_involved = True
                    for dr, dc in self._OFFSETS:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            # Check if this neighbor is burnable (has fuel)
                            if self.fuel_loads[rr, cc] > 0:
                                # If burnable, it must be burning or burned
                                if not (self.burning_mask[rr, cc] or self.burned_mask[rr, cc]):
                                    all_neighbors_involved = False
                                    break
                            # If not burnable (no fuel), we don't require it to be burning
                    
                    if all_neighbors_involved:
                        ready_to_burn.append((r, c))
            
            # Mark cells as burned only if all neighbors are involved
            for r, c in ready_to_burn:
                self.burned_mask[r, c] = True
                self.burning_mask[r, c] = False
            
            self.model.burned_count = int(np.sum(self.burned_mask))

        # Print progress throttled
        if not hasattr(self, '_last_print_time'):
            self._last_print_time = -999
        if self.model.time - self._last_print_time > 10.0:
            print(f"t={self.model.time:.2f} | burning={int(np.sum(self.burning_mask))} | burned={int(np.sum(self.burned_mask))}")
            self._last_print_time = self.model.time