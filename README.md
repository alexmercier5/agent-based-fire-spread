# Agent-Based Fire Spread Model

This repository contains an agent-based fire spread simulator built with [Mesa](https://mesa.readthedocs.io/) for Georgia Institute of Technology M.S. AE 8900 research project.  
Each landscape cell is modeled as an agent with terrain and fuel attributes derived from LANDFIRE-style raster layers.  
Fire spread is computed using Rothermel-style surface fire equations with Albini wind/slope adjustments, then compared against FlamMap outputs.

---

## Project Overview

At a high level, the workflow is:

1. **Preprocess landscape rasters**  
   Use `setup.py` to inspect and (optionally) resample multi-band GeoTIFFs (e.g., from 30 m x 30 m to 100 m x 100 m).

2. **Run the agent-based fire spread model**  
   `main.py` loads a preprocessed raster (e.g., `8900main.tif`), builds a `FireSpreadModel`, runs the simulation, and:
   - Plots fuel and fire progression.
   - Exports simulated fire-arrival times as a CSV for comparison with FlamMap.

3. **Compare ABM vs. FlamMap**  
   - `utils/output_utils.py` writes ABM arrival times to `fire_arrival_times.csv` (or a resampled version).
   - `flammap_arrival_times.csv` holds FlamMap output.  
   - Plots stored in `plots/` and `utils/comparison_plots/` visualize differences.

---

## Repository Structure & File Descriptions

### Top-Level Files

- **`main.py`**  
  Entry point for running the Mesa fire spread simulation.  
  - Loads a fuel colormap from `fuel_cmap.csv`.  
  - Instantiates `FireSpreadModel` from `model/fire_model.py` with a chosen GeoTIFF.  
  - Runs the model step loop and visualizes:
    - background fuel map (using `fuel_to_color` + colormap),
    - overlaid burned cells / fire arrival times.  
  - Optionally calls `export_fire_arrival_times` (from `utils/output_utils.py`) to write a FlamMap-compatible CSV of arrival times.

---

### `model/` – Core Mesa Model

Contains the Mesa `Model` and agent implementations.

- **`model/fire_model.py`** – `FireSpreadModel`  
  - Subclass of `mesa.Model` that:
    - Reads all 9 bands from the input GeoTIFF (elevation, slope, aspect, fuel, canopy cover, tree height, crown base height, crown bulk density, FCCS).  
    - Converts slope from percent to degrees.  
    - Creates a `CellAgent` for each grid cell and places it on a `mesa.space.MultiGrid`.  
    - Instantiates a single `FireAgent` that handles the fire spread computation over the grid.  
    - Maintains a `DataCollector` and a simple `step()` method which delegates one fire-spread time step to the `FireAgent`.  
  - Tracks `burned_count`, model time, and manages the interaction between agents and the fire-physics engine.

- **`model/cell_agent.py`** – `CellAgent`  
  - Represents a single landscape grid cell (one pixel).  
  - Stores terrain & fuel attributes:
    - elevation, slope (deg), aspect, canopy cover, tree height, crown base height, crown bulk density, FCCS code.  
    - fuel model code (LANDFIRE FBFM40) mapped via `utils.fuel_models.sb40_by_landfire`.  
  - Derives key fuel parameters (SI units):
    - `fuel_load` (kg/m²), `fuel_bed_depth` (m), `heat_content` (kJ/kg), extinction moisture fraction, surface-area-to-volume ratio, etc.  
  - Fire state variables:
    - `burning`, `burned`, `arrival_time`, `arrival_locked`, `burn_time`.  
  - `step()` logic:
    - Locks arrival times when set.  
    - Switches cells from unburned → burning → burned based on the model time and `arrival_time` computed by `FireAgent`.

- **`model/fire_agent.py`** – `FireAgent`  
  - A single agent that encapsulates the **fire physics and spread algorithm** over the grid.  
  - Responsibilities:
    - Stores arrays for each cell (fuel load, bed depth, heat content, moisture, slope, fuel code, SAV, etc.).  
    - Uses Rothermel surface fire equations with Albini wind and slope adjustments to compute a rate of spread (ROS) for each neighbor direction.  
    - Optionally uses Numba-accelerated functions when available to speed up ROS calculations.  
    - Maintains:
      - `burning_mask` and `burned_mask` grids,  
      - `arrival_times` and `arrival_locked` arrays,  
      - a frontier priority queue (min-heap) of candidate ignition times.  
    - Implements a seeded ignition ring around the ignition cell to match FlamMap’s initialization.  
    - Handles directional wind factors based on wind direction and neighbor headings, so fire spread is anisotropic (faster with the wind, slower against it).  
    - In `step()`:
      - Advances model time, pulls the next cells from the frontier, updates arrival times for neighbors, and updates which cells are burning/burned.  
      - Periodically prints debug information about burn counts, ROS, and fuel parameters.

---

### `utils/` – Utilities & Supporting Data

- **`utils/fuel_models.py`**  
  - Data module containing the Scott & Burgan 40 (**SB40**) fuel models in a unit-consistent form.  
  - Provides:
    - Base SB40 entries (`sb40_by_code`) with tons/acre, fuel bed depth (ft), extinction moisture %, heat content, etc.  
    - A derived mapping (`sb40_by_landfire`) from LANDFIRE fuel codes (e.g., `"91"`, `"92"`, etc.) to the corresponding SB40 model with precomputed SI fields:
      - `fuel_load_kg_m2`, `fuel_bed_depth_m`, `heat_content_kJ_kg`, `extinction_moisture_dead_fraction`, etc.  
  - Used by `CellAgent` to translate integer fuel codes from the raster into physical parameters used in the fire equations.

- **`utils/output_utils.py`**  
  - Contains helper functions for saving model outputs.  
  - `export_fire_arrival_times(model, output_path="fire_arrival_times.csv")`:
    - Iterates over the model grid and writes one CSV row per grid row.  
    - For each cell:
      - Writes a formatted arrival time if the cell has a finite `arrival_time`.  
      - Writes an empty string for unburned or missing data cells (so the file lines up with FlamMap’s expectations).  
    - Prints the export path when complete.

- **`utils/comparison_plots/`**  
  - Contains generated comparison plots (images) for ABM vs. FlamMap results.  
  - No core code; all files here are outputs that can be regenerated by re-running the `utils/comparison.py`.

- **`utils/setup.py`**  
  Utility script for working with GeoTIFF landscape files.  
  - Functions to **inspect** raster contents (bands, resolution, extent).  
  - `resample_tif(...)` to resample all bands of a multi-band GeoTIFF to a target cell size (e.g., 100 m), optionally snapping to a nice grid.  
  - Produces files like `resampled_main.tif` used as input to the Mesa model.

- **`utils/custom_fuelmap.csv`**  
  Lookup table mapping raw raster `VALUE` codes (e.g., LANDFIRE values) to Scott & Burgan (FBFM40) fuel models and color information. Useful when building customized fuel maps or reclassifying landscape data.

- **`utils/fuel_cmap.csv`**  
  Fuel colormap file used by `main.py` to color the background map.  
  - Columns include `VALUE`, `FBFM40`, RGB triplets, and normalized color channels.  
  - Drives how each fuel model is visualized in the plots.

- **`utils/ire_arrival_times.csv`**  
  ABM-generated fire arrival time grid exported by `utils/output_utils.export_fire_arrival_times(...)`.  
  - One row per model row, one column per model column.  
  - Burned cells have a numeric arrival time (minutes); unburned cells are left blank for FlamMap compatibility.

- **`utils/flammap_arrival_times.csv`**  
  Fire arrival time grid exported from FlamMap for the same landscape.  
  - Used as the reference dataset for comparison against the ABM output in `fire_arrival_times.csv`.

- **`utils/8900main.tif`**  
  Original multi-band landscape GeoTIFF (9 bands, ~30 m resolution).  
  - Bands include: elevation, slope, aspect, fuel model, and canopy / crown structure data.  
  - Serves as the high-resolution source file before resampling.

- **`utils/resampled_main.tif`**  
  Resampled 9-band landscape file (e.g., 100 m cells) created by `setup.py`.  
  - This is the faster raster passed into `FireSpreadModel` so the Mesa grid aligns with a coarser, simulation-friendly resolution.

---

### `plots/` – Saved Simulation Plots

- **`plots/`**  
  - A collection of plots generated when running simulations (e.g., fuel maps, fire arrival time heatmaps, side-by-side ABM vs. FlamMap comparisons).  
  - These are **outputs**, not source code, and can be safely deleted and regenerated if desired.

---

## How To Run (Quick Start)

```bash
# Create and activate a virtual environment

# Clone repository to local directory
git clone https://github.com/alexmercier5/agent-based-fire-spread

# Install required packages
pip install -r requirements.txt

# (Optional) Preprocess / resample the landscape
python setup.py  # or call resample_tif(...) from a Python session

# Run the agent-based fire spread model using a chosen raster
python main.py
