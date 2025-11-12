"""
Comprehensive diagnostic script to analyze why ABM arrival times differ from FlamMap.
This will help identify the specific issues in the ROS calculations.
"""
import numpy as np
import pandas as pd
import os

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
abm_path = os.path.join(script_dir, "fire_arrival_times.csv")
flammap_path = os.path.join(script_dir, "flammap_arrival_times.csv")

def load_csv_with_blanks(filepath):
    """Load CSV that may have blank cells, converting blanks to inf"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            row = []
            for val in line.strip().split(','):
                if val.strip() == '':
                    row.append(np.inf)
                else:
                    row.append(float(val))
            data.append(row)
    return np.array(data)

# Load data
try:
    abm_data = load_csv_with_blanks(abm_path)
    flammap_data = load_csv_with_blanks(flammap_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Find center
r_abm, c_abm = abm_data.shape[0] // 2, abm_data.shape[1] // 2
r_fm, c_fm = flammap_data.shape[0] // 2 - 1, flammap_data.shape[1] // 2

print("="*80)
print("DIAGNOSTIC ANALYSIS: ABM vs FlamMap Arrival Times")
print("="*80)

# 1. Extract 3x3 around ignition
print("\n1. IGNITION NEIGHBORHOOD (3x3 around center)")
print("-" * 60)
abm_3x3 = abm_data[r_abm-1:r_abm+2, c_abm-1:c_abm+2]
fm_3x3 = flammap_data[r_fm-1:r_fm+2, c_fm-1:c_fm+2]

print("ABM 3x3:")
print(abm_3x3)
print("\nFlamMap 3x3:")
print(fm_3x3)
print("\nDifference (ABM - FlamMap):")
print(abm_3x3 - fm_3x3)

# Check if the 8 neighbors match
neighbors_match = []
for dr in [-1, 0, 1]:
    for dc in [-1, 0, 1]:
        if dr == 0 and dc == 0:
            continue
        abm_val = abm_3x3[1+dr, 1+dc]
        fm_val = fm_3x3[1+dr, 1+dc]
        match = abs(abm_val - fm_val) < 0.01
        neighbors_match.append(match)
        if not match:
            print(f"  Neighbor ({dr:+2d},{dc:+2d}): ABM={abm_val:8.2f}, FM={fm_val:8.2f}, Diff={abm_val-fm_val:+8.2f} ❌")
        else:
            print(f"  Neighbor ({dr:+2d},{dc:+2d}): ABM={abm_val:8.2f}, FM={fm_val:8.2f}, Diff={abm_val-fm_val:+8.2f} ✓")

if all(neighbors_match):
    print("\n✓ All 8 ignition neighbors match!")
else:
    print(f"\n❌ {sum(neighbors_match)}/8 ignition neighbors match")

# 2. Analyze second ring (16 cells around the 8 neighbors)
print("\n\n2. SECOND RING ANALYSIS (cells beyond immediate neighbors)")
print("-" * 60)
abm_5x5 = abm_data[r_abm-2:r_abm+3, c_abm-2:c_abm+3]
fm_5x5 = flammap_data[r_fm-2:r_fm+3, c_fm-2:c_fm+3]

# Get second ring cells (5x5 minus 3x3)
second_ring_abm = []
second_ring_fm = []
second_ring_pos = []

for dr in range(-2, 3):
    for dc in range(-2, 3):
        # Skip the inner 3x3
        if abs(dr) <= 1 and abs(dc) <= 1:
            continue
        abm_val = abm_5x5[2+dr, 2+dc]
        fm_val = fm_5x5[2+dr, 2+dc]
        if np.isfinite(abm_val) and np.isfinite(fm_val):
            second_ring_abm.append(abm_val)
            second_ring_fm.append(fm_val)
            second_ring_pos.append((dr, dc))

if len(second_ring_abm) > 0:
    second_ring_abm = np.array(second_ring_abm)
    second_ring_fm = np.array(second_ring_fm)
    differences = second_ring_abm - second_ring_fm
    
    print(f"Second ring cells analyzed: {len(second_ring_abm)}")
    print(f"Mean ABM arrival time: {np.mean(second_ring_abm):.2f} min")
    print(f"Mean FlamMap arrival time: {np.mean(second_ring_fm):.2f} min")
    print(f"Mean difference: {np.mean(differences):.2f} min")
    print(f"Mean absolute difference: {np.mean(np.abs(differences)):.2f} min")
    print(f"RMSE: {np.sqrt(np.mean(differences**2)):.2f} min")
    print(f"Percentage error: {(np.mean(differences)/np.mean(second_ring_fm))*100:.1f}%")
    
    # Check if ABM is consistently faster or slower
    faster_count = np.sum(differences < 0)
    slower_count = np.sum(differences > 0)
    print(f"\nABM arrives faster in {faster_count}/{len(differences)} cells ({faster_count/len(differences)*100:.1f}%)")
    print(f"ABM arrives slower in {slower_count}/{len(differences)} cells ({slower_count/len(differences)*100:.1f}%)")

# 3. Overall statistics
print("\n\n3. OVERALL STATISTICS")
print("-" * 60)
abm_finite = abm_data[np.isfinite(abm_data)]
fm_finite = flammap_data[np.isfinite(flammap_data)]

print(f"ABM burned cells: {len(abm_finite)}")
print(f"ABM mean arrival: {np.mean(abm_finite):.2f} min")
print(f"ABM median arrival: {np.median(abm_finite):.2f} min")
print(f"ABM max arrival: {np.max(abm_finite):.2f} min")

print(f"\nFlamMap burned cells: {len(fm_finite)}")
print(f"FlamMap mean arrival: {np.mean(fm_finite):.2f} min")
print(f"FlamMap median arrival: {np.median(fm_finite):.2f} min")
print(f"FlamMap max arrival: {np.max(fm_finite):.2f} min")

# 4. Direction-specific analysis
print("\n\n4. DIRECTIONAL ANALYSIS")
print("-" * 60)
print("Analyzing spread in cardinal directions from ignition...")

directions = {
    'North': (r_abm-3, c_abm),
    'South': (r_abm+3, c_abm),
    'East': (r_abm, c_abm+3),
    'West': (r_abm, c_abm-3),
    'NE': (r_abm-3, c_abm+3),
    'NW': (r_abm-3, c_abm-3),
    'SE': (r_abm+3, c_abm+3),
    'SW': (r_abm+3, c_abm-3),
}

for dir_name, (r, c) in directions.items():
    if 0 <= r < abm_data.shape[0] and 0 <= c < abm_data.shape[1]:
        abm_val = abm_data[r, c]
        # Adjust for FlamMap center offset
        r_fm_adj = r - 1
        if 0 <= r_fm_adj < flammap_data.shape[0] and 0 <= c < flammap_data.shape[1]:
            fm_val = flammap_data[r_fm_adj, c]
            if np.isfinite(abm_val) and np.isfinite(fm_val):
                diff = abm_val - fm_val
                print(f"{dir_name:5s}: ABM={abm_val:7.2f}, FM={fm_val:7.2f}, Diff={diff:+7.2f} ({diff/fm_val*100:+6.1f}%)")

# 5. Key diagnostics
print("\n\n5. KEY DIAGNOSTIC CHECKS")
print("-" * 60)

# Check if the problem is systematic
if len(abm_finite) > 0 and len(fm_finite) > 0:
    # Compare cells that burned in both
    if abm_data.shape == flammap_data.shape:
        both_burned = np.isfinite(abm_data) & np.isfinite(flammap_data)
        if np.any(both_burned):
            shared_abm = abm_data[both_burned]
            shared_fm = flammap_data[both_burned]
            shared_diff = shared_abm - shared_fm
            
            print(f"Cells burned in both models: {np.sum(both_burned)}")
            print(f"Mean difference in shared cells: {np.mean(shared_diff):.2f} min")
            
            # Check for scaling issues
            ratio = np.mean(shared_abm) / np.mean(shared_fm)
            print(f"\nRatio (ABM/FlamMap): {ratio:.3f}")
            if ratio < 0.7:
                print("  → ABM is spreading MUCH FASTER than FlamMap")
                print("  → Possible causes: ROS too high, wrong units, missing damping")
            elif ratio > 1.3:
                print("  → ABM is spreading MUCH SLOWER than FlamMap")
                print("  → Possible causes: ROS too low, incorrect fuel params, over-damping")
            elif 0.9 < ratio < 1.1:
                print("  → ABM timing is CLOSE to FlamMap (±10%)")
            else:
                print("  → ABM has MODERATE timing difference from FlamMap")

print("\n" + "="*80)
print("END OF DIAGNOSTIC ANALYSIS")
print("="*80)