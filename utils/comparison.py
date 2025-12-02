import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.interpolate import griddata
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

'''
Enhanced Fire Arrival Time Comparison Script - LOCAL VERSION
Compares ABM (Agent-Based Model) and FlamMap MTT arrival times
ABM times are in seconds, FlamMap times are in minutes
'''

# File paths - looks in current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
abm_path = os.path.join(script_dir, "fire_arrival_times.csv")
flammap_path = os.path.join(script_dir, "flammap_arrival_times.csv")

# Create output directory if it doesn't exist
output_dir = os.path.join(script_dir, "comparison_outputs")
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("FIRE ARRIVAL TIME COMPARISON: ABM vs FlamMap")
print("="*80)
print(f"\nOutput directory: {output_dir}")

# Load data
abm_data_raw = pd.read_csv(abm_path, header=None).to_numpy()
flammap_data = pd.read_csv(flammap_path, header=None).to_numpy()

# Convert ABM from seconds to minutes for comparison
abm_data = abm_data_raw / 60.0

# Handle invalid values
abm_data = np.nan_to_num(abm_data, nan=0, posinf=0, neginf=0)
flammap_data = np.nan_to_num(flammap_data, nan=0, posinf=0, neginf=0)

print(f"\nData Shapes:")
print(f"  ABM:     {abm_data.shape}")
print(f"  FlamMap: {flammap_data.shape}")

# Get center coordinates
r1, c1 = abm_data.shape[0] // 2, abm_data.shape[1] // 2
r2, c2 = flammap_data.shape[0] // 2, flammap_data.shape[1] // 2

print(f"\nIgnition Points (Center):")
print(f"  ABM:     ({r1}, {c1}) - Arrival Time: {abm_data[r1, c1]:.2f} min")
print(f"  FlamMap: ({r2}, {c2}) - Arrival Time: {flammap_data[r2, c2]:.2f} min")

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)

# Create masks for valid (burned) cells
abm_mask = abm_data > 0
flammap_mask = flammap_data > 0

abm_valid = abm_data[abm_mask]
flammap_valid = flammap_data[flammap_mask]

print(f"\nBurned Area:")
print(f"  ABM:     {np.sum(abm_mask)} cells ({np.sum(abm_mask)/abm_data.size*100:.2f}%)")
print(f"  FlamMap: {np.sum(flammap_mask)} cells ({np.sum(flammap_mask)/flammap_data.size*100:.2f}%)")

print(f"\nArrival Time Statistics (minutes):")
print(f"  {'Metric':<15} {'ABM':>12} {'FlamMap':>12} {'Difference':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'Mean':<15} {np.mean(abm_valid):>12.2f} {np.mean(flammap_valid):>12.2f} {np.mean(abm_valid)-np.mean(flammap_valid):>12.2f}")
print(f"  {'Median':<15} {np.median(abm_valid):>12.2f} {np.median(flammap_valid):>12.2f} {np.median(abm_valid)-np.median(flammap_valid):>12.2f}")
print(f"  {'Std Dev':<15} {np.std(abm_valid):>12.2f} {np.std(flammap_valid):>12.2f} {np.std(abm_valid)-np.std(flammap_valid):>12.2f}")
print(f"  {'Min':<15} {np.min(abm_valid):>12.2f} {np.min(flammap_valid):>12.2f} {np.min(abm_valid)-np.min(flammap_valid):>12.2f}")
print(f"  {'Max':<15} {np.max(abm_valid):>12.2f} {np.max(flammap_valid):>12.2f} {np.max(abm_valid)-np.max(flammap_valid):>12.2f}")
print(f"  {'25th %ile':<15} {np.percentile(abm_valid, 25):>12.2f} {np.percentile(flammap_valid, 25):>12.2f} {np.percentile(abm_valid, 25)-np.percentile(flammap_valid, 25):>12.2f}")
print(f"  {'75th %ile':<15} {np.percentile(abm_valid, 75):>12.2f} {np.percentile(flammap_valid, 75):>12.2f} {np.percentile(abm_valid, 75)-np.percentile(flammap_valid, 75):>12.2f}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Find overlapping burned areas
common_mask = abm_mask & flammap_mask
abm_common = abm_data[common_mask]
flammap_common = flammap_data[common_mask]

if len(abm_common) > 0:
    correlation = np.corrcoef(abm_common, flammap_common)[0, 1]
    r_squared = correlation ** 2
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(flammap_common, abm_common)
    
    print(f"\nCommon Burned Cells: {len(abm_common)} ({len(abm_common)/max(len(abm_valid), len(flammap_valid))*100:.2f}%)")
    print(f"Pearson Correlation: {correlation:.4f}")
    print(f"R-squared:          {r_squared:.4f}")
    print(f"Linear Fit:         ABM = {slope:.4f} * FlamMap + {intercept:.4f}")
    print(f"P-value:            {p_value:.2e}")
    
    # Mean absolute error
    mae = np.mean(np.abs(abm_common - flammap_common))
    rmse = np.sqrt(np.mean((abm_common - flammap_common)**2))
    
    print(f"\nError Metrics:")
    print(f"  MAE (Mean Absolute Error):  {mae:.2f} minutes")
    print(f"  RMSE (Root Mean Sq Error):  {rmse:.2f} minutes")

# ============================================================================
# DIRECTIONAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("DIRECTIONAL SPREAD ANALYSIS")
print("="*80)

def analyze_direction(data, center_r, center_c, direction_name, row_slice, col_slice):
    """Analyze fire spread in a specific direction"""
    region = data[row_slice, col_slice]
    valid = region[region > 0]
    if len(valid) > 0:
        return {
            'name': direction_name,
            'mean': np.mean(valid),
            'max': np.max(valid),
            'cells': len(valid)
        }
    return None

# Define directions (N, S, E, W, NE, NW, SE, SW)
directions = [
    ('North', slice(0, r1), slice(c1-10, c1+10)),
    ('South', slice(r1, None), slice(c1-10, c1+10)),
    ('East', slice(r1-10, r1+10), slice(c1, None)),
    ('West', slice(r1-10, r1+10), slice(0, c1)),
    ('NE', slice(0, r1), slice(c1, None)),
    ('NW', slice(0, r1), slice(0, c1)),
    ('SE', slice(r1, None), slice(c1, None)),
    ('SW', slice(r1, None), slice(0, c1))
]

print(f"\n{'Direction':<10} {'ABM Mean':>10} {'FlamMap Mean':>12} {'Difference':>12}")
print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*12}")

for direction_name, row_slice, col_slice in directions:
    abm_dir = analyze_direction(abm_data, r1, c1, direction_name, row_slice, col_slice)
    flammap_dir = analyze_direction(flammap_data, r2, c2, direction_name, row_slice, col_slice)
    
    if abm_dir and flammap_dir:
        diff = abm_dir['mean'] - flammap_dir['mean']
        print(f"{direction_name:<10} {abm_dir['mean']:>10.2f} {flammap_dir['mean']:>12.2f} {diff:>12.2f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS...")
print("="*80)

# Figure 1: Side-by-side heatmaps
fig1 = plt.figure(figsize=(16, 7))
gs1 = GridSpec(1, 3, figure=fig1, width_ratios=[1, 1, 0.05])

# ABM heatmap
ax1 = fig1.add_subplot(gs1[0])
im1 = ax1.imshow(abm_data, cmap='hot', origin='upper', aspect='auto')
ax1.set_title('ABM Fire Arrival Times', fontsize=14, fontweight='bold')
ax1.set_xlabel('Column Index')
ax1.set_ylabel('Row Index')
ax1.plot(c1, r1, 'b*', markersize=15, label='Ignition Point')
ax1.legend()

# FlamMap heatmap
ax2 = fig1.add_subplot(gs1[1])
im2 = ax2.imshow(flammap_data, cmap='hot', origin='upper', aspect='auto')
ax2.set_title('FlamMap Fire Arrival Times', fontsize=14, fontweight='bold')
ax2.set_xlabel('Column Index')
ax2.set_ylabel('Row Index')
ax2.plot(c2, r2, 'b*', markersize=15, label='Ignition Point')
ax2.legend()

# Shared colorbar
cbar_ax = fig1.add_subplot(gs1[2])
vmin = min(np.min(abm_valid), np.min(flammap_valid))
vmax = max(np.max(abm_valid), np.max(flammap_valid))
fig1.colorbar(im1, cax=cbar_ax, label='Arrival Time')
leg1 = ax1.legend()
for text in leg1.get_texts():
    text.set_color("red")

leg2 = ax2.legend()
for text in leg2.get_texts():
    text.set_color("red")

plt.tight_layout()
output_path = os.path.join(output_dir, '1_heatmaps_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# Figure 2: Difference map
fig2, ax = plt.subplots(figsize=(10, 8))
difference = abm_data - flammap_data
diff_masked = np.ma.masked_where(~common_mask, difference)
im = ax.imshow(diff_masked, cmap='RdBu_r', origin='upper', vmin=-50, vmax=50, aspect='auto')
ax.set_title('Difference Map: ABM - FlamMap (minutes)\nRed = FlamMap Faster, Blue = ABM Faster', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Column Index')
ax.set_ylabel('Row Index')
cbar = plt.colorbar(im, ax=ax, label='Time Difference (minutes)')
ax.plot(c1, r1, 'k*', markersize=15, label='ABM Ignition')
ax.plot(c2, r2, 'g*', markersize=15, label='FlamMap Ignition')
ax.legend()
plt.tight_layout()
output_path = os.path.join(output_dir, '2_difference_map.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# Figure 3: Scatter plot with regression
if len(abm_common) > 0:
    fig3, ax = plt.subplots(figsize=(10, 8))
    
    # Sample points for clearer visualization if too many
    n_points = min(len(abm_common), 5000)
    indices = np.random.choice(len(abm_common), n_points, replace=False)
    
    ax.scatter(flammap_common[indices], abm_common[indices], alpha=0.3, s=10, label='Data Points')
    
    # Add regression line
    x_line = np.linspace(flammap_common.min(), flammap_common.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Linear Fit (R²={r_squared:.4f})')
    
    # Add perfect agreement line
    ax.plot([0, max(flammap_common.max(), abm_common.max())], 
            [0, max(flammap_common.max(), abm_common.max())], 
            'k--', linewidth=1, alpha=0.5, label='Perfect Agreement')
    
    ax.set_xlabel('FlamMap Arrival Time (minutes)', fontsize=12)
    ax.set_ylabel('ABM Arrival Time (minutes)', fontsize=12)
    ax.set_title('ABM vs FlamMap Arrival Times\n(Common Burned Cells Only)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = f'N = {len(abm_common)}\nR² = {r_squared:.4f}\nMAE = {mae:.2f} min\nRMSE = {rmse:.2f} min'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3_scatter_regression.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

# Figure 4: Distribution comparison
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(abm_valid, bins=50, alpha=0.6, label='ABM', density=True, color='blue')
axes[0, 0].hist(flammap_valid, bins=50, alpha=0.6, label='FlamMap', density=True, color='red')
axes[0, 0].set_xlabel('Arrival Time (minutes)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Distribution of Arrival Times', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Cumulative distribution
sorted_abm = np.sort(abm_valid)
sorted_flammap = np.sort(flammap_valid)
cdf_abm = np.arange(1, len(sorted_abm) + 1) / len(sorted_abm)
cdf_flammap = np.arange(1, len(sorted_flammap) + 1) / len(sorted_flammap)

axes[0, 1].plot(sorted_abm, cdf_abm, label='ABM', linewidth=2, color='blue')
axes[0, 1].plot(sorted_flammap, cdf_flammap, label='FlamMap', linewidth=2, color='red')
axes[0, 1].set_xlabel('Arrival Time (minutes)')
axes[0, 1].set_ylabel('Cumulative Probability')
axes[0, 1].set_title('Cumulative Distribution Function', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Box plots
box_data = [abm_valid, flammap_valid]
bp = axes[1, 0].boxplot(box_data, tick_labels=['ABM', 'FlamMap'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
axes[1, 0].set_ylabel('Arrival Time (minutes)')
axes[1, 0].set_title('Box Plot Comparison', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Q-Q plot
if len(abm_common) > 0:
    # Sample for Q-Q plot if too many points
    n_qq = min(len(abm_common), 1000)
    qq_indices = np.random.choice(len(abm_common), n_qq, replace=False)
    
    quantiles = np.linspace(0, 1, n_qq)
    abm_quantiles = np.quantile(abm_common[qq_indices], quantiles)
    flammap_quantiles = np.quantile(flammap_common[qq_indices], quantiles)
    
    axes[1, 1].scatter(flammap_quantiles, abm_quantiles, alpha=0.5, s=20)
    axes[1, 1].plot([0, max(flammap_quantiles.max(), abm_quantiles.max())],
                     [0, max(flammap_quantiles.max(), abm_quantiles.max())],
                     'r--', linewidth=2, label='Perfect Agreement')
    axes[1, 1].set_xlabel('FlamMap Quantiles (minutes)')
    axes[1, 1].set_ylabel('ABM Quantiles (minutes)')
    axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(output_dir, '4_distribution_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# Figure 5: Radial analysis (spread from center)
fig5, ax = plt.subplots(figsize=(10, 8))

# Calculate radial distance and average arrival time
def radial_analysis(data, center_r, center_c):
    rows, cols = np.indices(data.shape)
    distances = np.sqrt((rows - center_r)**2 + (cols - center_c)**2)
    
    # Bin by distance
    max_dist = int(min(center_r, center_c, data.shape[0]-center_r, data.shape[1]-center_c))
    bins = np.arange(0, max_dist, 5)
    
    avg_times = []
    std_times = []
    bin_centers = []
    
    for i in range(len(bins)-1):
        mask = (distances >= bins[i]) & (distances < bins[i+1]) & (data > 0)
        if np.sum(mask) > 0:
            avg_times.append(np.mean(data[mask]))
            std_times.append(np.std(data[mask]))
            bin_centers.append((bins[i] + bins[i+1]) / 2)
    
    return np.array(bin_centers), np.array(avg_times), np.array(std_times)

abm_dist, abm_time, abm_std = radial_analysis(abm_data, r1, c1)
flammap_dist, flammap_time, flammap_std = radial_analysis(flammap_data, r2, c2)

ax.plot(abm_dist, abm_time, 'o-', linewidth=2, label='ABM', color='blue', markersize=6)
ax.fill_between(abm_dist, abm_time - abm_std, abm_time + abm_std, alpha=0.2, color='blue')

ax.plot(flammap_dist, flammap_time, 's-', linewidth=2, label='FlamMap', color='red', markersize=6)
ax.fill_between(flammap_dist, flammap_time - flammap_std, flammap_time + flammap_std, alpha=0.2, color='red')

ax.set_xlabel('Distance from Ignition Point (cells)', fontsize=12)
ax.set_ylabel('Average Arrival Time (minutes)', fontsize=12)
ax.set_title('Radial Fire Spread Analysis\n(Shaded area = ±1 std dev)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(output_dir, '5_radial_spread_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# Figure 6: Error analysis
fig6, axes = plt.subplots(2, 2, figsize=(14, 10))

if len(abm_common) > 0:
    errors = abm_common - flammap_common
    abs_errors = np.abs(errors)
    rel_errors = 100 * errors / (flammap_common + 1e-6)  # Avoid division by zero
    
    # Error histogram
    axes[0, 0].hist(errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0, 0].axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean={np.mean(errors):.2f}')
    axes[0, 0].set_xlabel('Error: ABM - FlamMap (minutes)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Error Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Absolute error vs FlamMap time
    axes[0, 1].scatter(flammap_common, abs_errors, alpha=0.3, s=10)
    axes[0, 1].set_xlabel('FlamMap Arrival Time (minutes)')
    axes[0, 1].set_ylabel('Absolute Error (minutes)')
    axes[0, 1].set_title('Absolute Error vs Arrival Time', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add moving average
    sorted_indices = np.argsort(flammap_common)
    window = len(flammap_common) // 20
    if window > 10:
        moving_avg = np.convolve(abs_errors[sorted_indices], 
                                 np.ones(window)/window, mode='valid')
        axes[0, 1].plot(flammap_common[sorted_indices][window-1:], 
                       moving_avg, 'r-', linewidth=2, label='Moving Avg')
        axes[0, 1].legend()
    
    # Relative error histogram
    rel_errors_clipped = np.clip(rel_errors, -200, 200)  # Clip for better visualization
    axes[1, 0].hist(rel_errors_clipped, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Relative Error (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Relative Error Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Error spatial pattern
    error_grid = np.zeros_like(abm_data)
    error_grid[common_mask] = errors
    error_masked = np.ma.masked_where(~common_mask, error_grid)
    
    im = axes[1, 1].imshow(error_masked, cmap='RdBu_r', origin='upper', 
                           vmin=-30, vmax=30, aspect='auto')
    axes[1, 1].set_title('Spatial Error Pattern', fontweight='bold')
    axes[1, 1].set_xlabel('Column Index')
    axes[1, 1].set_ylabel('Row Index')
    plt.colorbar(im, ax=axes[1, 1], label='Error (minutes)')

plt.tight_layout()
output_path = os.path.join(output_dir, '6_error_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# Figure 7: Contour comparison
fig7, axes = plt.subplots(1, 2, figsize=(16, 7))

# ABM contours
levels = np.linspace(np.min(abm_valid), np.max(abm_valid), 15)
cs1 = axes[0].contourf(abm_data, levels=levels, cmap='hot', origin='upper')
axes[0].contour(abm_data, levels=levels, colors='black', linewidths=0.5, alpha=0.4, origin='upper')
axes[0].plot(c1, r1, 'b*', markersize=15, label='Ignition')
axes[0].set_title('ABM Isochrones (Contours)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Column Index')
axes[0].set_ylabel('Row Index')
axes[0].legend()
plt.colorbar(cs1, ax=axes[0], label='Arrival Time (minutes)')

# FlamMap contours
cs2 = axes[1].contourf(flammap_data, levels=levels, cmap='hot', origin='upper')
axes[1].contour(flammap_data, levels=levels, colors='black', linewidths=0.5, alpha=0.4, origin='upper')
axes[1].plot(c2, r2, 'b*', markersize=15, label='Ignition')
axes[1].set_title('FlamMap Isochrones (Contours)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Column Index')
axes[1].set_ylabel('Row Index')
axes[1].legend()
plt.colorbar(cs2, ax=axes[1], label='Arrival Time (minutes)')

plt.tight_layout()
output_path = os.path.join(output_dir, '7_contour_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nGenerated 7 visualization files in: {output_dir}")
print("\nSummary of Key Findings:")
print(f"  • Total cells analyzed: ABM={np.sum(abm_mask)}, FlamMap={np.sum(flammap_mask)}")
print(f"  • Common burned cells: {len(abm_common)} ({len(abm_common)/max(len(abm_valid), len(flammap_valid))*100:.1f}%)")
if len(abm_common) > 0:
    print(f"  • Correlation (R²): {r_squared:.4f}")
    print(f"  • Mean absolute error: {mae:.2f} minutes")
    print(f"  • RMSE: {rmse:.2f} minutes")
print("\n" + "="*80)