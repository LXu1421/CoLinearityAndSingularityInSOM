import os
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom
from scipy.interpolate import RectBivariateSpline
from skimage.restoration import inpaint

from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from skimage.measure import block_reduce
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from matplotlib import patches

# This part of the code is for data preparation


# Add GCI values for each model
GLOBAL_GCI_VALUES = {
    'Q004': 0.6125,
    'Q002': 0.55,
    'Q001': 0.5125,
    'Q003': 0.55,
    'Q006': 0.875,
    'Q005': 0.675
}


# Define lithology mappings for each Q file
lithology_mappings = {
    'Q004': {
        1: 'Mafic volcanic',
        2: 'Sedimentary cover',
        3: 'Psammitic sediment',
        4: 'Felsic intrusive',
        5: 'Pelitic sediment'
    },
    'Q002': {
        1: 'Metamorphic rocks',
        2: 'Sedimentary cover',
        3: 'Psammitic sediment',
        4: 'Pelitic sediment'
    },
    'Q001': {
        1: 'Sedimentary cover',
        2: 'Psammitic sediment',
        3: 'Felsic intrusive',
        4: 'Pelitic sediment'
    },
    'Q003': {
        1: 'Sedimentary cover',
        2: 'Psammitic sediment',
        3: 'Felsic intrusive',
        4: 'Pelitic sediment'
    },
    'Q006': {
        1: 'Chert',
        2: 'Carbonaceous rock',
        3: 'Sedimentary cover',
        4: 'Psammitic sediment',
        5: 'Felsic intrusive',
        6: 'Pelitic sediment'
    },
    'Q005': {
        1: 'Intermediate intrusive',
        2: 'Chert',
        3: 'Sedimentary cover',
        4: 'Carbonaceous rock',
        5: 'Pelitic sediment',
        6: 'Psammitic sediment'
    }
}


def create_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = "SyntheticNoddy"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")
    return output_dir


def load_bmp_to_matrix(file_path, expected_shape=None, keep_color=False):
    """Load BMP file into numpy matrix with enhanced validation."""
    try:
        img = Image.open(file_path)
        if not keep_color and img.mode != 'L':
            img = img.convert('L')

        # Use different data types for color vs grayscale
        if keep_color:
            matrix = np.array(img)  # Keep as uint8 for color images
        else:
            matrix = np.array(img, dtype=np.float64)  # Use float64 for geophysical data

        if expected_shape and matrix.shape[:2] != expected_shape:
            print(f"Warning: {file_path} has shape {matrix.shape}, expected {expected_shape}")

        # Check if data is constant or has very low variability
        if not keep_color and np.std(matrix) < 1e-6:
            print(f"WARNING: {file_path} has very low variability (std = {np.std(matrix):.6f})")

        return matrix
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def smooth_and_interpolate(matrix, smoothing_factor=2.0):
    """
    Apply smoothing and refined interpolation to the matrix while maintaining original resolution.

    Args:
        matrix: Input 2D magnetic data matrix
        smoothing_factor: Controls the smoothness (higher = smoother)

    Returns:
        Smoothed and interpolated matrix with same dimensions
    """
    # Create grid coordinates
    y, x = np.mgrid[0:matrix.shape[0], 0:matrix.shape[1]]

    # First apply substantial Gaussian smoothing
    smoothed = ndimage.gaussian_filter(matrix, sigma=smoothing_factor)

    # Create interpolation function with cubic spline
    interp_func = RectBivariateSpline(
        np.arange(matrix.shape[0]),
        np.arange(matrix.shape[1]),
        smoothed,
        kx=1, ky=1  # Linear interpolation or 3 for Cubic interpolation
    )

    # Evaluate on original grid points
    return interp_func(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))


def fill_gaps(data, threshold=0.01, max_iter=5):
    """
    Fill gaps (near-zero values) in the data using inpainting and interpolation.

    Args:
        data: Input 2D array with gaps
        threshold: Value below which is considered a gap
        max_iter: Maximum number of filling iterations

    Returns:
        Gap-filled array
    """
    # Create mask for gaps (near-zero areas)
    mask = (np.abs(data) < threshold * np.nanmax(np.abs(data)))

    # If no gaps found, return original data
    if not np.any(mask):
        return data

    # Use inpainting to fill gaps
    filled = inpaint.inpaint_biharmonic(data, mask)

    # Blend with original data at edges of gaps
    blend_mask = ndimage.distance_transform_edt(~mask)
    blend_mask = np.clip(blend_mask / blend_mask.max(), 0, 1)
    result = filled * blend_mask + data * (1 - blend_mask)

    return result


def calculate_1vd(matrix):
    """Calculate the first vertical derivative (1VD) with smoothing."""
    # Apply enhanced smoothing and interpolation
    smoothed = smooth_and_interpolate(matrix, smoothing_factor=20)

    # Calculate vertical gradient with proper spacing
    grad_y = np.gradient(smoothed, axis=0)

    # Fill gaps in the derivative
    grad_y = fill_gaps(grad_y)

    # Mild smoothing of the derivative
    grad_y = ndimage.gaussian_filter(grad_y, sigma=0.3)

    return grad_y


def calculate_tile(matrix, window_size=3):
    """Calculate tile (local mean)."""
    return ndimage.uniform_filter(matrix, size=window_size)


def calculate_analytical_signal(matrix):
    """Calculate the analytical signal magnitude with improved method."""
    # Apply enhanced smoothing and interpolation
    smoothed = smooth_and_interpolate(matrix, smoothing_factor=20)

    # Calculate gradients
    grad_y, grad_x = np.gradient(smoothed)

    # Calculate analytical signal
    analytical_signal = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Fill gaps in the analytical signal
    analytical_signal = fill_gaps(analytical_signal)

    # Apply final smoothing
    analytical_signal = ndimage.gaussian_filter(analytical_signal, sigma=0.5)

    return analytical_signal


def plot_geophysics(data, title, filename):
    """Create and save geophysics figure with proper formatting and better visualization."""
    fig, ax = plt.subplots(figsize=(12, 9))

    # Handle potential NaN or infinite values
    data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Use percentile-based clipping for better contrast
    vmin, vmax = np.percentile(data_clean, [2, 98])

    # Plot with y-axis increasing upward
    extent = [0, data.shape[1], 0, data.shape[0]]

    # Use different colormaps for different data types
    if '1VD' in title:
        cmap = 'RdBu_r'  # Red-Blue colormap for derivatives
    elif 'analytical' in title.lower():
        cmap = 'hot'  # Hot colormap for analytical signal
    else:
        cmap = 'viridis'  # Default colormap

    im = ax.imshow(data_clean, cmap=cmap, extent=extent, aspect='equal',
                   origin='lower', vmin=vmin, vmax=vmax, interpolation='bilinear')

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)

    # Add colorbar with better formatting
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    # Set colorbar label based on data type
    if '1VD' in title:
        cbar.set_label('1VD (nT/m or mGal/m)', rotation=270, labelpad=20)
    elif 'analytical' in title.lower():
        cbar.set_label('Analytical Signal Amplitude', rotation=270, labelpad=20)
    elif 'tile' in title.lower():
        cbar.set_label('Filtered Amplitude', rotation=270, labelpad=20)
    else:
        cbar.set_label('Intensity', rotation=270, labelpad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {filename} (Range: {vmin:.3f} to {vmax:.3f})")


def rgb_to_lithology_id(rgb_array, q_number):
    """Convert RGB values to lithology IDs based on the mapping for the specific Q model."""
    if rgb_array.ndim != 3:
        return rgb_array  # Already a lithology ID map

    # Get the lithology mapping for this Q number
    lith_mapping = lithology_mappings.get(q_number, {})

    # Create a unique identifier for each RGB color
    color_ids = (rgb_array[..., 0].astype(np.uint32) << 16 |
                 rgb_array[..., 1].astype(np.uint32) << 8 |
                 rgb_array[..., 2].astype(np.uint32))

    # Get unique colors in the image
    unique_colors = np.unique(color_ids)

    # Create a mapping from color ID to lithology ID
    color_to_lithology = {}
    for i, color_id in enumerate(unique_colors):
        if color_id == 0:  # Black background
            color_to_lithology[color_id] = 0
        else:
            # Assign lithology IDs sequentially starting from 1
            lithology_id = min(i, len(lith_mapping))
            color_to_lithology[color_id] = lithology_id

    # Apply the mapping
    lithology_map = np.zeros_like(color_ids, dtype=np.uint8)
    for color_id, lith_id in color_to_lithology.items():
        lithology_map[color_ids == color_id] = lith_id

    return lithology_map


def expand_planview_to_geophysics_size(planview_data, target_shape=(845, 1212), q_number=None):
    """Expand planview data to match geophysics data size and convert to lithology IDs."""
    if planview_data.ndim == 3:  # RGB image
        # Calculate zoom factors for each dimension
        zoom_factors = (target_shape[0] / planview_data.shape[0],
                        target_shape[1] / planview_data.shape[1], 1)
        expanded = zoom(planview_data, zoom_factors, order=0)  # order=0 for nearest neighbor

        # Convert RGB to lithology IDs
        if q_number:
            expanded = rgb_to_lithology_id(expanded, q_number)
    else:  # Already lithology IDs or grayscale
        zoom_factors = (target_shape[0] / planview_data.shape[0],
                        target_shape[1] / planview_data.shape[1])
        expanded = zoom(planview_data, zoom_factors, order=0)

    # Ensure the expanded data matches exactly the target shape
    if expanded.shape[0] > target_shape[0]:
        expanded = expanded[:target_shape[0], ...]
    if expanded.shape[1] > target_shape[1]:
        expanded = expanded[:, :target_shape[1], ...]

    return expanded


def plot_planview(data, title, filename, expanded=False, q_number=None):
    """Create and save planview figure with original colors and lithology legend."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set extent based on whether data is expanded or not
    if expanded:
        extent = [0, 1212, 0, 845]  # Match geophysics extent
    else:
        extent = [0, data.shape[1], 0, data.shape[0]]

    # Get unique colors/values for legend creation
    if data.ndim == 3:  # RGB image
        # Convert to uint32 for easier unique color identification
        color_data = (data[..., 0].astype(np.uint32) << 16) | \
                     (data[..., 1].astype(np.uint32) << 8) | \
                     data[..., 2].astype(np.uint32)
        unique_colors = np.unique(color_data)

        # Convert back to RGB tuples for legend
        rgb_colors = []
        for color in unique_colors:
            r = (color >> 16) & 0xFF
            g = (color >> 8) & 0xFF
            b = color & 0xFF
            rgb_colors.append((r, g, b))
    else:  # Lithology ID map
        unique_ids = np.unique(data)
        rgb_colors = None

    # Display the image
    if data.ndim == 3:  # RGB color data
        # Ensure data is uint8 for proper RGB display
        if data.dtype != np.uint8:
            plot_data = data.astype(np.uint8)
        else:
            plot_data = data
        im = ax.imshow(plot_data, extent=extent, aspect='equal', origin='lower')
    else:  # Lithology ID map
        # Use a colormap for lithology IDs
        cmap = plt.colormaps['tab10'].resampled(np.max(data) + 1)
        im = ax.imshow(data, cmap=cmap, extent=extent, aspect='equal', origin='lower', vmin=0, vmax=np.max(data))

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)

    # Create lithology legend
    patches = []
    if q_number and q_number in lithology_mappings:
        if data.ndim == 3:  # RGB image with lithology mapping
            # Create a mapping from color to lithology ID
            color_to_id = {}
            # This assumes the color values correspond to lithology IDs in order
            for i, color in enumerate(rgb_colors):
                lith_id = i + 1  # Assuming lithology IDs start at 1
                color_to_id[color] = lith_id

            # Create legend patches with proper names using original colors
            for color in rgb_colors:
                lith_id = color_to_id.get(color, 0)
                lith_name = lithology_mappings[q_number].get(lith_id, f"Unknown {lith_id}")
                patches.append(mpatches.Patch(color=np.array(color) / 255, label=f"{lith_name} (ID: {lith_id})"))
        else:  # Lithology ID map with mapping
            lith_mapping = lithology_mappings[q_number]
            cmap = plt.colormaps['tab10'].resampled(len(lith_mapping) + 1)

            for lith_id, lith_name in lith_mapping.items():
                color = cmap(lith_id)
                patches.append(mpatches.Patch(color=color, label=f"{lith_name} (ID: {lith_id})"))
    else:
        # Fallback to generic legend if no mapping available
        if data.ndim == 3:  # RGB image without mapping
            for i, color in enumerate(rgb_colors):
                patches.append(mpatches.Patch(color=np.array(color) / 255, label=f'Lithology{i + 1:02d}'))
        else:  # Lithology ID map without mapping
            cmap = plt.colormaps['tab10'].resampled(len(unique_ids))
            for i, lith_id in enumerate(unique_ids):
                color = cmap(i)
                patches.append(mpatches.Patch(color=color, label=f'Lithology{lith_id:02d}'))

    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def process_geophysical_data():
    """Process all gravity and magnetic data files."""
    results = {}
    for i in range(1, 7):
        for field_type in ['Grav', 'Mag']:
            file_name = f"Q{i:03d}-{field_type}.bmp"

            if os.path.exists(file_name):
                print(f"Processing {file_name}...")
                matrix = load_bmp_to_matrix(file_name, expected_shape=(845, 1212))
                if matrix is not None:
                    # Calculate derivatives
                    vd_1 = calculate_1vd(matrix)
                    tile_result = calculate_tile(matrix)
                    analytical_sig = calculate_analytical_signal(matrix)

                    # Print statistics for debugging
                    print(f"  Raw data range: {matrix.min():.3f} to {matrix.max():.3f}")
                    print(f"  1VD range: {vd_1.min():.3f} to {vd_1.max():.3f}")
                    print(f"  Analytical signal range: {analytical_sig.min():.3f} to {analytical_sig.max():.3f}")

                    results[file_name] = {
                        'raw': matrix,
                        '1VD': vd_1,
                        'tile': tile_result,
                        'analytical_signal': analytical_sig
                    }
            else:
                print(f"File {file_name} not found, skipping...")
    return results


def process_planview_data():
    """Process all planview files with original colors."""
    results = {}
    for i in range(1, 7):
        file_name = f"Q{i:03d}-planview.bmp"

        if os.path.exists(file_name):
            print(f"Processing {file_name}...")
            matrix = load_bmp_to_matrix(file_name, expected_shape=(342, 490), keep_color=True)
            if matrix is not None:
                results[file_name] = matrix
        else:
            print(f"File {file_name} not found, skipping...")
    return results


def validate_exported_data(file_path, expected_shape=None, min_size_kb=100):
    """Validate that exported data meets expectations."""
    try:
        # Check file size
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb < min_size_kb:
            print(f"  Warning: File size is small ({file_size_kb:.1f}KB < {min_size_kb}KB) - {file_path}")

        # Check data shape if expected_shape is provided
        if expected_shape:
            data = np.load(file_path)
            for key, arr in data.items():
                if arr.shape[:2] != expected_shape:
                    print(f"  Warning: Array {key} has shape {arr.shape}, expected {expected_shape} - {file_path}")
                else:
                    print(f"  Validation OK: {key} has correct shape {expected_shape} - {file_path}")
            data.close()

        return True
    except Exception as e:
        print(f"  Error validating {file_path}: {str(e)}")
        return False


def calculate_local_gci(lithology_map, window_size=100):
    """
    Calculate local Geological Complexity Index (GCI) based on lithology variance.

    Improved formula with better normalization and NaN handling
    """
    # Pad the lithology map to handle edges
    padded = np.pad(lithology_map, window_size // 2, mode='reflect')

    # Initialize output
    gci_map = np.zeros_like(lithology_map, dtype=np.float32)

    # Theoretical maximums (adjusted for 5x5 window)
    max_lith_count = min(4, window_size ** 2)  # Realistic maximum unique lithologies
    max_border_length = 4 * (window_size//10 - 1)  # Maximum possible borders

    for i in range(lithology_map.shape[0]):
        for j in range(lithology_map.shape[1]):
            window = padded[i:i + window_size, j:j + window_size]

            # Count unique lithologies (handle NaN/inf)
            unique_lith = np.unique(window[~np.isnan(window)])
            lith_count = len(unique_lith)

            # Calculate border length
            border_pixels = np.sum(window[:-1, :] != window[1:, :]) + \
                            np.sum(window[:, :-1] != window[:, 1:])

            # Calculate GCI with safe division
            gci_map[i, j] = 0.5 * (min(lith_count, max_lith_count) / max_lith_count + \
                                   0.5 * min(border_pixels, max_border_length) / max_border_length)

    return gci_map, lith_count, border_pixels


def calculate_vif(features):
    """Calculate Variance Inflation Factors for feature matrix."""
    n_features = features.shape[-1]
    vif = np.zeros(n_features)

    for i in range(n_features):
        # Calculate VIF for each feature against all others
        vif[i] = variance_inflation_factor(features.reshape(-1, n_features), i)

    return vif


def plot_collinearity_heatmap(corr_map, gci_map, global_gci, title, filename):
    """Plot collinearity heatmap overlaid with GCI zones."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create custom colormap for correlation
    cmap_corr = LinearSegmentedColormap.from_list(
        'corr_cmap', ['blue', 'white', 'red'], N=256)

    # Plot correlation heatmap
    im1 = ax.imshow(corr_map, cmap=cmap_corr, vmin=-1, vmax=1,
                    alpha=0.7, origin='lower')

    # Create GCI contours
    levels = np.linspace(0, 1, 11)
    cs = ax.contour(gci_map, levels=levels, colors='black', linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8)

    # Add colorbar for correlation
    cbar1 = fig.colorbar(im1, ax=ax, shrink=0.8, pad=0.02)
    cbar1.set_label('Pearson Correlation (TMI vs Gravity)', rotation=270, labelpad=20)

    # Clean correlation map
    clean_corr = np.nan_to_num(corr_map, nan=0.0)
    global_mean = np.mean(clean_corr[clean_corr != 0])  # Exclude zeros

    # Calculate local correlations
    local_corr = block_reduce(clean_corr, (5, 5), np.mean)

    # Calculate ratio with protection against division by zero
    r_ratio = np.zeros_like(local_corr)
    valid_mask = (global_mean > 1e-6) & (local_corr != 0)
    r_ratio[valid_mask] = local_corr[valid_mask] / global_mean

    # Overlay hotspots where R_ratio > Percentile-based threshold
    threshold = np.percentile(r_ratio, 95)  # Top 10% as hotspots
    hotspot_mask = r_ratio > threshold
    y, x = np.where(hotspot_mask)
    ax.scatter(x * 5, y * 5, c='yellow', s=2, alpha=0.4, marker='.',
               label=f'Hotspots (R > {threshold:.1f}×global)')

    ax.set_title(f'{title}\nGlobal GCI: {global_gci:.4f}', fontsize=14, pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def enhanced_analysis(geophys_data, planview_data, output_dir):
    """Perform enhanced analysis with improved correlations and GCI mapping."""
    enhanced_dir = os.path.join(output_dir, "Enhanced_Analysis")
    os.makedirs(enhanced_dir, exist_ok=True)

    # Store summary statistics for all models
    all_stats = []

    for i in range(1, 7):
        q_key = f"Q{i:03d}"
        grav_key = f"{q_key}-Grav.bmp"
        mag_key = f"{q_key}-Mag.bmp"
        pv_key = f"{q_key}-planview.bmp"

        if grav_key not in geophys_data or mag_key not in geophys_data or pv_key not in planview_data:
            continue

        # Get the data
        tmi = geophys_data[mag_key]['raw']
        gravity = geophys_data[grav_key]['raw']
        lithology = expand_planview_to_geophysics_size(planview_data[pv_key])

        # DEBUG: Check data variability for problematic models
        if q_key in ['Q003', 'Q005']:
            print(f"DEBUG {q_key}: TMI range = {tmi.min():.6f} to {tmi.max():.6f}, std = {tmi.std():.6f}")
            print(
                f"DEBUG {q_key}: Gravity range = {gravity.min():.6f} to {gravity.max():.6f}, std = {gravity.std():.6f}")

            # Check if data is nearly constant
            if tmi.std() < 1e-6 or gravity.std() < 1e-6:
                print(f"WARNING: {q_key} has very low variability - this may cause zero correlation")

                # Apply slight noise to avoid division by zero in correlation
                if tmi.std() < 1e-6:
                    tmi = tmi + np.random.normal(0, 1e-6, tmi.shape)
                    print(f"Applied minimal noise to TMI for {q_key}")
                if gravity.std() < 1e-6:
                    gravity = gravity + np.random.normal(0, 1e-6, gravity.shape)
                    print(f"Applied minimal noise to Gravity for {q_key}")

        # Convert lithology to single channel if RGB
        if lithology.ndim == 3:
            lithology = (lithology[..., 0] << 16) | (lithology[..., 1] << 8) | lithology[..., 2]

        # Calculate moving-window correlations (now using absolute values)
        corr_map = calculate_moving_correlation(tmi, gravity)

        # DEBUG: Check correlation results
        if q_key in ['Q003', 'Q005']:
            print(f"DEBUG {q_key}: Correlation map range = {corr_map.min():.6f} to {corr_map.max():.6f}")
            print(f"DEBUG {q_key}: Non-zero correlation values count = {np.sum(corr_map != 0)}")

            # If still all zeros, try a different approach
            if np.all(corr_map == 0):
                print(f"WARNING: {q_key} still has all zero correlations, trying alternative method")
                corr_map = alternative_correlation_method(tmi, gravity)

        # Calculate local GCI
        gci_map, lith_count, border_length = calculate_local_gci(lithology)

        # Get global GCI for this model
        global_gci = GLOBAL_GCI_VALUES.get(q_key, 0.5)

        # Create feature matrix for VIF calculation
        features = np.stack([tmi, gravity,
                             geophys_data[mag_key]['1VD'],
                             geophys_data[grav_key]['1VD']], axis=-1)
        vif = calculate_vif(features)

        # Create enhanced collinearity plot
        zones, corr_threshold, gci_median = plot_enhanced_collinearity_heatmap(
            corr_map, gci_map, global_gci,
            f"Model {q_key}",
            os.path.join(enhanced_dir, f"{q_key}_enhanced_collinearity_analysis.png"),
                                       smooth_sigma=1.0
        )

        # Calculate zone statistics
        zone_stats = {}
        for zone_id in [1, 2, 3, 4]:
            zone_mask = zones == zone_id
            zone_area = np.sum(zone_mask)
            zone_stats[f'zone_{zone_id}_area_pct'] = (zone_area / zones.size) * 100
            if zone_area > 0:
                zone_stats[f'zone_{zone_id}_mean_corr'] = np.mean(corr_map[zone_mask])
                zone_stats[f'zone_{zone_id}_mean_gci'] = np.mean(gci_map[zone_mask])

        # Save all metrics
        np.savez_compressed(
            os.path.join(enhanced_dir, f"{q_key}_enhanced_metrics.npz"),
            correlation_map=corr_map,
            gci_map=gci_map,
            zones=zones,
            lithology_count=lith_count,
            border_length=border_length,
            vif=vif,
            global_gci=global_gci,
            thresholds={'corr_threshold': corr_threshold, 'gci_median': gci_median}
        )

        # Create comprehensive summary statistics
        stats = {
            'model': q_key,
            'global_gci': global_gci,
            'mean_local_gci': np.mean(gci_map),
            'std_local_gci': np.std(gci_map),
            'mean_abs_correlation': np.mean(corr_map),
            'std_abs_correlation': np.std(corr_map),
            'correlation_threshold': corr_threshold,
            'gci_median': gci_median,
            'vif_tmi': vif[0] if len(vif) > 0 else np.nan,
            'vif_gravity': vif[1] if len(vif) > 1 else np.nan,
            'vif_tmi_1vd': vif[2] if len(vif) > 2 else np.nan,
            'vif_gravity_1vd': vif[3] if len(vif) > 3 else np.nan,
            **zone_stats
        }

        all_stats.append(stats)

    # Save comprehensive statistics
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(os.path.join(enhanced_dir, "comprehensive_statistics.csv"), index=False)

        # Create summary report
        create_analysis_summary_report(stats_df, enhanced_dir)


def alternative_correlation_method(data1, data2, window_size=5):
    """
    Alternative correlation calculation method for data with very low variability.
    Uses a normalized cross-correlation approach that's more robust for near-constant data.
    """
    h, w = data1.shape
    corr_map = np.zeros_like(data1)

    pad_size = window_size // 2
    data1_pad = np.pad(data1, pad_size, mode='reflect')
    data2_pad = np.pad(data2, pad_size, mode='reflect')

    for i in range(h):
        for j in range(w):
            window1 = data1_pad[i:i + window_size, j:j + window_size].flatten()
            window2 = data2_pad[i:i + window_size, j:j + window_size].flatten()

            # Handle near-constant windows
            if np.std(window1) < 1e-6 or np.std(window2) < 1e-6:
                # If either window is constant, set correlation to 0
                corr_map[i, j] = 0
                continue

            # Normalize the windows
            window1_norm = (window1 - np.mean(window1)) / np.std(window1)
            window2_norm = (window2 - np.mean(window2)) / np.std(window2)

            # Calculate cross-correlation
            cross_corr = np.sum(window1_norm * window2_norm) / (window_size * window_size)

            # Use absolute value for collinearity measure
            corr_map[i, j] = abs(cross_corr)

    return corr_map


def create_analysis_summary_report(stats_df, output_dir):
    """Create a summary report of the analysis results."""
    report_path = os.path.join(output_dir, "analysis_summary_report.txt")

    with open(report_path, 'w') as f:
        f.write("ENHANCED COLLINEARITY AND GCI ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("METHODOLOGY:\n")
        f.write("- Collinearity measured using absolute Pearson correlation |R| in 5×5 moving windows\n")
        f.write("- GCI calculated locally using lithology count and boundary length\n")
        f.write("- Four zones classified based on correlation and GCI thresholds\n")
        f.write("- Hotspots identified using 90th percentile threshold\n\n")

        f.write("MODEL STATISTICS:\n")
        for _, row in stats_df.iterrows():
            f.write(f"\n{row['model']}:\n")
            f.write(f"  Global GCI: {row['global_gci']:.3f}\n")
            f.write(f"  Mean |R|: {row['mean_abs_correlation']:.3f}\n")
            f.write(f"  Correlation threshold: {row['correlation_threshold']:.3f}\n")
            f.write(f"  Zone 1 (High Col./High GCI): {row.get('zone_1_area_pct', 0):.1f}%\n")
            f.write(f"  Zone 2 (High Col./Low GCI): {row.get('zone_2_area_pct', 0):.1f}%\n")
            f.write(f"  Zone 3 (Low Col./High GCI): {row.get('zone_3_area_pct', 0):.1f}%\n")
            f.write(f"  Zone 4 (Low Col./Low GCI): {row.get('zone_4_area_pct', 0):.1f}%\n")


def plot_lithology_geophysics(model, lithology_map, tmi, gravity, lithology_mappings, output_dir):
    """
    Generate box plots showing TMI and gravity distributions per lithology unit.

    Args:
        model: Model identifier (e.g., 'Q001')
        lithology_map: 2D array of lithology IDs
        tmi: Total Magnetic Intensity data (2D array)
        gravity: Gravity data (2D array)
        lithology_mappings: Dictionary mapping lithology IDs to names
        output_dir: Output directory for saving plots
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Flatten arrays
    lith_flat = lithology_map.flatten()
    tmi_flat = tmi.flatten()
    grav_flat = gravity.flatten()

    # Prepare data for DataFrame
    data = []
    unique_ids = np.unique(lith_flat)

    for lith_id in unique_ids:
        # Get the proper lithology name from the mapping
        lith_name = lithology_mappings.get(int(lith_id), f'Unknown_{lith_id}')
        mask = (lith_flat == lith_id)
        count = np.sum(mask)

        # Skip lithologies with insufficient data
        if count < 100:
            continue

        data.extend([{
            'Lithology': lith_name,
            'Lithology_ID': lith_id,  # Add ID for sorting
            'Value': val,
            'Property': 'TMI (nT)',
            'Count': count
        } for val in tmi_flat[mask]])

        data.extend([{
            'Lithology': lith_name,
            'Lithology_ID': lith_id,  # Add ID for sorting
            'Value': val,
            'Property': 'Gravity (µm/s²)',
            'Count': count
        } for val in grav_flat[mask]])

    if not data:
        print(f"No valid lithology data found for {model}")
        return

    df = pd.DataFrame(data)

    # Sort lithologies by their ID to maintain consistent order
    df = df.sort_values('Lithology_ID')

    # Create plot
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(
        x='Lithology',
        y='Value',
        hue='Property',
        data=df,
        palette={'TMI (nT)': 'royalblue', 'Gravity (µm/s²)': 'firebrick'},
        showfliers=False,
        width=0.7,
        order=sorted(df['Lithology'].unique(), key=lambda x: df[df['Lithology'] == x]['Lithology_ID'].iloc[0])
    )

    # Add sample count annotations
    lith_order = sorted(df['Lithology'].unique(), key=lambda x: df[df['Lithology'] == x]['Lithology_ID'].iloc[0])
    for i, lith in enumerate(lith_order):
        count = df[df['Lithology'] == lith]['Count'].iloc[0]
        ax.text(i, ax.get_ylim()[0] - 0.05 * np.diff(ax.get_ylim())[0],
                f'n={count:,}',
                ha='center',
                va='top',
                fontsize=9)

    plt.title(f'Geophysical Properties by Lithology - {model}\n(GCI: {GLOBAL_GCI_VALUES.get(model, "N/A")})',
              fontsize=16, pad=20)
    plt.xlabel('Lithology Unit', fontsize=12)
    plt.ylabel('Geophysical Value', fontsize=12)

    # Adjust x-axis labels rotation and alignment
    plt.xticks(fontsize=10)

    # Improve legend
    plt.legend(title='Geophysical Property', loc='upper right', framealpha=1)

    # Add grid and styling
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    fig_path = os.path.join(output_dir, "Figures", f"{model}_lithology_geophysics.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved lithology geophysics plot: {fig_path}")


def calculate_moving_correlation(data1, data2, window_size=5):
    """Calculate moving window correlation using absolute values for proper collinearity detection."""
    h, w = data1.shape
    corr_map = np.zeros_like(data1)

    pad_size = window_size // 2
    data1_pad = np.pad(data1, pad_size, mode='reflect')
    data2_pad = np.pad(data2, pad_size, mode='reflect')

    for i in range(h):
        for j in range(w):
            window1 = data1_pad[i:i + window_size, j:j + window_size].flatten()
            window2 = data2_pad[i:i + window_size, j:j + window_size].flatten()

            # Calculate correlation and take absolute value for true collinearity
            corr = np.corrcoef(window1, window2)[0, 1]
            if not np.isnan(corr):
                corr_map[i, j] = abs(corr)  # Use absolute correlation

    return corr_map


def identify_collinearity_zones(corr_map, gci_map, threshold_percentile=90,
                                min_corr_threshold=0.3, min_gci_threshold=0.3):
    """
    Identify and classify different collinearity zones with adaptive thresholding.

    Parameters:
    -----------
    corr_map : np.ndarray
        Absolute correlation map
    gci_map : np.ndarray
        GCI (Geological Complexity Index) map
    threshold_percentile : float
        Percentile for adaptive thresholding (default: 90)
    min_corr_threshold : float
        Minimum correlation threshold to ensure meaningful separation (default: 0.3)
    min_gci_threshold : float
        Minimum GCI threshold to ensure meaningful separation (default: 0.3)

    Returns:
    --------
    zones : np.ndarray
        Zone classification map (1-4)
    corr_threshold : float
        Applied correlation threshold
    gci_threshold : float
        Applied GCI threshold
    zone_stats : dict
        Statistics about each zone
    """
    # Remove NaN values for threshold calculation
    valid_corr = corr_map[~np.isnan(corr_map)]
    valid_gci = gci_map[~np.isnan(gci_map)]

    # Calculate basic statistics
    corr_mean = np.mean(valid_corr)
    corr_std = np.std(valid_corr)
    corr_nonzero_pct = np.sum(valid_corr > 0.01) / len(valid_corr) * 100

    gci_mean = np.mean(valid_gci)
    gci_std = np.std(valid_gci)

    # Adaptive correlation threshold
    # Use percentile-based threshold, but ensure it's meaningful
    corr_percentile_threshold = np.percentile(valid_corr, threshold_percentile)

    # If most values are near zero, use a more robust approach
    if corr_nonzero_pct < 20:  # Less than 20% have meaningful correlation
        # Use mean + 1 std for sparse correlation maps
        corr_threshold = max(min_corr_threshold,
                             corr_mean + corr_std,
                             np.percentile(valid_corr[valid_corr > 0.01], 75))
        print(f"⚠️ Sparse correlation detected ({corr_nonzero_pct:.1f}% non-zero)")
        print(f"   Using robust threshold: {corr_threshold:.3f} (mean+std approach)")
    else:
        corr_threshold = max(min_corr_threshold, corr_percentile_threshold)

    # Adaptive GCI threshold
    # Use median for balanced split, but ensure minimum separation
    gci_median = np.median(valid_gci)

    # Check if GCI values are too low or too uniform
    if gci_median < min_gci_threshold:
        # Use a more robust threshold
        gci_threshold = max(min_gci_threshold,
                            gci_mean,
                            np.percentile(valid_gci, 60))
        print(f"⚠️ Low GCI values detected (median={gci_median:.3f})")
        print(f"   Using adjusted threshold: {gci_threshold:.3f}")
    elif gci_std / (gci_mean + 1e-10) < 0.2:  # Low variability (CoV < 20%)
        # Use mean instead of median for low-variability data
        gci_threshold = gci_mean
        print(f"⚠️ Low GCI variability detected (CoV={gci_std / gci_mean:.2%})")
        print(f"   Using mean threshold: {gci_threshold:.3f}")
    else:
        gci_threshold = gci_median

    # Create zone classifications
    zones = np.zeros_like(corr_map, dtype=int)

    # Zone 1: High collinearity + High GCI (complex geology with correlated responses)
    high_corr_high_gci = (corr_map >= corr_threshold) & (gci_map >= gci_threshold)
    zones[high_corr_high_gci] = 1

    # Zone 2: High collinearity + Low GCI (simple geology with correlated responses)
    high_corr_low_gci = (corr_map >= corr_threshold) & (gci_map < gci_threshold)
    zones[high_corr_low_gci] = 2

    # Zone 3: Low collinearity + High GCI (complex geology with uncorrelated responses)
    low_corr_high_gci = (corr_map < corr_threshold) & (gci_map >= gci_threshold)
    zones[low_corr_high_gci] = 3

    # Zone 4: Low collinearity + Low GCI (simple geology with uncorrelated responses)
    low_corr_low_gci = (corr_map < corr_threshold) & (gci_map < gci_threshold)
    zones[low_corr_low_gci] = 4

    # Calculate zone statistics
    zone_stats = {}
    for zone_id in range(1, 5):
        zone_mask = zones == zone_id
        zone_count = np.sum(zone_mask)
        zone_pct = zone_count / zones.size * 100

        if zone_count > 0:
            zone_corr_mean = np.mean(corr_map[zone_mask])
            zone_gci_mean = np.mean(gci_map[zone_mask])
        else:
            zone_corr_mean = 0
            zone_gci_mean = 0

        zone_stats[zone_id] = {
            'count': zone_count,
            'percentage': zone_pct,
            'mean_corr': zone_corr_mean,
            'mean_gci': zone_gci_mean
        }

    # Print summary
    print(f"\n📊 Zone Classification Summary:")
    print(f"   Correlation threshold: {corr_threshold:.3f} (non-zero: {corr_nonzero_pct:.1f}%)")
    print(f"   GCI threshold: {gci_threshold:.3f}")
    print(f"\n   Zone Distribution:")
    zone_names = {
        1: "High Corr + High GCI",
        2: "High Corr + Low GCI",
        3: "Low Corr + High GCI",
        4: "Low Corr + Low GCI"
    }
    for zone_id in range(1, 5):
        stats = zone_stats[zone_id]
        if stats['percentage'] > 0.1:  # Only show zones with >0.1% coverage
            print(f"   Zone {zone_id} ({zone_names[zone_id]}): "
                  f"{stats['percentage']:.1f}% "
                  f"(|R|={stats['mean_corr']:.3f}, GCI={stats['mean_gci']:.3f})")

    return zones, corr_threshold, gci_threshold, zone_stats





def plot_enhanced_collinearity_heatmap(corr_map, gci_map, global_gci, title, save_path,
                                       smooth_sigma=1.0):
    """Create an enhanced collinearity heatmap with clear zone labels and explanations.

    Parameters:
    -----------
    smooth_sigma : float
        Sigma for Gaussian smoothing. Set to 0 to disable smoothing.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Identify zones
    # zones, corr_threshold, gci_median = identify_collinearity_zones(corr_map, gci_map)
    # Update this line in the plotting function
    zones, corr_threshold, gci_median, zone_stats = identify_collinearity_zones(corr_map, gci_map)

    # Then update references from gci_median to gci_threshold

    # Apply Gaussian smoothing if requested
    corr_map_smooth = gaussian_filter(corr_map, sigma=smooth_sigma) if smooth_sigma > 0 else corr_map
    gci_map_smooth = gaussian_filter(gci_map, sigma=smooth_sigma) if smooth_sigma > 0 else gci_map

    # ===================== LEFT PANEL: Correlation map with GCI contours =====================
    im1 = ax1.imshow(corr_map_smooth, cmap='RdYlBu_r', vmin=0, vmax=1, origin='lower',
                     interpolation='bilinear')

    # Add GCI contours with distinct colors and thicker lines
    # contour_levels = [0.2, 0.4, 0.6, 0.8]
    # contour_colors = ['#00FF00', '#FFFF00', '#FF8800', '#FF0000']  # Green to Red gradient

    # for level, color in zip(contour_levels, contour_colors):
    #     cs = ax1.contour(gci_map_smooth, levels=[level], colors=[color],
    #                      linewidths=2.5, alpha=0.9)
    #     # Add labels with background for better visibility
    #     ax1.clabel(cs, inline=True, fontsize=9, fmt=f'GCI={level:.1f}',
    #                inline_spacing=10,
    #                manual=False)

    # Add zone annotations in corners (avoiding overlap)
    # h, w = corr_map.shape

    # zone_labels = {
    #     1: 'High Collinearity\nHigh GCI\n(Critical)',
    #     2: 'High Collinearity\nLow GCI\n(Warning)',
    #     3: 'Low Collinearity\nHigh GCI\n(Good)',
    #     4: 'Low Collinearity\nLow GCI\n(Acceptable)'
    # }

    # zone_colors = {1: '#FF3333', 2: '#FF9933', 3: '#3366FF', 4: '#33CC33'}

    # Position labels in corners to avoid overlap
    # label_positions = {
    #     1: (0.95, 0.95, 'top right'),  # Top right
    #     2: (0.05, 0.95, 'top left'),  # Top left
    #     3: (0.95, 0.05, 'bottom right'),  # Bottom right
    #     4: (0.05, 0.05, 'bottom left')  # Bottom left
    # }

    # for zone_id in [1, 2, 3, 4]:
    #     zone_mask = zones == zone_id
    #     if np.any(zone_mask):
    #         pos_x, pos_y, alignment = label_positions[zone_id]
    #
    #         # Determine text alignment
    #         ha = 'right' if 'right' in alignment else 'left'
    #         va = 'top' if 'top' in alignment else 'bottom'
    #
    #        # Add label with distinct styling
    #         ax1.text(pos_x, pos_y, zone_labels[zone_id],
    #                  transform=ax1.transAxes,
    #                  fontsize=9, ha=ha, va=va,
    #                  bbox=dict(boxstyle="round,pad=0.5",
    #                            facecolor='white',
    #                            edgecolor=zone_colors[zone_id],
    #                            linewidth=2.5,
    #                            alpha=0.95),
    #                  weight='bold',
    #                  color=zone_colors[zone_id])

    ax1.set_title(f'{title}\nAbsolute Pearson Correlation |R| with GCI Isolines',
                  fontsize=13, weight='bold', pad=10)
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)

    # Add colorbar for correlation
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.85)
    cbar1.set_label('|Pearson Correlation|', fontsize=11, weight='bold')

    # Add threshold info at bottom
    threshold_text = (f'Thresholds:'
                      f'Correlation: |R| ≥ {corr_threshold:.3f}'
                      f'  GCI: {gci_median:.2f}')
    ax1.text(0.5, -0.15, threshold_text,
             transform=ax1.transAxes, fontsize=9,
             ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))

    # ===================== RIGHT PANEL: Zone classification map =====================
    # Use distinct colors for each zone
    from matplotlib.colors import ListedColormap
    zone_colors_list = ['#33CC33', '#3366FF', '#FF9933', '#FF3333']  # Order: 4,3,2,1
    zone_cmap = ListedColormap(zone_colors_list)

    im2 = ax2.imshow(zones, cmap=zone_cmap, vmin=1, vmax=4, origin='lower',
                     interpolation='nearest')

    ax2.set_title('Collinearity-GCI Zone Classification', fontsize=13, weight='bold', pad=10)
    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)

    # Add colorbar with better labels
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.85, ticks=[1, 2, 3, 4])
    cbar2.set_label('Zone Type', fontsize=11, weight='bold')
    cbar2.ax.set_yticklabels(['High Col.\nHigh GCI', 'High Col.\nLow GCI',
                              'Low Col.\nHigh GCI', 'Low Col.\nLow GCI'],
                             fontsize=9)

    # Add legend for zone interpretation
    # legend_elements = [
    #     patches.Patch(facecolor='#FF3333', edgecolor='black', label='Zone 1: Critical (avoid)'),
    #     patches.Patch(facecolor='#FF9933', edgecolor='black', label='Zone 2: Warning'),
    #     patches.Patch(facecolor='#3366FF', edgecolor='black', label='Zone 3: Good'),
    #     patches.Patch(facecolor='#33CC33', edgecolor='black', label='Zone 4: Acceptable')
    # ]
    # ax2.legend(handles=legend_elements, loc='upper center',
    #           bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9,
    #            framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return zones, corr_threshold, gci_median


def save_results(output_dir, geophys_data, planview_data):
    """Save all results including figures and data with enhanced lithology analysis."""
    # Create subdirectories
    fig_dir = os.path.join(output_dir, "Figures")
    data_dir = os.path.join(output_dir, "Data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("\nSaving planview figures...")
    # Save planview figures
    for pv_key, pv_data in planview_data.items():
        base_name = os.path.splitext(pv_key)[0]
        q_number = pv_key[:4]  # Extract Q number (e.g., 'Q001')

        # Save original size figure
        fig_path = os.path.join(fig_dir, f"{base_name}.png")
        plot_planview(pv_data,
                      f"Plan View: {pv_key.replace('.bmp', '')}",
                      fig_path,
                      expanded=False,
                      q_number=q_number)

        # Expand planview to match geophysics size and convert to lithology IDs
        expanded_pv = expand_planview_to_geophysics_size(pv_data, q_number=q_number)

        # Save expanded figure
        fig_path = os.path.join(fig_dir, f"{base_name}_expanded.png")
        plot_planview(expanded_pv,
                      f"Plan View (Expanded): {pv_key.replace('.bmp', '')}",
                      fig_path,
                      expanded=True,
                      q_number=q_number)

        # Verify expansion worked
        if expanded_pv.shape[:2] != (845, 1212):
            print(f"Error: Expanded planview has wrong shape {expanded_pv.shape}, expected (845, 1212)")

        # Save both original and expanded data
        data_path = os.path.join(data_dir, f"{base_name}_processed.npz")
        np.savez_compressed(data_path, original=pv_data, expanded=expanded_pv)

        # Validate the saved file
        print(f"\nValidating {data_path}:")
        validate_exported_data(data_path, expected_shape=(845, 1212), min_size_kb=100)

        # Print array info for debugging
        with np.load(data_path) as data:
            print("  Contents:")
            for key in data.files:
                arr = data[key]
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes / 1024:.1f}KB")
                if arr.dtype == np.uint8:
                    print(f"      Values: unique={len(np.unique(arr))}, min={arr.min()}, max={arr.max()}")
                else:
                    print(f"      Values: min={arr.min():.3f}, max={arr.max():.3f}")

        # Add lithology-geophysics analysis if corresponding geophysics data exists
        grav_key = f"{q_number}-Grav.bmp"
        mag_key = f"{q_number}-Mag.bmp"

        if grav_key in geophys_data and mag_key in geophys_data:
            tmi = geophys_data[mag_key]['raw']
            gravity = geophys_data[grav_key]['raw']

            # Generate lithology geophysics plot
            plot_lithology_geophysics(
                q_number,
                expanded_pv,  # Use the lithology ID map
                tmi,
                gravity,
                lithology_mappings.get(q_number, {}),
                output_dir
            )

    print("\nSaving geophysical figures...")
    # Save geophysical data figures and data
    for geo_key, geo_data in geophys_data.items():
        base_name = os.path.splitext(geo_key)[0]

        # Save raw data figure
        fig_path = os.path.join(fig_dir, f"{base_name}_raw.png")
        plot_geophysics(geo_data['raw'],
                        f"Raw Data: {geo_key.replace('.bmp', '')}",
                        fig_path)

        # Save processed data figures
        for proc_type in ['1VD', 'tile', 'analytical_signal']:
            fig_path = os.path.join(fig_dir, f"{base_name}_{proc_type}.png")
            plot_geophysics(geo_data[proc_type],
                            f"{proc_type}: {geo_key.replace('.bmp', '')}",
                            fig_path)

        # Save data
        data_path = os.path.join(data_dir, f"{base_name}_processed.npz")
        np.savez_compressed(data_path, **geo_data)

        # Validate the saved file
        print(f"\nValidating {data_path}:")
        validate_exported_data(data_path, expected_shape=(845, 1212), min_size_kb=1000)

        # Print array info for debugging
        with np.load(data_path) as data:
            print("  Contents:")
            for key in data.files:
                arr = data[key]
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes / 1024:.1f}KB")
                print(f"      Values: min={arr.min():.3f}, max={arr.max():.3f}")

    print(f"\nAll results saved to: {output_dir}")

    print("\nPerforming enhanced analysis...")
    enhanced_analysis(geophys_data, planview_data, output_dir)
    print("\nEnhanced analysis complete!")


def main():
    print("Starting processing...")

    # Create output directory
    output_dir = create_output_dir()

    print("\nProcessing geophysical data...")
    geophys_data = process_geophysical_data()

    print("\nProcessing planview data...")
    planview_data = process_planview_data()

    if geophys_data or planview_data:
        print("\nSaving results...")
        save_results(output_dir, geophys_data, planview_data)
        print("\nProcessing complete!")
    else:
        print("\nNo data files found to process.")


if __name__ == "__main__":
    main()






