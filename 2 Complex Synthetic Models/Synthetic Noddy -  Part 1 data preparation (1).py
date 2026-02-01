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
    """Load BMP file into numpy matrix."""
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


def expand_planview_to_geophysics_size(planview_data, target_shape=(845, 1212)):
    """Expand planview data to match geophysics data size using nearest neighbor interpolation."""
    if planview_data.ndim == 3:  # RGB image
        # Calculate zoom factors for each dimension
        zoom_factors = (target_shape[0] / planview_data.shape[0],
                        target_shape[1] / planview_data.shape[1], 1)
        expanded = zoom(planview_data, zoom_factors, order=0)  # order=0 for nearest neighbor
    else:  # Grayscale image
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

    # Get unique colors in the image
    if data.ndim == 3:  # RGB image
        # Convert to uint32 for easier unique color identification
        color_data = (data[..., 0].astype(np.uint32) << 16) | \
                     (data[..., 1].astype(np.uint32) << 8) | \
                     data[..., 2].astype(np.uint32)
        unique_colors = np.unique(color_data)

        # Convert back to RGB tuples for plotting
        rgb_colors = []
        for color in unique_colors:
            r = (color >> 16) & 0xFF
            g = (color >> 8) & 0xFF
            b = color & 0xFF
            rgb_colors.append((r, g, b))
    else:  # Grayscale image
        unique_colors = np.unique(data)
        rgb_colors = None

    # Set extent based on whether data is expanded or not
    if expanded:
        extent = [0, 1212, 0, 845]  # Match geophysics extent
    else:
        extent = [0, data.shape[1], 0, data.shape[0]]

    # Display the image
    if data.ndim == 3:  # RGB color data
        # Ensure data is uint8 for proper RGB display
        if data.dtype != np.uint8:
            plot_data = data.astype(np.uint8)
        else:
            plot_data = data
        im = ax.imshow(plot_data, extent=extent, aspect='equal', origin='lower')
    else:
        im = ax.imshow(data, cmap='gray', extent=extent, aspect='equal', origin='lower')

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)

    # Create lithology legend using the mapping dictionary
    patches = []
    if q_number and q_number in lithology_mappings:
        # For color images, we need to map colors to lithology IDs
        if data.ndim == 3:
            # Create a mapping from color to lithology ID
            color_to_id = {}
            # This assumes the color values correspond to lithology IDs in order
            # You may need to adjust this based on how your data is structured
            for i, color in enumerate(rgb_colors):
                lith_id = i + 1  # Assuming lithology IDs start at 1
                color_to_id[color] = lith_id

            # Create legend patches with proper names
            for color in rgb_colors:
                lith_id = color_to_id.get(color, 0)
                lith_name = lithology_mappings[q_number].get(lith_id, f"Unknown {lith_id}")
                patches.append(mpatches.Patch(color=np.array(color) / 255,
                                              label=f"{lith_name} (ID: {lith_id})"))
        else:
            # For grayscale images, use the unique values directly
            for value in unique_colors:
                lith_id = int(value)
                lith_name = lithology_mappings[q_number].get(lith_id, f"Unknown {lith_id}")
                patches.append(mpatches.Patch(color=plt.cm.gray(value / 255),
                                              label=f"{lith_name} (ID: {lith_id})"))
    else:
        # Fallback to generic legend if no mapping available
        for i, color in enumerate(unique_colors if rgb_colors is None else rgb_colors):
            if rgb_colors is not None:
                patches.append(mpatches.Patch(color=np.array(color) / 255,
                                              label=f'Lithology{i + 1:02d}'))
            else:
                patches.append(mpatches.Patch(color=plt.cm.gray(color / 255),
                                              label=f'Lithology{i + 1:02d}'))

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


def calculate_local_gci(lithology_map, window_size=5):
    """
    Calculate local Geological Complexity Index (GCI) based on lithology variance.

    Improved formula with better normalization and NaN handling
    """
    # Pad the lithology map to handle edges
    padded = np.pad(lithology_map, window_size // 2, mode='reflect')

    # Initialize output
    gci_map = np.zeros_like(lithology_map, dtype=np.float32)

    # Theoretical maximums (adjusted for 5x5 window)
    max_lith_count = min(8, window_size ** 2)  # Realistic maximum unique lithologies
    max_border_length = 4 * (window_size - 1)  # Maximum possible borders

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


def calculate_moving_correlation(tmi, gravity, window_size=5):
    """Calculate moving-window Pearson correlation between TMI and Gravity."""
    # Pad the data to handle edges
    pad_size = window_size // 2
    tmi_padded = np.pad(tmi, pad_size, mode='reflect')
    grav_padded = np.pad(gravity, pad_size, mode='reflect')

    # Initialize output
    corr_map = np.zeros_like(tmi, dtype=np.float32)

    # Calculate correlation for each window
    for i in range(tmi.shape[0]):
        for j in range(tmi.shape[1]):
            tmi_window = tmi_padded[i:i + window_size, j:j + window_size].flatten()
            grav_window = grav_padded[i:i + window_size, j:j + window_size].flatten()

            # Calculate Pearson correlation
            corr, _ = pearsonr(tmi_window, grav_window)
            corr_map[i, j] = corr

    return corr_map


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
    threshold = np.percentile(r_ratio, 90)  # Top 10% as hotspots
    hotspot_mask = r_ratio > threshold
    y, x = np.where(hotspot_mask)
    ax.scatter(x * 5, y * 5, c='yellow', s=10, alpha=0.7,
               label=f'Hotspots (R > {threshold:.1f}×global)')

    ax.set_title(f'{title}\nGlobal GCI: {global_gci:.4f}', fontsize=14, pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def enhanced_analysis(geophys_data, planview_data, output_dir):
    """Perform enhanced analysis with correlations and GCI mapping."""
    enhanced_dir = os.path.join(output_dir, "Enhanced_Analysis")
    os.makedirs(enhanced_dir, exist_ok=True)

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

        # Convert lithology to single channel if RGB
        if lithology.ndim == 3:
            lithology = (lithology[..., 0] << 16) | (lithology[..., 1] << 8) | lithology[..., 2]

        # Calculate moving-window correlations
        corr_map = calculate_moving_correlation(tmi, gravity)

        # Calculate local GCI
        gci_map, lith_count, border_length = calculate_local_gci(lithology)

        # Get global GCI for this model
        global_gci = GLOBAL_GCI_VALUES.get(q_key, 0.5)

        # Create feature matrix for VIF calculation
        features = np.stack([tmi, gravity,
                             geophys_data[mag_key]['1VD'],
                             geophys_data[grav_key]['1VD']], axis=-1)
        vif = calculate_vif(features)

        # Save all metrics
        np.savez_compressed(
            os.path.join(enhanced_dir, f"{q_key}_enhanced_metrics.npz"),
            correlation_map=corr_map,
            gci_map=gci_map,
            lithology_count=lith_count,
            border_length=border_length,
            vif=vif,
            global_gci=global_gci
        )

        # Plot collinearity heatmap with GCI
        plot_collinearity_heatmap(
            corr_map, gci_map, global_gci,
            f"Collinearity and GCI Zones - {q_key}",
            os.path.join(enhanced_dir, f"{q_key}_collinearity_gci.png")
        )

        # Create summary statistics
        stats = {
            'model': q_key,
            'global_gci': global_gci,
            'mean_local_gci': np.mean(gci_map),
            'mean_correlation': np.mean(corr_map),
            'vif_tmi': vif[0],
            'vif_gravity': vif[1],
            'vif_tmi_1vd': vif[2],
            'vif_gravity_1vd': vif[3],
            'mean_r_ratio': np.mean(block_reduce(corr_map, (5, 5), np.mean)) / np.mean(corr_map)
        }

        # Save stats to CSV (append if file exists)
        stats_file = os.path.join(enhanced_dir, "summary_statistics.csv")
        if os.path.exists(stats_file):
            existing = pd.read_csv(stats_file)
            updated = pd.concat([existing, pd.DataFrame([stats])], ignore_index=True)
            updated.to_csv(stats_file, index=False)
        else:
            pd.DataFrame([stats]).to_csv(stats_file, index=False)


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

        # Expand planview to match geophysics size
        expanded_pv = expand_planview_to_geophysics_size(pv_data)

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

            # Convert RGB to single-channel lithology IDs if needed
            if expanded_pv.ndim == 3:
                lithology_map = (expanded_pv[..., 0].astype(np.uint32) << 16 |
                                 (expanded_pv[..., 1].astype(np.uint32) << 8) |
                                 expanded_pv[..., 2].astype(np.uint32))
            else:
                lithology_map = expanded_pv

            # Get lithology mapping for this model
            lith_mapping = lithology_mappings.get(q_number, {})

            # Generate lithology geophysics plot
            plot_lithology_geophysics(
                q_number,
                lithology_map,
                tmi,
                gravity,
                lith_mapping,
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