import os
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom

# This part of the code is for data preparation

# Define lithology mappings for each Q file
lithology_mappings = {
    'Q001': {
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
    'Q003': {
        1: 'Sedimentary cover',
        2: 'Psammitic sediment',
        3: 'Felsic intrusive',
        4: 'Pelitic sediment'
    },
    'Q004': {
        1: 'Sedimentary cover',
        2: 'Psammitic sediment',
        3: 'Felsic intrusive',
        4: 'Pelitic sediment'
    },
    'Q005': {
        1: 'Chert',
        2: 'Carbonaceous rock',
        3: 'Sedimentary cover',
        4: 'Psammitic sediment',
        5: 'Felsic intrusive',
        6: 'Pelitic sediment'
    },
    'Q006': {
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


def calculate_1vd(matrix):
    """Calculate the first vertical derivative (1VD) with smoothing."""
    # Apply slight smoothing to reduce noise before differentiation
    smoothed = ndimage.gaussian_filter(matrix, sigma=0.5)
    # Calculate gradient with proper spacing
    grad_y = np.gradient(smoothed, axis=0)
    return grad_y


def calculate_tile(matrix, window_size=3):
    """Calculate tile (local mean)."""
    return ndimage.uniform_filter(matrix, size=window_size)


def calculate_analytical_signal(matrix):
    """Calculate the analytical signal magnitude with improved method."""
    # Apply slight smoothing to reduce noise
    smoothed = ndimage.gaussian_filter(matrix, sigma=0.5)

    # Calculate gradients
    grad_y, grad_x = np.gradient(smoothed)

    # Calculate analytical signal
    analytical_signal = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Apply slight smoothing to the result to create smoother surfaces
    analytical_signal = ndimage.gaussian_filter(analytical_signal, sigma=0.3)

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


def save_results(output_dir, geophys_data, planview_data):
    """Save all results including figures and data."""
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