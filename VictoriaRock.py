import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

# Define colors for rock types (using Bcolor from previous experiments)
Bcolor = np.array([
    [0, 0, 0],  # 0 Pad
    [0.4940, 0.1840, 0.5560],  # Regional metamorphic
    [1, 0, 1],  # Contact metamorphic
    [0.4660, 0.6740, 0.1880],  # Felsic volcanic
    [0, 0.5, 0],  # I/m/u volcanic
    [1, 0, 0],  # Felsic intrusive
    [0.6350, 0.0780, 0.1840],  # I/m/u intrusive
    [1, 1, 0],  # Psammitic sediment
    [0.9290, 0.6940, 0.1250],  # Pelitic sediment
    [0.75, 0.75, 0],  # Carbonaceous rock
    [0.8500, 0.3250, 0.0980],  # Chert
    [0.3010, 0.7450, 0.9330],  # Ironstone
    [0, 0, 0]  # Massive sulphide
])


def create_utm_grid(geo_data, grid_size=50):
    """Create UTM grid coordinates from geographic data"""
    x_coords = np.arange(np.floor(np.min(geo_data[:, 0])),
                         np.floor(np.max(geo_data[:, 0])) + grid_size,
                         grid_size)
    y_coords = np.arange(np.floor(np.min(geo_data[:, 1])),
                         np.floor(np.max(geo_data[:, 1])) + grid_size,
                         grid_size)
    return np.meshgrid(x_coords, y_coords)


def load_and_validate_data(file_path, is_matrix=True, replace_nan=0, replace_inf=0):
    """Load data from CSV with validation checks for NaN, Inf, and zeros"""
    try:
        if is_matrix:
            data = pd.read_csv(file_path, header=None).values
        else:
            data = pd.read_csv(file_path).values

        # Replace NaN and Inf with specified values
        data = np.nan_to_num(data, nan=replace_nan, posinf=replace_inf, neginf=replace_inf)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def process_victoria_rocks(input_dir):
    """Process Victoria rock samples data and create PCA visualizations"""
    # Define category names matching the Bcolor scheme
    category_names = [
        "Regional metamorphic",
        "Contact metamorphic",
        "Felsic volcanic",
        "I/m/u volcanic",
        "Felsic intrusive",
        "I/m/u intrusive",
        "Psammitic sediment",
        "Pelitic sediment",
        "Carbonaceous rock",
        "Chert",
        "Ironstone",
        "Massive sulphide"
    ]

    # Load all input data
    GeoA1 = load_and_validate_data(os.path.join(input_dir, 'GeoA1.csv'))
    TMI = load_and_validate_data(os.path.join(input_dir, 'Mag.csv'))
    VD1 = load_and_validate_data(os.path.join(input_dir, 'Mag1vd.csv'))
    Potassium = load_and_validate_data(os.path.join(input_dir, 'Potassium.csv'))
    Thorium = load_and_validate_data(os.path.join(input_dir, 'Thorium.csv'))
    Totalcount = load_and_validate_data(os.path.join(input_dir, 'Totalcount.csv'))
    Uran = load_and_validate_data(os.path.join(input_dir, 'Uranium.csv'))
    G4 = load_and_validate_data(os.path.join(input_dir, 'G4.csv'))
    ASM = load_and_validate_data(os.path.join(input_dir, 'ASM.csv'))

    # Load geological data
    Geo1 = load_and_validate_data(os.path.join(input_dir, 'Geo1.csv'), is_matrix=False)
    Geo2 = load_and_validate_data(os.path.join(input_dir, 'Geo2.csv'), is_matrix=False)

    # Combine geological data and get rock types
    Rock = np.vstack([Geo1, Geo2]).astype(float)
    rock_coords = Rock[:, :2]  # UTM coordinates
    rock_types = Rock[:, 2].astype(int)
    unique_rock_types = np.unique(rock_types)

    # Create UTM grids and get grid coordinates
    UTME, UTMN = create_utm_grid(GeoA1)
    grid_shape = UTME.shape

    # Create coordinate pairs for each grid point
    grid_coords = np.column_stack([UTME.ravel(), UTMN.ravel()])

    # Find nearest grid points for each rock sample
    from scipy.spatial import cKDTree
    tree = cKDTree(grid_coords)
    _, rock_grid_indices = tree.query(rock_coords, k=1)

    # Prepare feature matrix (only for grid points with rock samples)
    features = np.stack([
        TMI.ravel()[rock_grid_indices],
        VD1.ravel()[rock_grid_indices],
        Potassium.ravel()[rock_grid_indices],
        Thorium.ravel()[rock_grid_indices],
        Totalcount.ravel()[rock_grid_indices],
        Uran.ravel()[rock_grid_indices],
        gaussian_filter(G4, sigma=1).ravel()[rock_grid_indices],
        ASM.ravel()[rock_grid_indices]
    ], axis=-1)

    # Standardize features and perform PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)

    # Create output directory
    output_dir = os.path.join(input_dir, 'PCA_Results')
    os.makedirs(output_dir, exist_ok=True)

    # Create mapping from rock types to category names
    # (You'll need to define which rock types correspond to which categories)
    rock_type_to_category = {
        1: "Regional metamorphic",
        2: "Contact metamorphic",
        3: "Felsic volcanic",
        4: "I/m/u volcanic",
        5: "Felsic intrusive",
        6: "I/m/u intrusive",
        7: "Psammitic sediment",
        8: "Pelitic sediment",
        9: "Carbonaceous rock",
        10: "Chert",
        11: "Ironstone",
        12: "Massive sulphide"
    }

    # Plot 1: Detailed scatter plot with all rock types
    plt.figure(figsize=(14, 10))
    for i, rock_type in enumerate(unique_rock_types):
        if rock_type == 0:
            continue
        print(i, rock_type)

        mask = (rock_types == rock_type)
        color = Bcolor[i % len(Bcolor)]

        # Get category name for this rock type
        category = rock_type_to_category.get(rock_type, f"Unknown Type {rock_type}")

        plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                    color=color,
                    label=f'{rock_type} {category}',
                    alpha=0.6, s=30)

    plt.xlabel('PC1 (%.2f%%)' % (pca.explained_variance_ratio_[0] * 100))
    plt.ylabel('PC2 (%.2f%%)' % (pca.explained_variance_ratio_[1] * 100))
    plt.title('PCA of Victoria Rock Samples (All Data Points)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'PCA_All_Points.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Simplified plot with centers and standard deviations
    plt.figure(figsize=(12, 8))
    for i, rock_type in enumerate(unique_rock_types):
        if rock_type == 0:
            continue
        print(i, rock_type)
        mask = (rock_types == rock_type)
        if np.sum(mask) == 0:
            continue

        color = Bcolor[i % len(Bcolor)]
        center = np.mean(pca_result[mask], axis=0)
        std = np.std(pca_result[mask], axis=0)

        # Get category name for this rock type
        category = rock_type_to_category.get(rock_type, f"Unknown Type {rock_type}")

        plt.scatter(center[0], center[1], color=color,
                    s=200,
                    label=f'{rock_type} {category}',
                    edgecolor='black', linewidth=1.5,
                    zorder=3)  # Higher z-order ensures it's on top

        ellipse = Ellipse(center, width=std[0] * 2, height=std[1] * 2,
                          angle=0, color=color, alpha=0.2,
                          zorder=2)  # Lower z-order keeps it behind
        plt.gca().add_patch(ellipse)

    plt.xlabel('PC1 (%.2f%%)' % (pca.explained_variance_ratio_[0] * 100))
    plt.ylabel('PC2 (%.2f%%)' % (pca.explained_variance_ratio_[1] * 100))
    plt.title('PCA of Victoria Rock Samples (Centers & Standard Deviations)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'PCA_Centers.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PCA visualizations saved to: {output_dir}")

    # New analysis: Spatial vs PCA distance correlation (memory-efficient version)
    def efficient_spatial_pca_analysis(coords, pca_result, rock_types, output_dir):
            """Analyze spatial-PCA relationships without full distance matrices"""
            n_samples = len(coords)
            sample_size = min(1000, n_samples)  # Adjust based on your memory capacity

            # Create subplot grid
            unique_types = np.unique(rock_types)
            n_types = len(unique_types)-1
            cols = 4
            rows = int(np.ceil(n_types / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(28, 5 * rows))
            axes = axes.flatten() if n_types > 1 else [axes]

            for i, rock_type in enumerate(unique_rock_types):
                if rock_type == 0:
                    continue
                ax = axes[i-1]
                mask = (rock_types == rock_type)
                type_coords = coords[mask]
                type_pca = pca_result[mask]
                n_type_samples = len(type_coords)

                if n_type_samples < 2:
                    ax.text(0.5, 0.5, f"Only 1 sample\nfor Type {rock_type}",
                            ha='center', va='center')
                    ax.set_title(f'Rock Type {rock_type}')
                    continue

                # Randomly sample pairs to avoid memory issues
                n_pairs = min(10000, n_type_samples * (n_type_samples - 1) // 2)
                idx1, idx2 = np.triu_indices(n_type_samples, k=1)
                selected = np.random.choice(len(idx1), n_pairs, replace=False)

                # Calculate distances for sampled pairs
                spatial_dists = np.linalg.norm(type_coords[idx1[selected]] - type_coords[idx2[selected]], axis=1)
                pca_dists = np.linalg.norm(type_pca[idx1[selected]] - type_pca[idx2[selected]], axis=1)

                # Plot and calculate correlation
                color = Bcolor[i % len(Bcolor)]
                ax.scatter(spatial_dists, pca_dists, color=color, alpha=0.6, s=20)

                try:
                    from scipy.stats import linregress
                    slope, intercept, r_value, _, _ = linregress(spatial_dists, pca_dists)
                    x_vals = np.array([min(spatial_dists), max(spatial_dists)])
                    y_vals = intercept + slope * x_vals
                    ax.plot(x_vals, y_vals, '--', color='black',
                            label=f'R²={r_value ** 2:.2f}')
                except:
                    pass

                category = rock_type_to_category.get(rock_type, f"Type {rock_type}")
                ax.set_title(f'{rock_type} {category}')
                ax.set_xlabel('Spatial Distance (m)')
                ax.set_ylabel('PCA Distance')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Remove empty subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'Spatial_PCA_Distance_Correlation.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

    # Run the efficient analysis
    efficient_spatial_pca_analysis(rock_coords, pca_result, rock_types, output_dir)

    # Create gridded interpolations for spatial visualization
    def create_interpolation_grid(coords, values, grid_shape):
            """Create gridded interpolation of values"""
            from scipy.interpolate import griddata
            grid_x, grid_y = np.mgrid[
                             np.min(coords[:, 0]):np.max(coords[:, 0]):grid_shape[0] * 1j,
                             np.min(coords[:, 1]):np.max(coords[:, 1]):grid_shape[1] * 1j
                             ]
            grid_z = griddata(coords, values, (grid_x, grid_y), method='linear')
            return grid_x, grid_y, grid_z

    # PC1 spatial distribution
    grid_x, grid_y, pc1_grid = create_interpolation_grid(
            rock_coords, pca_result[:, 0], (100, 100))

    plt.figure(figsize=(14, 10))
    plt.imshow(pc1_grid.T, extent=(np.min(rock_coords[:, 0]), np.max(rock_coords[:, 0]),
                                       np.min(rock_coords[:, 1]), np.max(rock_coords[:, 1])),
                   origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='PC1 Value')
    plt.scatter(rock_coords[:, 0], rock_coords[:, 1], c='white', s=5, alpha=0.3)
    plt.xlabel('UTM Easting (m)')
    plt.ylabel('UTM Northing (m)')
    plt.title('Interpolated PC1 Spatial Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Interpolated_PC1_Spatial.png'),
                    dpi=300, bbox_inches='tight')
    plt.close()

    # PC2 spatial distribution
    _, _, pc2_grid = create_interpolation_grid(
            rock_coords, pca_result[:, 1], (100, 100))

    plt.figure(figsize=(14, 10))
    plt.imshow(pc2_grid.T, extent=(np.min(rock_coords[:, 0]), np.max(rock_coords[:, 0]),
                                       np.min(rock_coords[:, 1]), np.max(rock_coords[:, 1])),
                   origin='lower', cmap='plasma', aspect='auto')
    plt.colorbar(label='PC2 Value')
    plt.scatter(rock_coords[:, 0], rock_coords[:, 1], c='white', s=5, alpha=0.3)
    plt.xlabel('UTM Easting (m)')
    plt.ylabel('UTM Northing (m)')
    plt.title('Interpolated PC2 Spatial Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Interpolated_PC2_Spatial.png'),
                    dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Added efficient spatial distribution analysis to output in: {output_dir}")


if __name__ == "__main__":
    input_dir = "VictoriaRock"
    process_victoria_rocks(input_dir)