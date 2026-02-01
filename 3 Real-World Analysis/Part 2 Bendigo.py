import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance_matrix, cKDTree
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


def filter_by_utm_range(data, utm_range):
    """Filter data points to only include those within specified UTM range"""
    if data is None or len(data) == 0:
        return data

    # Unpack UTM range
    min_easting, max_easting, min_northing, max_northing = utm_range

    # For coordinate data (Geo1, Geo2)
    if data.shape[1] >= 2:
        mask = (data[:, 0] >= min_easting) & (data[:, 0] <= max_easting) & \
               (data[:, 1] >= min_northing) & (data[:, 1] <= max_northing)
        return data[mask]

    return data


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

    # Define the specific UTM range we want to analyze
    focus_utm_range = (245000, 305000, 5855000, 5895000)

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

    # Create the full grid once using GeoA1
    UTME_full, UTMN_full = create_utm_grid(GeoA1)
    grid_shape_full = UTME_full.shape
    grid_coords_full = np.column_stack([UTME_full.ravel(), UTMN_full.ravel()])
    tree_full = cKDTree(grid_coords_full)

    # Filter geological data to focus area
    Geo1_focus = filter_by_utm_range(Geo1, focus_utm_range)
    Geo2_focus = filter_by_utm_range(Geo2, focus_utm_range)

    # Combine geological data and get rock types (both full and focus area)
    Rock_full = np.vstack([Geo1, Geo2]).astype(float)

    # For focus area, only use samples within the UTM range
    Rock_focus = None
    if Geo1_focus is not None and Geo2_focus is not None and len(Geo1_focus) > 0 and len(Geo2_focus) > 0:
        Rock_focus = np.vstack([Geo1_focus, Geo2_focus]).astype(float)

    # Process both full dataset and focus area
    for dataset, suffix in zip([Rock_full, Rock_focus], ['', '_FocusArea']):
        if dataset is None or len(dataset) == 0:
            print(f"No data found for {suffix} dataset")
            continue

        rock_coords = dataset[:, :2]  # UTM coordinates
        rock_types = dataset[:, 2].astype(int)
        unique_rock_types = np.unique(rock_types)

        # Find nearest grid points for each rock sample IN THE FULL GRID
        _, rock_grid_indices = tree_full.query(rock_coords, k=1)

        # Prepare feature matrix using the FULL GRID indices
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
        output_dir = os.path.join(input_dir, f'PCA_Results{suffix}')
        os.makedirs(output_dir, exist_ok=True)

        # Create mapping from rock types to category names
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
        plt.title(f'PCA of Victoria Rock Samples{suffix} (All Data Points)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'PCA_All_Points{suffix}.png'), dpi=300, bbox_inches='tight')
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
        plt.title(f'PCA of Victoria Rock Samples{suffix} (Centers & Standard Deviations)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'PCA_Centers{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"PCA visualizations saved to: {output_dir}")

        # Enhanced PCA analysis with more components
        pca = PCA()  # No n_components specified - will compute all components
        pca_result = pca.fit_transform(features_scaled)

        # Create output directory
        output_dir = os.path.join(input_dir, f'Enhanced_PCA_Results{suffix}')
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save variance explained information
        variance_table = pd.DataFrame({
            'Principal Component': [f'PC{i + 1}' for i in range(len(pca.explained_variance_ratio_))],
            'Variance Explained': pca.explained_variance_ratio_,
            'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        variance_table.to_csv(os.path.join(output_dir, f'variance_explained{suffix}.csv'), index=False)

        # 2. Create scree plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                 pca.explained_variance_ratio_, 'o-', label='Individual')
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                 np.cumsum(pca.explained_variance_ratio_), 's-', label='Cumulative')
        plt.axhline(y=0.05, color='r', linestyle='--', label='5% threshold')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title(f'Scree Plot of Explained Variance{suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'scree_plot{suffix}.png'), dpi=300)
        plt.close()

        # 3. Enhanced 2D PCA plots (existing functionality but with more components)
        significant_pcs = [i for i, var in enumerate(pca.explained_variance_ratio_)
                           if var > 0.05]  # Components explaining >5% variance

        # Generate all possible pairs of significant PCs
        from itertools import combinations
        pc_pairs = list(combinations(significant_pcs, 2))

        for pc1, pc2 in pc_pairs:
            plt.figure(figsize=(12, 8))
            for i, rock_type in enumerate(unique_rock_types):
                if rock_type == 0:
                    continue

                mask = (rock_types == rock_type)
                color = Bcolor[i % len(Bcolor)]
                category = rock_type_to_category.get(rock_type, f"Unknown Type {rock_type}")

                plt.scatter(pca_result[mask, pc1], pca_result[mask, pc2],
                            color=color, label=f'{rock_type} {category}',
                            alpha=0.6, s=30)

            plt.xlabel(f'PC{pc1 + 1} (%.2f%%)' % (pca.explained_variance_ratio_[pc1] * 100))
            plt.ylabel(f'PC{pc2 + 1} (%.2f%%)' % (pca.explained_variance_ratio_[pc2] * 100))
            plt.title(f'PCA of Victoria Rock Samples{suffix} (PC{pc1 + 1}-PC{pc2 + 1})')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'PCA_PC{pc1 + 1}_PC{pc2 + 1}{suffix}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # 4. 3D PCA plot if we have at least 3 significant components
        if len(significant_pcs) >= 3:
            pc1, pc2, pc3 = significant_pcs[:3]

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            for i, rock_type in enumerate(unique_rock_types):
                if rock_type == 0:
                    continue

                mask = (rock_types == rock_type)
                color = Bcolor[i % len(Bcolor)]
                category = rock_type_to_category.get(rock_type, f"Unknown Type {rock_type}")

                ax.scatter(pca_result[mask, pc1], pca_result[mask, pc2], pca_result[mask, pc3],
                           color=color, label=f'{rock_type} {category}',
                           alpha=0.6, s=30)

            ax.set_xlabel(f'PC{pc1 + 1} (%.2f%%)' % (pca.explained_variance_ratio_[pc1] * 100))
            ax.set_ylabel(f'PC{pc2 + 1} (%.2f%%)' % (pca.explained_variance_ratio_[pc2] * 100))
            ax.set_zlabel(f'PC{pc3 + 1} (%.2f%%)' % (pca.explained_variance_ratio_[pc3] * 100))
            ax.set_title(f'3D PCA of Victoria Rock Samples{suffix}')
            ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'3D_PCA_Plot{suffix}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # 5. Quantitative cluster separation metrics
        separation_metrics = []
        for i, rock_type in enumerate(unique_rock_types):
            if rock_type == 0:
                continue

            mask = (rock_types == rock_type)
            if np.sum(mask) == 0:
                continue

            category = rock_type_to_category.get(rock_type, f"Type {rock_type}")
            pca_data = pca_result[mask][:, significant_pcs]  # Only significant PCs

            # Calculate statistics for each significant PC
            pc_stats = {}
            for j, pc in enumerate(significant_pcs):
                pc_data = pca_result[mask, pc]
                pc_stats[f'PC{pc + 1}_mean'] = np.mean(pc_data)
                pc_stats[f'PC{pc + 1}_std'] = np.std(pc_data)
                pc_stats[f'PC{pc + 1}_min'] = np.min(pc_data)
                pc_stats[f'PC{pc + 1}_max'] = np.max(pc_data)

            # Calculate pairwise distances between samples (memory efficient)
            n_samples = len(pca_data)
            sample_size = min(100, n_samples)  # Use subset for large datasets
            idx = np.random.choice(n_samples, sample_size, replace=False)
            dist_matrix = distance_matrix(pca_data[idx], pca_data[idx])
            np.fill_diagonal(dist_matrix, np.nan)  # Ignore self-distances

            separation_metrics.append({
                'Rock_Type': rock_type,
                'Category': category,
                'N_Samples': n_samples,
                'Mean_Intra_Distance': np.nanmean(dist_matrix),
                'Std_Intra_Distance': np.nanstd(dist_matrix),
                **pc_stats
            })

        # Calculate inter-class distances (between rock types)
        if len(separation_metrics) > 1:
            rock_type_pairs = list(combinations(range(len(separation_metrics)), 2))
            inter_distances = []

            for (i, j) in rock_type_pairs:
                mask_i = (rock_types == unique_rock_types[i + 1])  # Skip 0
                mask_j = (rock_types == unique_rock_types[j + 1])

                # Sample subsets if large
                sample_i = pca_result[mask_i][:, significant_pcs]
                sample_j = pca_result[mask_j][:, significant_pcs]

                if len(sample_i) > 100:
                    sample_i = sample_i[np.random.choice(len(sample_i), 100, replace=False)]
                if len(sample_j) > 100:
                    sample_j = sample_j[np.random.choice(len(sample_j), 100, replace=False)]

                dist = distance_matrix(sample_i, sample_j)
                inter_distances.append({
                    'Type_A': separation_metrics[i]['Rock_Type'],
                    'Type_B': separation_metrics[j]['Rock_Type'],
                    'Category_A': separation_metrics[i]['Category'],
                    'Category_B': separation_metrics[j]['Category'],
                    'Mean_Distance': np.mean(dist),
                    'Std_Distance': np.std(dist),
                    'Min_Distance': np.min(dist),
                    'Max_Distance': np.max(dist)
                })

            # Save inter-class distances
            inter_distance_df = pd.DataFrame(inter_distances)
            inter_distance_df.to_csv(os.path.join(output_dir, f'inter_class_distances{suffix}.csv'), index=False)

        # Save intra-class metrics
        separation_df = pd.DataFrame(separation_metrics)
        separation_df.to_csv(os.path.join(output_dir, f'intra_class_metrics{suffix}.csv'), index=False)

        # New analysis: Spatial vs PCA distance correlation (memory-efficient version)
        def efficient_spatial_pca_analysis(coords, pca_result, rock_types, output_dir, suffix=''):
            """Analyze spatial-PCA relationships without full distance matrices"""
            n_samples = len(coords)
            sample_size = min(1000, n_samples)  # Adjust based on your memory capacity

            # Create subplot grid
            unique_types = np.unique(rock_types)
            n_types = len(unique_types) - 1
            cols = 4
            rows = int(np.ceil(n_types / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(28, 5 * rows))
            axes = axes.flatten() if n_types > 1 else [axes]

            for i, rock_type in enumerate(unique_rock_types):
                if rock_type == 0:
                    continue
                ax = axes[i - 1] if n_types > 1 else axes
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
            plt.savefig(os.path.join(output_dir, f'Spatial_PCA_Distance_Correlation{suffix}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # Run the efficient analysis
        efficient_spatial_pca_analysis(rock_coords, pca_result, rock_types, output_dir, suffix)

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
        plt.title(f'Interpolated PC1 Spatial Distribution{suffix}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Interpolated_PC1_Spatial{suffix}.png'),
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
        plt.title(f'Interpolated PC2 Spatial Distribution{suffix}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Interpolated_PC2_Spatial{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Added efficient spatial distribution analysis to output in: {output_dir}")


if __name__ == "__main__":
    input_dir = "VictoriaRock"
    process_victoria_rocks(input_dir)