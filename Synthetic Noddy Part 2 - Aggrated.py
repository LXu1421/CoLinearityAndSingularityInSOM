"""
Enhanced geophysical processing with Self-Organizing Map and lithology priors integration.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import argparse
import rasterio
from rasterio.transform import from_bounds
from minisom import MiniSom
import subprocess
from collections import Counter
import warnings
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import RectBivariateSpline
from skimage.restoration import inpaint
from scipy.ndimage import zoom
import json

warnings.filterwarnings('ignore')

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


def load_processed_data(data_dir):
    """Load processed data from NPZ files created by the first script."""
    geophys_data = {}
    planview_data = {}

    # Load geophysical data
    for i in range(1, 7):
        for field_type in ['Grav', 'Mag']:
            file_name = f"Q{i:03d}-{field_type}_processed.npz"
            file_path = os.path.join(data_dir, file_name)

            if os.path.exists(file_path):
                try:
                    data = np.load(file_path)
                    geophys_data[f"Q{i:03d}-{field_type}.bmp"] = dict(data)
                    data.close()
                    print(f"Loaded geophysical data: {file_name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

    # Load planview data
    for i in range(1, 7):
        file_name = f"Q{i:03d}-planview_processed.npz"
        file_path = os.path.join(data_dir, file_name)

        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                planview_data[f"Q{i:03d}-planview.bmp"] = data['expanded']
                data.close()
                print(f"Loaded planview data: {file_name}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

    return geophys_data, planview_data


def rgb_to_lithology_id(rgb_array, q_number):
    """Convert RGB values to lithology IDs based on color mapping."""
    if rgb_array.ndim == 3:
        # Create unique color identifier
        color_map = (rgb_array[..., 0].astype(np.uint32) << 16 |
                     rgb_array[..., 1].astype(np.uint32) << 8 |
                     rgb_array[..., 2].astype(np.uint32))
    else:
        return rgb_array

    # Get unique colors and assign sequential IDs
    unique_colors = np.unique(color_map)
    lithology_ids = np.zeros_like(color_map, dtype=np.uint8)

    for i, color in enumerate(unique_colors[1:], 1):  # Skip background (0)
        lithology_ids[color_map == color] = min(i, len(lithology_mappings.get(q_number, {})))

    return lithology_ids


def prepare_features_for_som(geophys_data, q_number):
    """Prepare integrated 2D features for SOM training."""
    features_list = []
    feature_names = []

    # Extract features from both gravity and magnetic data
    grav_key = f"{q_number}-Grav.bmp"
    mag_key = f"{q_number}-Mag.bmp"

    if grav_key in geophys_data:
        grav_data = geophys_data[grav_key]
        features_list.extend([
            grav_data['raw'].flatten(),
            grav_data['1VD'].flatten(),
            grav_data['analytical_signal'].flatten()
        ])
        feature_names.extend(['grav_raw', 'grav_1VD', 'grav_analytical'])

    if mag_key in geophys_data:
        mag_data = geophys_data[mag_key]
        features_list.extend([
            mag_data['raw'].flatten(),
            mag_data['1VD'].flatten(),
            mag_data['analytical_signal'].flatten()
        ])
        feature_names.extend(['mag_raw', 'mag_1VD', 'mag_analytical'])

    if not features_list:
        return None, None, None

    # Stack features and standardize
    features = np.column_stack(features_list)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, feature_names, scaler


def initialize_som_with_priors(som_shape, n_features, lithology_map, q_number, features=None):
    """Initialize SOM codebook vectors with lithology priors."""
    som = MiniSom(som_shape[0], som_shape[1], n_features,
                  sigma=1.5, learning_rate=0.5, random_seed=42)

    # Initialize with random weights using features if available, otherwise random
    if features is not None:
        som.random_weights_init(features)
    else:
        # Create dummy data for initialization if no features provided
        dummy_data = np.random.random((100, n_features))
        som.random_weights_init(dummy_data)

    return som


def train_som_with_priors(features, lithology_map, q_number, som_shape=(20, 20), n_iterations=10000):
    """Train SOM with lithology priors initialization."""
    print(f"Training SOM with shape {som_shape} for {n_iterations} iterations...")

    # Initialize SOM with priors
    som = initialize_som_with_priors(som_shape, features.shape[1], lithology_map, q_number, features)

    # Train the SOM
    som.train(features, n_iterations, verbose=True)

    # Get cluster assignments
    cluster_map = np.zeros(len(features), dtype=np.int32)
    for i, x in enumerate(features):
        winner = som.winner(x)
        cluster_map[i] = winner[0] * som_shape[1] + winner[1] + 1  # +1 to avoid 0 label

    return som, cluster_map


def save_npz_file(path, array, shape):
    """Save array as NPZ file."""
    try:
        array_2d = array.reshape(shape)
        np.savez_compressed(path, data=array_2d)
        print(f"Saved NPZ: {path}")
    except Exception as e:
        print(f"Error saving NPZ {path}: {str(e)}")
        raise


def load_npz_file(path):
    """Load data from NPZ file."""
    try:
        data = np.load(path)
        return data['data']
    except Exception as e:
        print(f"Error loading NPZ {path}: {str(e)}")
        raise


def run_aggregate_clusters(clusters_path, geology_path, features_paths, target_k, output_path, temp_dir):
    """Run the aggregate_clusters_postprocess.py script with NPZ files."""
    try:
        cmd = [
            'python', 'aggregate_clusters_postprocess.py',
            '--clusters', clusters_path,
            '--geology', geology_path,
            '--target_k', str(target_k),
            '--out', output_path,
            '--log', os.path.join(temp_dir, 'aggregate_log.csv'),
            '--map', os.path.join(temp_dir, 'aggregate_map.csv'),
            '--report', os.path.join(temp_dir, 'aggregate_report.txt')
        ]

        if features_paths:
            cmd.extend(['--features'] + features_paths)

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error in aggregate_clusters: {result.stderr}")
            return False

        return True
    except Exception as e:
        print(f"Exception running aggregate_clusters: {str(e)}")
        return False


def run_merge_clusters(clusters_path, geology_path, output_path, temp_dir, purity=0.6, sieve_pixels=25):
    """Run the merge_clusters_to_geology.py script with NPZ files."""
    try:
        cmd = [
            'python', 'merge_clusters_to_geology.py',
            '--clusters', clusters_path,
            '--geology', geology_path,
            '--out_merged', output_path,
            '--out_map', os.path.join(temp_dir, 'merge_map.csv'),
            '--report', os.path.join(temp_dir, 'merge_report.csv'),
            '--purity', str(purity),
            '--sieve_pixels', str(sieve_pixels)
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error in merge_clusters: {result.stderr}")
            return False

        return True
    except Exception as e:
        print(f"Exception running merge_clusters: {str(e)}")
        return False


def assign_final_lithology_ids(merged_clusters, lithology_priors, q_number):
    """Assign final lithology IDs using argmax rule with data likelihood and section prior."""
    lith_mapping = lithology_mappings.get(q_number, {})

    final_labels = np.zeros_like(merged_clusters)
    mapping_table = {}

    unique_clusters = np.unique(merged_clusters)
    unique_clusters = unique_clusters[unique_clusters > 0]

    for cluster_id in unique_clusters:
        mask = merged_clusters == cluster_id
        prior_values = lithology_priors[mask]

        # Find most common lithology in this cluster region
        if len(prior_values) > 0:
            most_common_lith = Counter(prior_values[prior_values > 0]).most_common(1)
            if most_common_lith:
                assigned_lith = most_common_lith[0][0]
                final_labels[mask] = assigned_lith
                # Convert uint8 key to string for JSON serialization
                mapping_table[str(cluster_id)] = {
                    'lithology_id': int(assigned_lith),  # Ensure int for JSON
                    'lithology_name': lith_mapping.get(int(assigned_lith), f'Unknown_{assigned_lith}'),
                    'pixels': int(np.sum(mask))  # Ensure int for JSON
                }

    return final_labels, mapping_table


def calculate_metrics(true_labels, predicted_labels, features):
    """Calculate clustering and classification metrics."""
    metrics = {}

    # Remove background pixels (0 labels)
    mask = (true_labels > 0) & (predicted_labels > 0)

    if np.sum(mask) > 0:
        true_masked = true_labels[mask]
        pred_masked = predicted_labels[mask]

        # Clustering metrics
        metrics['adjusted_rand_score'] = float(adjusted_rand_score(true_masked, pred_masked))

        # Silhouette score (if feasible - sample for large datasets)
        if len(pred_masked) > 10000:
            indices = np.random.choice(len(pred_masked), 5000, replace=False)
            features_sample = features[mask][indices]
            pred_sample = pred_masked[indices]
        else:
            features_sample = features[mask]
            pred_sample = pred_masked

        try:
            if len(np.unique(pred_sample)) > 1:
                metrics['silhouette_score'] = float(silhouette_score(features_sample, pred_sample))
        except:
            metrics['silhouette_score'] = -1.0

        # Classification accuracy
        metrics['accuracy'] = float(np.mean(true_masked == pred_masked))

        # Per-class metrics - convert keys to strings for JSON
        unique_true = np.unique(true_masked)
        class_metrics = {}
        for class_id in unique_true:
            class_mask = true_masked == class_id
            if np.sum(class_mask) > 0:
                class_pred = pred_masked[class_mask]
                class_acc = float(np.mean(class_pred == class_id))
                class_metrics[str(int(class_id))] = {  # Convert to string for JSON
                    'accuracy': class_acc,
                    'pixel_count': int(np.sum(class_mask))
                }
        metrics['per_class'] = class_metrics

    return metrics


def create_high_dpi_figures(final_labels, lithology_map, geophys_data, q_number, output_dir, dpi=300):
    """Create high-resolution figures at 300+ DPI."""
    lith_mapping = lithology_mappings.get(q_number, {})

    # Create figure directory
    fig_dir = os.path.join(output_dir, "High_DPI_Figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Lithology Classification Results - {q_number}', fontsize=16, fontweight='bold')

    # Common settings for all subplots
    for ax in axes.flat:
        ax.set_xticks(np.arange(0, lithology_map.shape[1], 100))
        ax.set_yticks(np.arange(0, lithology_map.shape[0], 100))
        ax.grid(True, color='lightgray', linestyle='--', linewidth=0.3)
        ax.set_ylim(-0.5, lithology_map.shape[0] - 0.5)  # Y increases upwards
        ax.set_aspect('equal')  # Optional: keep square pixels

    # Original lithology map
    im1 = axes[0, 0].imshow(lithology_map, cmap='tab10', interpolation='nearest')
    axes[0, 0].set_title('Original Lithology Map')

    # Final classified map
    im2 = axes[0, 1].imshow(final_labels, cmap='tab10', interpolation='nearest')
    axes[0, 1].set_title('SOM Classification Result')

    # Difference map
    diff = (lithology_map != final_labels).astype(int)
    im3 = axes[0, 2].imshow(diff, cmap='RdYlBu_r', interpolation='nearest')
    axes[0, 2].set_title('Classification Differences')

    # Geophysical data
    grav_key = f"{q_number}-Grav.bmp"
    mag_key = f"{q_number}-Mag.bmp"

    if grav_key in geophys_data:
        im4 = axes[1, 0].imshow(geophys_data[grav_key]['raw'], cmap='RdBu_r')
        axes[1, 0].set_title('Gravity Data')
        #axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)

    if mag_key in geophys_data:
        im5 = axes[1, 1].imshow(geophys_data[mag_key]['raw'], cmap='RdBu_r')
        axes[1, 1].set_title('Magnetic Data')
        #axes[1, 1].axis('off')
        plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)

    # Legend for lithology
    unique_labels = np.unique(final_labels[final_labels > 0])
    legend_text = []
    for label in unique_labels:
        name = lith_mapping.get(int(label), f'Class_{label}')
        count = np.sum(final_labels == label)
        legend_text.append(f'{label}: {name} ({count} pixels)')

    axes[1, 2].text(0.1, 0.9, '\n'.join(legend_text), transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontsize=10)
    axes[1, 2].set_title('Classification Legend')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save at high DPI
    output_path = os.path.join(fig_dir, f'{q_number}_som_classification_results_{dpi}dpi.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"High-DPI figure saved: {output_path}")

    return output_path


def process_single_model_with_som(q_number, geophys_data, planview_data, output_dir):
    """Process a single geological model using SOM with lithology priors."""
    print(f"\n{'=' * 60}")
    print(f"Processing {q_number} with SOM and lithology priors")
    print(f"{'=' * 60}")

    # Get planview data
    pv_key = f"{q_number}-planview.bmp"
    if pv_key not in planview_data:
        print(f"No planview data for {q_number}")
        return None

    # Create temporary directory for intermediate files
    temp_dir = os.path.join(output_dir, "temp", q_number)
    os.makedirs(temp_dir, exist_ok=True)

    # Process lithology map
    expanded_pv = planview_data[pv_key]
    lithology_map = rgb_to_lithology_id(expanded_pv, q_number)

    # Prepare features for SOM
    features_scaled, feature_names, scaler = prepare_features_for_som(geophys_data, q_number)
    if features_scaled is None:
        print(f"No geophysical data available for {q_number}")
        return None

    # Check if features are valid
    if features_scaled.shape[0] == 0 or features_scaled.shape[1] == 0:
        print(f"Invalid features shape: {features_scaled.shape}")
        return None

    # Train SOM with lithology priors
    som_shape = (25, 25)
    som, initial_clusters = train_som_with_priors(
        features_scaled, lithology_map.flatten(), q_number,
        som_shape=som_shape, n_iterations=15000
    )

    # Save initial cluster map as NPZ
    initial_clusters_path = os.path.join(temp_dir, "initial_som_clusters.npz")
    save_npz_file(initial_clusters_path, initial_clusters, lithology_map.shape)

    # Save lithology map for processing as NPZ
    lithology_path = os.path.join(temp_dir, "lithology_map.npz")
    save_npz_file(lithology_path, lithology_map.flatten(), lithology_map.shape)

    # Save feature maps for aggregate processing as NPZ
    feature_paths = []
    for i, name in enumerate(feature_names):
        feature_path = os.path.join(temp_dir, f"feature_{name}.npz")
        save_npz_file(feature_path, features_scaled[:, i], lithology_map.shape)
        feature_paths.append(feature_path)

    # Step 1: Aggregate clusters to target K
    target_k = len(lithology_mappings.get(q_number, {}))
    aggregated_path = os.path.join(temp_dir, "aggregated_clusters.npz")

    print(f"\nStep 1: Aggregating to {target_k} clusters...")
    success = run_aggregate_clusters(
        initial_clusters_path, lithology_path, feature_paths,
        target_k, aggregated_path, temp_dir
    )

    if not success:
        print("Failed to aggregate clusters")
        return None

    # Step 2: Merge clusters to geology
    merged_path = os.path.join(temp_dir, "merged_clusters.npz")

    print("\nStep 2: Merging clusters to geology...")
    success = run_merge_clusters(
        aggregated_path, lithology_path, merged_path, temp_dir,
        purity=0.6, sieve_pixels=25
    )

    if not success:
        print("Failed to merge clusters")
        return None

    # Step 3: Assign final lithology IDs
    print("\nStep 3: Assigning final lithology IDs...")
    # Load merged clusters from NPZ
    merged_clusters = load_npz_file(merged_path)

    final_labels, mapping_table = assign_final_lithology_ids(
        merged_clusters, lithology_map, q_number
    )

    # Step 4: Export results
    print("\nStep 4: Exporting results...")

    # Save final label map as NPZ
    final_labels_path = os.path.join(output_dir, f"{q_number}_final_lithology_map.npz")
    save_npz_file(final_labels_path, final_labels.flatten(), final_labels.shape)

    # Save mapping table
    mapping_df = pd.DataFrame.from_dict(mapping_table, orient='index')
    mapping_df.index.name = 'cluster_id'
    mapping_path = os.path.join(output_dir, f"{q_number}_lithology_mapping.csv")
    mapping_df.to_csv(mapping_path)

    # Calculate metrics
    metrics = calculate_metrics(lithology_map.flatten(), final_labels.flatten(), features_scaled)

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{q_number}_classification_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Create high-DPI figures
    fig_path = create_high_dpi_figures(
        final_labels, lithology_map, geophys_data, q_number, output_dir, dpi=300
    )

    # Also save final labels as PNG for visualization
    png_path = os.path.join(output_dir, f"{q_number}_final_lithology_map.png")
    plt.figure(figsize=(12, 9))
    plt.imshow(final_labels, cmap='tab10')
    plt.title(f'Final Lithology Classification - {q_number}')
    plt.colorbar()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResults for {q_number}:")
    print(f"  Final labels (NPZ): {final_labels_path}")
    print(f"  Final labels (PNG): {png_path}")
    print(f"  Mapping table: {mapping_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  High-DPI figure: {fig_path}")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
    print(f"  Adjusted Rand Score: {metrics.get('adjusted_rand_score', 'N/A'):.3f}")

    return {
        'q_number': q_number,
        'final_labels_path': final_labels_path,
        'final_labels_png': png_path,
        'mapping_path': mapping_path,
        'metrics_path': metrics_path,
        'figure_path': fig_path,
        'metrics': metrics,
        'final_labels': final_labels,
        'mapping_table': mapping_table
    }


def enhanced_analysis_with_som(geophys_data, planview_data, output_dir):
    """Enhanced analysis using SOM with lithology priors for all available models."""
    print("\n" + "=" * 80)
    print("ENHANCED SOM-BASED LITHOLOGY CLASSIFICATION")
    print("=" * 80)

    results = {}

    # Process each available Q model
    available_q_numbers = []
    for pv_key in planview_data.keys():
        if pv_key.endswith('-planview.bmp'):
            q_number = pv_key[:4]
            available_q_numbers.append(q_number)

    print(f"Found {len(available_q_numbers)} models to process: {available_q_numbers}")

    for q_number in sorted(available_q_numbers):
        try:
            result = process_single_model_with_som(q_number, geophys_data, planview_data, output_dir)
            if result:
                results[q_number] = result
        except Exception as e:
            print(f"Error processing {q_number}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Create summary report
    summary_path = os.path.join(output_dir, "som_classification_summary.csv")
    summary_data = []

    for q_number, result in results.items():
        metrics = result.get('metrics', {})
        summary_data.append({
            'Q_Number': q_number,
            'Accuracy': metrics.get('accuracy', 0),
            'Adjusted_Rand_Score': metrics.get('adjusted_rand_score', 0),
            'Silhouette_Score': metrics.get('silhouette_score', -1),
            'Total_Classes': len(lithology_mappings.get(q_number, {})),
            'Final_Labels_File': os.path.basename(result['final_labels_path']),
            'Mapping_File': os.path.basename(result['mapping_path'])
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSummary report saved: {summary_path}")
    print(f"Successfully processed {len(results)} models")

    return results


def main():
    print("Starting SOM-enhanced processing...")

    # Create output directory
    output_dir = "SyntheticNoddy_SOM"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")

    # Load processed data from first script
    data_dir = "SyntheticNoddy/Data"  # Assuming first script output is here
    print(f"\nLoading processed data from: {data_dir}")

    geophys_data, planview_data = load_processed_data(data_dir)

    if geophys_data or planview_data:
        print(f"\nLoaded {len(geophys_data)} geophysical datasets and {len(planview_data)} planview datasets")
        print("\nPerforming enhanced SOM-based analysis...")
        enhanced_analysis_with_som(geophys_data, planview_data, output_dir)
        print("\nEnhanced SOM analysis complete!")
    else:
        print("\nNo processed data found. Please run the first script first.")


if __name__ == "__main__":
    main()