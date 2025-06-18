import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from minisom import MiniSom
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

# This part of the code completes the SOM experiments and the PCA plot

def run_som_experiments(geophys_data, planview_data, output_dir):
    """Run all SOM experiments and generate summary figures."""
    os.makedirs(os.path.join(output_dir, "SOM_Results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "SOM_Results", "PCA_Plots"), exist_ok=True)

    for i in range(1, 7):
        grav_key = f"Q{i:03d}-Grav.bmp"
        mag_key = f"Q{i:03d}-Mag.bmp"
        pv_key = f"Q{i:03d}-PlanView.bmp"

        if grav_key not in geophys_data or mag_key not in geophys_data or pv_key not in planview_data:
            continue

        grav_raw = geophys_data[grav_key]['raw']
        mag_raw = geophys_data[mag_key]['raw']
        pv_data = planview_data[pv_key]['expanded']

        if pv_data.ndim == 3:
            pv_data = pv_data.astype(np.uint32)
            unique_colors = np.unique(
                (pv_data[..., 0] << 16) | (pv_data[..., 1] << 8) | pv_data[..., 2],
                axis=None
            )
            color_to_id = {color: idx for idx, color in enumerate(unique_colors)}
            pv_labels = np.vectorize(color_to_id.get)(
                (pv_data[..., 0] << 16) | (pv_data[..., 1] << 8) | pv_data[..., 2]
            )
            original_colors = []
            for color_val in unique_colors:
                r = (color_val >> 16) & 0xFF
                g = (color_val >> 8) & 0xFF
                b = color_val & 0xFF
                original_colors.append([r, g, b])
            original_colors = np.array(original_colors)
        else:
            unique_colors = np.unique(pv_data)
            color_to_id = {color: idx for idx, color in enumerate(unique_colors)}
            pv_labels = np.vectorize(color_to_id.get)(pv_data)
            original_colors = unique_colors

        lithology_map = {idx: f'Lithology{idx + 1:02d}' for idx in range(len(unique_colors))}

        # Prepare datasets with their own PCA transformers
        datasets = {
            'Raw': {
                'data': np.stack([mag_raw, grav_raw], axis=-1),
                'pca': PCA(n_components=2)
            },
            'All Features': {
                'data': np.stack([
                    mag_raw,
                    grav_raw,
                    geophys_data[mag_key]['tile'],
                    geophys_data[grav_key]['tile'],
                ], axis=-1),
                'pca': PCA(n_components=2)
            }
        }

        # Fit PCA for each dataset
        for name in datasets:
            flat_data = datasets[name]['data'].reshape(-1, datasets[name]['data'].shape[-1])
            datasets[name]['pca'].fit(flat_data)

        # Add PCA dataset (already in PCA space)
        datasets['PCA'] = {
            'data': datasets['All Features']['pca'].transform(
                datasets['All Features']['data'].reshape(-1, datasets['All Features']['data'].shape[-1])
            ).reshape(datasets['All Features']['data'].shape[0], datasets['All Features']['data'].shape[1], 2),
            'pca': None  # No PCA needed as data is already transformed
        }

        results = {}
        accuracy_rows = []
        for name, data_dict in datasets.items():
            results[name] = run_som(data_dict['data'], pv_labels, som_shape=(7, 7))

            dominant_lith_map = np.zeros_like(results[name]['cluster_map'])
            for cluster, summary in results[name]['cluster_summary'].items():
                mask = results[name]['cluster_map'] == cluster
                dominant_lith_map[mask] = summary['dominant']

            flat_true = pv_labels.ravel()
            flat_pred = dominant_lith_map.ravel()

            overall_acc = accuracy_score(flat_true, flat_pred)
            cm = confusion_matrix(flat_true, flat_pred)
            class_acc = cm.diagonal() / cm.sum(axis=1)

            # Calculate lithology standard deviations
            lithology_stds = []
            for lith_id in range(len(lithology_map)):
                mask = results[name]['flat_labels'] == lith_id
                lith_data = results[name]['scaled_data'][mask]
                if len(lith_data) > 0:
                    stds = np.std(lith_data, axis=0)
                    lithology_stds.append(np.mean(stds))
                else:
                    lithology_stds.append(0)

            # Get cluster centers in PCA space
            if name == 'PCA':
                cluster_centers = results[name]['som'].get_weights().reshape(-1,results[name]['som'].get_weights().shape[-1])
                cluster_centers = cluster_centers[:, :2]  # Already in PCA space
            else:
                cluster_centers = results[name]['som'].get_weights().reshape(-1,results[name]['som'].get_weights().shape[-1])
                cluster_centers = datasets[name]['pca'].transform(cluster_centers)[:, :2]

            # Create PCA plot for this experiment
            create_pca_plot(q_num=i,
                            exp_name=name,
                            results=results[name],
                            datasets=datasets,
                            lithology_map=lithology_map,
                            original_colors=original_colors,
                            output_dir=output_dir)

            padded_class_acc = np.zeros(20)
            padded_class_acc[:len(class_acc)] = class_acc

            # Create accuracy table data with lithology statistics and cluster centers
            row_data = {
                'Q_Number': f'Q{i:03d}',
                'Experiment': name,
                'Overall_Accuracy': overall_acc,
            }

            # Get the scaled data from results
            scaled_data = results[name]['scaled_data']

            # Add lithology standard deviations (one column per lithology)
            for lith_id in range(len(lithology_map)):
                mask = results[name]['flat_labels'] == lith_id
                if np.any(mask):
                    lith_data = scaled_data[mask]
                    row_data[f'Lithology_{lith_id + 1}_Std_Dev'] = np.mean(np.std(lith_data, axis=0))
                else:
                    row_data[f'Lithology_{lith_id + 1}_Std_Dev'] = np.nan

            # Calculate and add lithology centers (mean PC1 and PC2)
            if name == 'PCA':
                pca_data = scaled_data[:, :2]  # Already in PCA space
            else:
                pca_data = datasets[name]['pca'].transform(scaled_data)[:, :2]

            for lith_id in range(len(lithology_map)):
                mask = results[name]['flat_labels'] == lith_id
                if np.any(mask):
                    lith_center = np.mean(pca_data[mask], axis=0)
                    row_data[f'Lithology_{lith_id + 1}_Center_PC1'] = lith_center[0]
                    row_data[f'Lithology_{lith_id + 1}_Center_PC2'] = lith_center[1]
                else:
                    row_data[f'Lithology_{lith_id + 1}_Center_PC1'] = np.nan
                    row_data[f'Lithology_{lith_id + 1}_Center_PC2'] = np.nan

            # Add cluster centers (one column per cluster)
            if name == 'PCA':
                cluster_centers = results[name]['som'].get_weights().reshape(-1,
                                                                             results[name]['som'].get_weights().shape[
                                                                                 -1])
                cluster_centers = cluster_centers[:, :2]  # Already in PCA space
            else:
                cluster_centers = results[name]['som'].get_weights().reshape(-1,
                                                                             results[name]['som'].get_weights().shape[
                                                                                 -1])
                cluster_centers = datasets[name]['pca'].transform(cluster_centers)[:, :2]

            for center_idx, center in enumerate(cluster_centers):
                row_data[f'Cluster_{center_idx + 1}_PC1'] = center[0]
                row_data[f'Cluster_{center_idx + 1}_PC2'] = center[1]

            # Add accuracy for each lithology
            for lith_id in range(len(lithology_map)):
                row_data[f'Lithology_{lith_id + 1}_Accuracy'] = class_acc[lith_id] if lith_id < len(
                    class_acc) else np.nan

            accuracy_rows.append(row_data)

        # Create DataFrame from the list of dictionaries
        accuracy_df = pd.DataFrame(accuracy_rows)

        # Save accuracy metrics for this Q number separately
        accuracy_csv_path = os.path.join(output_dir, "SOM_Results", f"Q{i:03d}_accuracy_metrics.csv")
        accuracy_df.to_csv(accuracy_csv_path, index=False)
        print(f"Saved accuracy metrics to: {accuracy_csv_path}")

        create_summary_figure(i, results, pv_data, lithology_map, original_colors, output_dir, datasets)


def create_pca_plot(q_num, exp_name, results, datasets, lithology_map, original_colors, output_dir):
    """Create separate PCA plot with legend."""
    plt.figure(figsize=(10, 8))

    if original_colors.ndim == 2:
        colors = [tuple(c / 255) for c in original_colors]
    else:
        colors = plt.cm.gray(np.linspace(0, 1, len(original_colors)))

    # Get the correct PCA-transformed data
    if exp_name == 'PCA':
        pca_data = results['scaled_data'][:, :2]  # Already in PCA space
    else:
        # Use the dataset-specific PCA transform
        flat_data = results['scaled_data']
        pca_data = datasets[exp_name]['pca'].transform(flat_data)[:, :2]

    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    for lith_id in range(len(lithology_map)):
        mask = results['flat_labels'] == lith_id
        plt.scatter(pca_data[mask, 0], pca_data[mask, 1],
                    color=colors[lith_id], marker=markers[lith_id % len(markers)],
                    alpha=0.3, s=10, label=lithology_map[lith_id])

    # Plot cluster centers with colored markers and black borders
    if exp_name == 'PCA':
        cluster_centers = results['som'].get_weights().reshape(-1, results['som'].get_weights().shape[-1])
        cluster_centers = cluster_centers[:, :2]  # Already in PCA space
    else:
        cluster_centers = results['som'].get_weights().reshape(-1, results['som'].get_weights().shape[-1])
        cluster_centers = datasets[exp_name]['pca'].transform(cluster_centers)[:, :2]

    # Get the dominant lithology for each cluster center
    cluster_dominant_liths = []
    for cluster in range(49):  # 7x7 SOM
        if cluster in results['cluster_summary']:
            cluster_dominant_liths.append(results['cluster_summary'][cluster]['dominant'])
        else:
            cluster_dominant_liths.append(0)  # Default to first lithology if no data

    # Plot each center with its dominant lithology color and black border
    for center_idx, (center, lith_id) in enumerate(zip(cluster_centers, cluster_dominant_liths)):
        plt.scatter(center[0], center[1],
                    c=[colors[lith_id]],  # Use dominant lithology color
                    marker='X',
                    s=100,
                    edgecolor='black',
                    linewidth=1)

    plt.title(f'Q{q_num:03d} {exp_name} - PCA Space', fontsize=12, pad=10)
    plt.xlabel('PC1', fontsize=10)
    plt.ylabel('PC2', fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the PCA plot
    pca_plot_path = os.path.join(output_dir, "SOM_Results", "PCA_Plots", f"Q{q_num:03d}_{exp_name}_PCA.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot to: {pca_plot_path}")


def create_summary_figure(q_num, results, planview, lithology_map, original_colors, output_dir, datasets):
    """Create summary figure with correct PCA transformations."""
    fig = plt.figure(figsize=(24, 12))  # Reduced height since we removed the last row
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)  # Now 3 rows instead of 4

    if original_colors.ndim == 2:
        colors = [tuple(c / 255) for c in original_colors]
    else:
        colors = plt.cm.gray(np.linspace(0, 1, len(original_colors)))

    x_coords = np.arange(0, planview.shape[1], 100)
    y_coords = np.arange(0, planview.shape[0], 100)
    x_labels = [str(x) for x in x_coords]
    y_labels = [str(y) for y in y_coords]

    # Row 1: Plan view + SOM cluster maps
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(planview, origin='lower', aspect='equal')
    ax.set_title('Plan View', fontsize=12, pad=10)
    ax.set_xticks(x_coords)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_coords)
    ax.set_yticklabels(y_labels)

    experiments = ['Raw', 'All Features', 'PCA']
    for col, exp in enumerate(experiments, 1):
        res = results[exp]
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(res['cluster_map'], origin='lower', aspect='equal', cmap='tab20')
        ax.set_title(f'{exp} Clusters (7x7)', fontsize=12, pad=10)
        ax.set_xticks(x_coords)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_coords)
        ax.set_yticklabels(y_labels)

    # Row 2: Stacked bar plots
    for col, exp in enumerate(experiments):
        res = results[exp]
        ax = fig.add_subplot(gs[1, col + 1])

        all_clusters = range(49)
        cluster_data = []
        for cluster in all_clusters:
            if cluster in res['cluster_summary']:
                cluster_data.append(res['cluster_summary'][cluster]['composition'])
            else:
                cluster_data.append({})

        bottom_values = np.zeros(49)
        for lith_id in range(len(lithology_map)):
            values = []
            for cluster_comp in cluster_data:
                values.append(cluster_comp.get(lith_id, 0))
            ax.bar(range(49), values, bottom=bottom_values,
                   color=colors[lith_id], width=1.0, edgecolor='none')
            bottom_values += values

        ax.set_title(f'{exp} - Cluster Composition', fontsize=12, pad=10)
        ax.set_xlabel('Cluster Node (1-49)', fontsize=10)
        ax.set_ylabel('Composition %', fontsize=10)
        ax.set_xlim(0, 48)
        ax.set_ylim(0, 1)

    # Row 3: Dominant lithology maps
    for col, exp in enumerate(experiments):
        res = results[exp]
        ax = fig.add_subplot(gs[2, col + 1])

        dominant_lith_map = np.zeros_like(res['cluster_map'])
        for cluster, summary in res['cluster_summary'].items():
            mask = res['cluster_map'] == cluster
            dominant_lith_map[mask] = summary['dominant']

        custom_cmap = ListedColormap(colors[:len(lithology_map)])
        im = ax.imshow(dominant_lith_map, origin='lower', aspect='equal',
                       cmap=custom_cmap, vmin=0, vmax=len(lithology_map) - 1)
        ax.set_title(f'{exp} - Dominant Lithology', fontsize=12, pad=10)
        ax.set_xticks(x_coords)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_coords)
        ax.set_yticklabels(y_labels)

    plt.suptitle(f'Q{q_num:03d} SOM Analysis Results (7x7)', y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "SOM_Results", f"Q{q_num:03d}_SOM_Analysis.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


def run_som(data, labels, som_shape=(7, 7), n_iter=400):
    """Run SOM clustering and analyze lithology composition."""
    original_shape = data.shape[:2]
    flat_data = data.reshape(-1, data.shape[-1])
    flat_labels = labels.reshape(-1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(flat_data)

    som = MiniSom(som_shape[0], som_shape[1], scaled_data.shape[1],
                  sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(scaled_data)
    som.train_random(scaled_data, n_iter, verbose=False)

    winner_coordinates = np.array([som.winner(x) for x in scaled_data])
    cluster_assignments = winner_coordinates[:, 0] * som_shape[1] + winner_coordinates[:, 1]
    cluster_map = cluster_assignments.reshape(original_shape)

    cluster_composition = defaultdict(lambda: defaultdict(int))
    for coord, lith in zip(winner_coordinates, flat_labels):
        cluster = coord[0] * som_shape[1] + coord[1]
        cluster_composition[cluster][lith] += 1

    cluster_summary = {}
    for cluster, counts in cluster_composition.items():
        total = sum(counts.values())
        percentages = {k: v / total for k, v in counts.items()}
        dominant = max(percentages.items(), key=lambda x: x[1])
        cluster_summary[cluster] = {
            'composition': percentages,
            'dominant': dominant[0],
            'dominant_percentage': dominant[1],
            'counts': counts
        }

    return {
        'cluster_map': cluster_map,
        'cluster_summary': cluster_summary,
        'unique_clusters': np.unique(cluster_assignments),
        'som_shape': som_shape,
        'som': som,
        'scaled_data': scaled_data,  # Make sure this is included
        'flat_labels': flat_labels,
        'original_data': flat_data  # Keep original data for reference
    }


def main():
    output_dir = "SyntheticNoddy"
    os.makedirs(output_dir, exist_ok=True)

    # Load previously processed data
    geophys_data = {}
    planview_data = {}
    for i in range(1, 7):
        grav_file = os.path.join(output_dir, "Data", f"Q{i:03d}-Grav_processed.npz")
        mag_file = os.path.join(output_dir, "Data", f"Q{i:03d}-Mag_processed.npz")
        pv_file = os.path.join(output_dir, "Data", f"Q{i:03d}-planview_processed.npz")

        if os.path.exists(grav_file):
            geophys_data[f"Q{i:03d}-Grav.bmp"] = np.load(grav_file)
        if os.path.exists(mag_file):
            geophys_data[f"Q{i:03d}-Mag.bmp"] = np.load(mag_file)
        if os.path.exists(pv_file):
            planview_data[f"Q{i:03d}-PlanView.bmp"] = np.load(pv_file)

    run_som_experiments(geophys_data, planview_data, output_dir)


if __name__ == "__main__":
    main()