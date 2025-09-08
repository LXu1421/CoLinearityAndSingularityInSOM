import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

# This part of the code completes the SOM experiments and the PCA plot

def calculate_clustering_metrics(data, labels):
    """Calculate various clustering validation metrics."""
    if len(np.unique(labels)) < 2:
        return {'silhouette': np.nan, 'davies_bouldin': np.nan}

    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)

    return {'silhouette': silhouette, 'davies_bouldin': davies_bouldin}


def hierarchical_cluster_refinement(som_weights, n_clusters_range=None, method='ward'):
    """Apply hierarchical clustering to SOM codebook vectors."""
    print(f"    🌳 Running hierarchical clustering...")

    # Flatten SOM weights to get codebook vectors
    codebook = som_weights.reshape(-1, som_weights.shape[-1])
    print(f"       Codebook shape: {codebook.shape}")

    if n_clusters_range is None:
        n_clusters_range = range(2, min(len(codebook), 10))

    results = {}

    # Perform hierarchical clustering for different numbers of clusters
    print(f"       Computing linkage matrix...")
    linkage_matrix = linkage(codebook, method=method)

    print(f"       Testing {len(n_clusters_range)} cluster configurations...")
    for i, n_clusters in enumerate(n_clusters_range, 1):
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        metrics = calculate_clustering_metrics(codebook, cluster_labels)

        results[n_clusters] = {
            'labels': cluster_labels,
            'linkage_matrix': linkage_matrix,
            'metrics': metrics
        }

        if i % max(1, len(n_clusters_range) // 5) == 0:  # Progress every 20%
            print(f"       Progress: {i}/{len(n_clusters_range)} ({i / len(n_clusters_range) * 100:.0f}%)")

    return results


def kmeans_cluster_refinement(som_weights, n_clusters_range=None, random_state=42):
    """Apply K-means clustering to SOM codebook vectors."""
    print(f"    🎯 Running K-means clustering...")

    codebook = som_weights.reshape(-1, som_weights.shape[-1])
    print(f"       Codebook shape: {codebook.shape}")

    if n_clusters_range is None:
        n_clusters_range = range(2, min(len(codebook), 10))

    results = {}

    print(f"       Testing {len(n_clusters_range)} cluster configurations...")
    for i, n_clusters in enumerate(n_clusters_range, 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(codebook)
        metrics = calculate_clustering_metrics(codebook, cluster_labels)

        results[n_clusters] = {
            'labels': cluster_labels,
            'kmeans_model': kmeans,
            'metrics': metrics,
            'inertia': kmeans.inertia_
        }

        if i % max(1, len(n_clusters_range) // 5) == 0:  # Progress every 20%
            print(f"       Progress: {i}/{len(n_clusters_range)} ({i / len(n_clusters_range) * 100:.0f}%)")

    return results


def find_optimal_clusters(hierarchical_results, kmeans_results):
    """Find optimal number of clusters based on validation metrics."""
    optimal_results = {}

    # Find best hierarchical clustering
    best_hier_score = -np.inf
    best_hier_n = None
    for n, result in hierarchical_results.items():
        if not np.isnan(result['metrics']['silhouette']):
            if result['metrics']['silhouette'] > best_hier_score:
                best_hier_score = result['metrics']['silhouette']
                best_hier_n = n

    # Find best K-means clustering
    best_kmeans_score = -np.inf
    best_kmeans_n = None
    for n, result in kmeans_results.items():
        if not np.isnan(result['metrics']['silhouette']):
            if result['metrics']['silhouette'] > best_kmeans_score:
                best_kmeans_score = result['metrics']['silhouette']
                best_kmeans_n = n

    optimal_results = {
        'hierarchical': {'n_clusters': best_hier_n, 'score': best_hier_score},
        'kmeans': {'n_clusters': best_kmeans_n, 'score': best_kmeans_score}
    }

    return optimal_results


def apply_cluster_refinement_to_som(som_result, refinement_labels, som_shape):
    """Apply cluster refinement labels to SOM results."""
    # Create mapping from SOM nodes to refined clusters
    node_to_refined_cluster = {}
    for node_idx, refined_label in enumerate(refinement_labels):
        node_to_refined_cluster[node_idx] = refined_label

    # Apply mapping to the cluster map
    refined_cluster_map = np.zeros_like(som_result['cluster_map'])
    for i in range(som_result['cluster_map'].shape[0]):
        for j in range(som_result['cluster_map'].shape[1]):
            som_node = som_result['cluster_map'][i, j]
            refined_cluster_map[i, j] = node_to_refined_cluster[som_node]

    # Update cluster summary
    refined_cluster_summary = {}
    unique_refined_clusters = np.unique(refined_cluster_map)

    for refined_cluster in unique_refined_clusters:
        mask = refined_cluster_map == refined_cluster
        masked_labels = som_result['flat_labels'][mask.ravel()]

        if len(masked_labels) > 0:
            unique_labels, counts = np.unique(masked_labels, return_counts=True)
            dominant_idx = np.argmax(counts)
            dominant_label = unique_labels[dominant_idx]
            purity = counts[dominant_idx] / len(masked_labels)

            refined_cluster_summary[refined_cluster] = {
                'dominant': dominant_label,
                'purity': purity,
                'size': len(masked_labels),
                'distribution': dict(zip(unique_labels, counts))
            }

    return {
        'cluster_map': refined_cluster_map,
        'cluster_summary': refined_cluster_summary,
        'flat_labels': som_result['flat_labels'],
        'scaled_data': som_result['scaled_data']
    }


def supervised_refinement(som_result, true_labels, som_weights):
    """Use supervised learning to refine SOM clusters based on geological knowledge."""
    # Prepare training data
    codebook = som_weights.reshape(-1, som_weights.shape[-1])

    # Get the dominant geological label for each SOM node
    node_labels = []
    som_shape_flat = som_weights.shape[0] * som_weights.shape[1]

    for node_idx in range(som_shape_flat):
        mask = som_result['cluster_map'].ravel() == node_idx
        if np.any(mask):
            node_true_labels = true_labels.ravel()[mask]
            if len(node_true_labels) > 0:
                unique_labels, counts = np.unique(node_true_labels, return_counts=True)
                dominant_label = unique_labels[np.argmax(counts)]
                node_labels.append(dominant_label)
            else:
                node_labels.append(-1)  # No data for this node
        else:
            node_labels.append(-1)  # No data for this node

    node_labels = np.array(node_labels)

    # Train classifier on nodes with valid labels
    valid_mask = node_labels != -1
    if np.sum(valid_mask) < 2:
        return None

    X_train = codebook[valid_mask]
    y_train = node_labels[valid_mask]

    # Train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predict labels for all nodes
    predicted_labels = rf.predict(codebook)

    # Calculate cross-validation score
    cv_score = np.mean(cross_val_score(rf, X_train, y_train, cv=min(5, len(np.unique(y_train)))))

    return {
        'model': rf,
        'predicted_labels': predicted_labels,
        'cv_score': cv_score,
        'feature_importance': rf.feature_importances_
    }


def create_clustering_comparison_plot(q_num, results_dict, output_dir):
    """Create comparison plots for different clustering approaches."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Q{q_num:03d} - Clustering Refinement Comparison', fontsize=16)

    methods = ['original', 'reduced_grid', 'hierarchical', 'kmeans', 'supervised']

    for idx, (method, result) in enumerate(results_dict.items()):
        if idx >= 5:  # Limit to 5 methods for display
            break

        row = idx // 3
        col = idx % 3

        if idx < len(axes.flat):
            ax = axes.flat[idx]

            # Plot cluster map
            im = ax.imshow(result['cluster_map'], cmap='tab20')
            ax.set_title(f'{method.replace("_", " ").title()}')
            ax.axis('off')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(len(results_dict), len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "SOM_Results", f"Q{q_num:03d}_clustering_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_comparison_plot(metrics_df, output_dir):
    """Create comparison plots for clustering validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Silhouette scores by method
    axes[0, 0].boxplot([metrics_df[metrics_df['Method'] == method]['Silhouette_Score'].dropna()
                        for method in metrics_df['Method'].unique()],
                       labels=metrics_df['Method'].unique())
    axes[0, 0].set_title('Silhouette Scores by Method')
    axes[0, 0].set_ylabel('Silhouette Score')
    plt.setp(axes[0, 0].get_xticklabels(), rotation=45)

    # Davies-Bouldin scores by method
    axes[0, 1].boxplot([metrics_df[metrics_df['Method'] == method]['Davies_Bouldin_Score'].dropna()
                        for method in metrics_df['Method'].unique()],
                       labels=metrics_df['Method'].unique())
    axes[0, 1].set_title('Davies-Bouldin Scores by Method')
    axes[0, 1].set_ylabel('Davies-Bouldin Score')
    plt.setp(axes[0, 1].get_xticklabels(), rotation=45)

    # Overall accuracy by method
    axes[1, 0].boxplot([metrics_df[metrics_df['Method'] == method]['Overall_Accuracy'].dropna()
                        for method in metrics_df['Method'].unique()],
                       labels=metrics_df['Method'].unique())
    axes[1, 0].set_title('Overall Accuracy by Method')
    axes[1, 0].set_ylabel('Accuracy')
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45)

    # Number of clusters by method
    axes[1, 1].boxplot([metrics_df[metrics_df['Method'] == method]['N_Clusters'].dropna()
                        for method in metrics_df['Method'].unique()],
                       labels=metrics_df['Method'].unique())
    axes[1, 1].set_title('Number of Clusters by Method')
    axes[1, 1].set_ylabel('Number of Clusters')
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "SOM_Results", "clustering_methods_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_som_experiments_enhanced(geophys_data, planview_data, output_dir):
    """Run enhanced SOM experiments with multiple clustering refinement techniques."""
    print("🚀 Starting Enhanced SOM Analysis with Clustering Refinement")
    print("=" * 60)

    os.makedirs(os.path.join(output_dir, "SOM_Results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "SOM_Results", "PCA_Plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "SOM_Results", "Clustering_Refinement"), exist_ok=True)

    all_metrics = []
    total_datasets = sum(1 for i in range(1, 7)
                         if all(key in geophys_data for key in [f"Q{i:03d}-Grav.bmp", f"Q{i:03d}-Mag.bmp"])
                         and f"Q{i:03d}-PlanView.bmp" in planview_data)

    print(f"📊 Found {total_datasets} valid Q datasets to process")
    print(f"🔬 Each dataset will test 3 feature combinations: Raw, All Features, PCA")
    print(f"⚡ Total processing steps: ~{total_datasets * 3 * 6} major operations")
    print("=" * 60)

    processed_count = 0

    for i in range(1, 7):
        grav_key = f"Q{i:03d}-Grav.bmp"
        mag_key = f"Q{i:03d}-Mag.bmp"
        pv_key = f"Q{i:03d}-PlanView.bmp"

        if grav_key not in geophys_data or mag_key not in geophys_data or pv_key not in planview_data:
            print(f"⚠️  Skipping Q{i:03d} - missing required data files")
            continue

        print(f"\n🎯 STARTING Q{i:03d} ANALYSIS")
        print(f"📁 Processing: {grav_key}, {mag_key}, {pv_key}")

        grav_raw = geophys_data[grav_key]['raw']
        mag_raw = geophys_data[mag_key]['raw']
        pv_data = planview_data[pv_key]['expanded']

        print(f"📐 Data shapes - Grav: {grav_raw.shape}, Mag: {mag_raw.shape}, PlanView: {pv_data.shape}")

        # Process plan view data (same as original)
        print("🎨 Processing plan view data for lithology labels...")
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
        n_expected_classes = len(unique_colors)
        print(f"🧭 Found {n_expected_classes} unique lithology classes: {list(lithology_map.values())}")

        # Prepare datasets (same as original)
        print("⚙️  Preparing feature datasets...")
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
        print("🔄 Fitting PCA transformations...")
        for name in datasets:
            flat_data = datasets[name]['data'].reshape(-1, datasets[name]['data'].shape[-1])
            datasets[name]['pca'].fit(flat_data)
            print(f"  ✓ PCA fitted for {name} dataset ({datasets[name]['data'].shape[-1]} features)")

        # Add PCA dataset
        datasets['PCA'] = {
            'data': datasets['All Features']['pca'].transform(
                datasets['All Features']['data'].reshape(-1, datasets['All Features']['data'].shape[-1])
            ).reshape(datasets['All Features']['data'].shape[0], datasets['All Features']['data'].shape[1], 2),
            'pca': None
        }
        print(f"  ✓ PCA dataset created with 2 components")
        print(f"📋 Ready to process {len(datasets)} datasets: {list(datasets.keys())}")

        # Run experiments for each dataset
        for dataset_name, data_dict in datasets.items():
            print(f"\n=== Processing Q{i:03d} - {dataset_name} ===")

            clustering_results = {}

            # 1. Original SOM (7x7 grid)
            print(f"  [1/6] Running original SOM (7x7 grid)...")
            original_som = run_som(data_dict['data'], pv_labels, som_shape=(7, 7))
            clustering_results['original'] = original_som
            print(f"  ✓ Original SOM complete - {len(original_som['cluster_summary'])} clusters found")

            # 2. Reduced grid size SOM (4x4, 5x5, 6x6)
            print(f"  [2/6] Running reduced grid size SOMs...")
            for grid_size in [4, 5, 6]:
                print(f"    - Testing {grid_size}x{grid_size} grid...")
                reduced_som = run_som(data_dict['data'], pv_labels, som_shape=(grid_size, grid_size))
                clustering_results[f'reduced_grid_{grid_size}x{grid_size}'] = reduced_som
                print(f"    ✓ {grid_size}x{grid_size} complete - {len(reduced_som['cluster_summary'])} clusters")

            # 3. Hierarchical clustering refinement
            print(f"  [3/6] Running hierarchical clustering refinement...")
            som_weights = original_som['som'].get_weights()
            hierarchical_results = hierarchical_cluster_refinement(
                som_weights,
                n_clusters_range=range(2, min(n_expected_classes + 3, 15))
            )
            print(f"  ✓ Hierarchical clustering complete - tested {len(hierarchical_results)} cluster numbers")

            # 4. K-means clustering refinement
            print(f"  [4/6] Running K-means clustering refinement...")
            kmeans_results = kmeans_cluster_refinement(
                som_weights,
                n_clusters_range=range(2, min(n_expected_classes + 3, 15))
            )
            print(f"  ✓ K-means clustering complete - tested {len(kmeans_results)} cluster numbers")

            # Find optimal clustering
            print(f"  [5/6] Finding optimal clustering parameters...")
            optimal_clusters = find_optimal_clusters(hierarchical_results, kmeans_results)

            # Apply best hierarchical clustering
            if optimal_clusters['hierarchical']['n_clusters'] is not None:
                print(
                    f"    - Applying optimal hierarchical clustering ({optimal_clusters['hierarchical']['n_clusters']} clusters)...")
                best_hier_labels = hierarchical_results[optimal_clusters['hierarchical']['n_clusters']]['labels']
                hier_refined = apply_cluster_refinement_to_som(original_som, best_hier_labels, (7, 7))
                clustering_results['hierarchical_optimal'] = hier_refined
                print(f"    ✓ Hierarchical refinement applied")
            else:
                print(f"    ⚠ No valid hierarchical clustering found")

            # Apply best K-means clustering
            if optimal_clusters['kmeans']['n_clusters'] is not None:
                print(
                    f"    - Applying optimal K-means clustering ({optimal_clusters['kmeans']['n_clusters']} clusters)...")
                best_kmeans_labels = kmeans_results[optimal_clusters['kmeans']['n_clusters']]['labels']
                kmeans_refined = apply_cluster_refinement_to_som(original_som, best_kmeans_labels, (7, 7))
                clustering_results['kmeans_optimal'] = kmeans_refined
                print(f"    ✓ K-means refinement applied")
            else:
                print(f"    ⚠ No valid K-means clustering found")

            # 5. Supervised refinement
            print(f"  [6/6] Running supervised refinement...")
            supervised_result = supervised_refinement(original_som, pv_labels, som_weights)
            if supervised_result is not None:
                supervised_refined = apply_cluster_refinement_to_som(
                    original_som,
                    supervised_result['predicted_labels'],
                    (7, 7)
                )
                clustering_results['supervised'] = supervised_refined
                print(f"  ✓ Supervised refinement complete - CV score: {supervised_result['cv_score']:.3f}")
            else:
                print(f"  ⚠ Supervised refinement failed - insufficient training data")

            # Calculate metrics for all clustering approaches
            print(f"  → Calculating metrics for {len(clustering_results)} clustering methods...")
            for idx, (method_name, clustering_result) in enumerate(clustering_results.items(), 1):
                print(f"    [{idx}/{len(clustering_results)}] Analyzing {method_name}...")

                # Calculate dominant lithology map
                dominant_lith_map = np.zeros_like(clustering_result['cluster_map'])
                for cluster, summary in clustering_result['cluster_summary'].items():
                    mask = clustering_result['cluster_map'] == cluster
                    dominant_lith_map[mask] = summary['dominant']

                # Calculate accuracy
                flat_true = pv_labels.ravel()
                flat_pred = dominant_lith_map.ravel()
                overall_acc = accuracy_score(flat_true, flat_pred)

                # Calculate clustering metrics
                if dataset_name == 'PCA':
                    cluster_data = clustering_result['scaled_data'][:, :2]
                else:
                    cluster_data = datasets[dataset_name]['pca'].transform(
                        clustering_result['scaled_data']
                    )[:, :2]

                cluster_labels = clustering_result['cluster_map'].ravel()
                clustering_metrics = calculate_clustering_metrics(cluster_data, cluster_labels)

                # Store metrics
                metrics_row = {
                    'Q_Number': f'Q{i:03d}',
                    'Dataset': dataset_name,
                    'Method': method_name,
                    'Overall_Accuracy': overall_acc,
                    'N_Clusters': len(np.unique(cluster_labels)),
                    'Expected_Classes': n_expected_classes,
                    'Silhouette_Score': clustering_metrics['silhouette'],
                    'Davies_Bouldin_Score': clustering_metrics['davies_bouldin']
                }

                # Add optimal cluster information
                if method_name == 'hierarchical_optimal':
                    metrics_row['Optimal_Silhouette'] = optimal_clusters['hierarchical']['score']
                elif method_name == 'kmeans_optimal':
                    metrics_row['Optimal_Silhouette'] = optimal_clusters['kmeans']['score']
                elif method_name == 'supervised' and supervised_result is not None:
                    metrics_row['CV_Score'] = supervised_result['cv_score']

                all_metrics.append(metrics_row)
                print(f"    ✓ {method_name}: Accuracy={overall_acc:.3f}, Clusters={len(np.unique(cluster_labels))}")

            # Create comparison plot for this Q number and dataset
            print(f"  → Creating comparison plots...")
            create_clustering_comparison_plot(i, clustering_results, output_dir)
            print(f"  ✓ Comparison plot saved")

            # Save detailed results for this Q number and dataset
            print(f"  → Saving detailed results...")
            results_path = os.path.join(
                output_dir, "SOM_Results", "Clustering_Refinement",
                f"Q{i:03d}_{dataset_name}_clustering_results.pkl"
            )
            pd.to_pickle({
                'clustering_results': clustering_results,
                'hierarchical_analysis': hierarchical_results,
                'kmeans_analysis': kmeans_results,
                'optimal_clusters': optimal_clusters,
                'supervised_analysis': supervised_result
            }, results_path)
            print(f"  ✓ Results saved to {results_path}")
            print(f"=== Q{i:03d} - {dataset_name} COMPLETE ===\n")
            processed_count += 1

    # Create overall metrics DataFrame and save
    print(f"\n🏁 ANALYSIS COMPLETE!")
    print(f"📊 Processed {processed_count} dataset combinations")
    print(f"💾 Compiling final results...")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(output_dir, "SOM_Results", "clustering_refinement_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Metrics summary saved to: {metrics_path}")

    # Create comparison plots
    print(f"📈 Creating final comparison plots...")
    create_metrics_comparison_plot(metrics_df, output_dir)
    print(f"✅ Comparison plots saved")

    print(f"\n🎉 Enhanced SOM analysis complete!")
    print(f"📁 All results saved to: {output_dir}/SOM_Results/")
    print(f"📈 Metrics summary: {metrics_path}")
    print("=" * 60)

    return metrics_df


# Note: You'll need to ensure that the run_som function and create_pca_plot function
# are available from your original code, as they're called but not defined here.

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
    """Create enhanced PCA plots with variance explained, 3D visualization, and additional biplots."""
    # Create output directory for enhanced PCA results
    pca_output_dir = os.path.join(output_dir, "SOM_Results", "Enhanced_PCA")
    os.makedirs(pca_output_dir, exist_ok=True)

    if original_colors.ndim == 2:
        colors = [tuple(c / 255) for c in original_colors]
    else:
        colors = plt.cm.gray(np.linspace(0, 1, len(original_colors)))

    # Get the correct PCA-transformed data and variance information
    if exp_name == 'PCA':
        pca_data = results['scaled_data']  # Already in PCA space
        explained_var = None  # Not available for pre-transformed data
    else:
        # Use the dataset-specific PCA transform
        flat_data = results['scaled_data']
        pca = datasets[exp_name]['pca']
        pca_data = pca.transform(flat_data)
        explained_var = pca.explained_variance_ratio_

        # Save variance explained table
        var_table = pd.DataFrame({
            'PC': [f'PC{i + 1}' for i in range(len(explained_var))],
            'Variance Explained': explained_var,
            'Cumulative Variance': np.cumsum(explained_var)
        })
        var_table_path = os.path.join(pca_output_dir, f"Q{q_num:03d}_{exp_name}_variance.csv")
        var_table.to_csv(var_table_path, index=False)
        print(f"Saved variance explained table to: {var_table_path}")

    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']

    # Create standard 2D PC1-PC2 plot (existing functionality)
    plt.figure(figsize=(10, 8))
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

    # Add variance explained to title if available
    if explained_var is not None:
        title = (f'Q{q_num:03d} {exp_name} - PCA Space\n'
                 f'PC1: {explained_var[0] * 100:.1f}%, PC2: {explained_var[1] * 100:.1f}%')
    else:
        title = f'Q{q_num:03d} {exp_name} - PCA Space'

    plt.title(title, fontsize=12, pad=10)
    plt.xlabel('PC1', fontsize=10)
    plt.ylabel('PC2', fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the standard PCA plot
    pca_plot_path = os.path.join(pca_output_dir, f"Q{q_num:03d}_{exp_name}_PCA_PC1-PC2.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved standard PCA plot to: {pca_plot_path}")

    # Only proceed with enhanced analysis if we have variance information (not for 'PCA' experiment)
    if explained_var is None:
        return

    # Create 3D PCA plot if we have at least 3 components with some variance
    if len(explained_var) >= 3 and explained_var[2] > 0.01:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for lith_id in range(len(lithology_map)):
            mask = results['flat_labels'] == lith_id
            ax.scatter(pca_data[mask, 0], pca_data[mask, 1], pca_data[mask, 2],
                       color=colors[lith_id], marker=markers[lith_id % len(markers)],
                       alpha=0.3, s=10, label=lithology_map[lith_id])

        # Plot cluster centers in 3D
        cluster_centers_3d = datasets[exp_name]['pca'].transform(
            results['som'].get_weights().reshape(-1, results['som'].get_weights().shape[-1]))[:, :3]

        for center_idx, (center, lith_id) in enumerate(zip(cluster_centers_3d, cluster_dominant_liths)):
            ax.scatter(center[0], center[1], center[2],
                       c=[colors[lith_id]],
                       marker='X',
                       s=100,
                       edgecolor='black',
                       linewidth=1)

        title_3d = (f'Q{q_num:03d} {exp_name} - 3D PCA Space\n'
                    f'PC1: {explained_var[0] * 100:.1f}%, PC2: {explained_var[1] * 100:.1f}%, PC3: {explained_var[2] * 100:.1f}%')
        ax.set_title(title_3d, fontsize=12, pad=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        pca_3d_path = os.path.join(pca_output_dir, f"Q{q_num:03d}_{exp_name}_PCA_3D.png")
        plt.savefig(pca_3d_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved 3D PCA plot to: {pca_3d_path}")

    # Create additional biplots for significant components (variance > 5%)
    significant_pcs = [i for i, var in enumerate(explained_var) if var > 0.05]

    # Generate all possible pairs of significant PCs beyond PC1-PC2
    from itertools import combinations
    pc_pairs = [pair for pair in combinations(significant_pcs, 2) if pair != (0, 1)]

    for pc1, pc2 in pc_pairs:
        plt.figure(figsize=(10, 8))

        for lith_id in range(len(lithology_map)):
            mask = results['flat_labels'] == lith_id
            plt.scatter(pca_data[mask, pc1], pca_data[mask, pc2],
                        color=colors[lith_id], marker=markers[lith_id % len(markers)],
                        alpha=0.3, s=10, label=lithology_map[lith_id])

        # Plot cluster centers for this PC pair
        cluster_centers_pc = datasets[exp_name]['pca'].transform(
            results['som'].get_weights().reshape(-1, results['som'].get_weights().shape[-1]))[:, [pc1, pc2]]

        for center_idx, (center, lith_id) in enumerate(zip(cluster_centers_pc, cluster_dominant_liths)):
            plt.scatter(center[0], center[1],
                        c=[colors[lith_id]],
                        marker='X',
                        s=100,
                        edgecolor='black',
                        linewidth=1)

        title_pc = (f'Q{q_num:03d} {exp_name} - PCA Space\n'
                    f'PC{pc1 + 1}: {explained_var[pc1] * 100:.1f}%, PC{pc2 + 1}: {explained_var[pc2] * 100:.1f}%')
        plt.title(title_pc, fontsize=12, pad=10)
        plt.xlabel(f'PC{pc1 + 1}')
        plt.ylabel(f'PC{pc2 + 1}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        biplot_path = os.path.join(pca_output_dir, f"Q{q_num:03d}_{exp_name}_PCA_PC{pc1 + 1}-PC{pc2 + 1}.png")
        plt.savefig(biplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved biplot to: {biplot_path}")

    # Calculate and save quantitative cluster separation metrics
    if len(explained_var) >= 3:
        # Get cluster centers in 3D PCA space
        cluster_centers_3d = datasets[exp_name]['pca'].transform(
            results['som'].get_weights().reshape(-1, results['som'].get_weights().shape[-1]))[:, :3]

        # Calculate pairwise distances between cluster centers
        from scipy.spatial import distance_matrix
        center_distances = distance_matrix(cluster_centers_3d, cluster_centers_3d)

        # Calculate mean and std of distances (excluding diagonal)
        np.fill_diagonal(center_distances, np.nan)
        mean_dist = np.nanmean(center_distances)
        std_dist = np.nanstd(center_distances)

        # Calculate lithology separation metrics
        lithology_separation = []
        for lith_id in range(len(lithology_map)):
            mask = results['flat_labels'] == lith_id
            if np.sum(mask) > 0:
                lith_data = pca_data[mask][:, :3]
                mean = np.mean(lith_data, axis=0)
                std = np.std(lith_data, axis=0)
                lithology_separation.append({
                    'Lithology': lithology_map[lith_id],
                    'Mean_PC1': mean[0],
                    'Mean_PC2': mean[1],
                    'Mean_PC3': mean[2],
                    'Std_PC1': std[0],
                    'Std_PC2': std[1],
                    'Std_PC3': std[2],
                    'Cluster_Count': np.sum(mask)
                })

        # Save separation metrics
        separation_df = pd.DataFrame({
            'Metric': ['Cluster Center Mean Distance', 'Cluster Center Distance Std'],
            'Value': [mean_dist, std_dist]
        })

        lithology_df = pd.DataFrame(lithology_separation)

        # Save to CSV
        separation_path = os.path.join(pca_output_dir, f"Q{q_num:03d}_{exp_name}_cluster_separation.csv")
        separation_df.to_csv(separation_path, index=False)

        lithology_path = os.path.join(pca_output_dir, f"Q{q_num:03d}_{exp_name}_lithology_separation.csv")
        lithology_df.to_csv(lithology_path, index=False)

        print(f"Saved cluster separation metrics to: {separation_path}")
        print(f"Saved lithology separation metrics to: {lithology_path}")

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