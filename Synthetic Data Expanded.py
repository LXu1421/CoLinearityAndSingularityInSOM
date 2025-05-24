import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.ticker as ticker



def generate_spherical_clusters(n_samples=1000, cluster_distance=2.0, cluster_std=0.4):
    """Generate two spherical clusters with variable distance and standard deviation."""
    cluster_center1 = np.array([-cluster_distance / 2, -cluster_distance / 2])
    cluster_center2 = np.array([cluster_distance / 2, cluster_distance / 2])

    # Generate clusters
    cluster1 = np.random.randn(n_samples // 2, 2) * cluster_std + cluster_center1
    cluster2 = np.random.randn(n_samples // 2, 2) * cluster_std + cluster_center2

    # Labels (0 for cluster1, 1 for cluster2)
    labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    # Original features (PC1 and PC2)
    X = np.vstack([cluster1, cluster2])

    # Add collinear feature (PC1b is a noisy version of PC1)
    PC1b = X[:, 0] + np.random.normal(0, 0.02, n_samples)

    # Add noise features
    noise_features = np.random.randn(n_samples, 3)  # 3 noise features

    # Combine all features
    X_extended = np.column_stack([X, PC1b, noise_features])

    return X, X_extended, labels


def train_som(X, som_shape=(10, 10)):
    """Train a Self-Organizing Map on the input data."""
    som = MiniSom(som_shape[0], som_shape[1], X.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, 100)
    return som


def evaluate_som_performance(som, X, labels):
    """Evaluate SOM performance using precision, recall, F1, and accuracy."""
    # Get the winning neurons for all samples
    winners = np.array([som.winner(x) for x in X])

    # Create a label mapping for each neuron
    neuron_labels = {}
    neuron_counts = {}

    # First pass: count label occurrences in each neuron
    for (x, y), label in zip(winners, labels):
        if (x, y) not in neuron_counts:
            neuron_counts[(x, y)] = [0, 0]  # [count_class0, count_class1]
        neuron_counts[(x, y)][int(label)] += 1

    # Second pass: assign majority label to each neuron
    for neuron in neuron_counts:
        counts = neuron_counts[neuron]
        neuron_labels[neuron] = 0 if counts[0] > counts[1] else 1

    # Predict based on neuron labels (handle unseen neurons with default class)
    pred_labels = np.array([neuron_labels.get(som.winner(x), 0) for x in X])

    # Calculate metrics - handle division by zero cases
    try:
        precision = precision_score(labels, pred_labels, zero_division=0)
        recall = recall_score(labels, pred_labels, zero_division=0)
        f1 = f1_score(labels, pred_labels, zero_division=0)
        accuracy = accuracy_score(labels, pred_labels)
    except:
        # Fallback in case of unexpected errors
        precision, recall, f1, accuracy = 0, 0, 0, 0

    return precision, recall, f1, accuracy


def run_experiment_grid(cluster_distances=np.arange(0.5, 4.5, 0.5),
                        cluster_stds=np.arange(0.2, 0.7, 0.1)):
    """Run experiments across a grid of cluster distances and standard deviations."""
    results = []

    for dist in cluster_distances:
        for std in cluster_stds:
            # Generate data
            X, X_extended, labels = generate_spherical_clusters(cluster_distance=dist, cluster_std=std)

            # Train SOMs
            som_simple = train_som(X)
            som_extended = train_som(X_extended)

            # Evaluate performance
            metrics_simple = evaluate_som_performance(som_simple, X, labels)
            metrics_extended = evaluate_som_performance(som_extended, X_extended, labels)

            # Calculate performance ratios (extended/simple)
            ratios = np.array(metrics_extended) / np.array(metrics_simple)

            results.append({
                'cluster_distance': dist,
                'cluster_std': std,
                'simple_precision': metrics_simple[0],
                'simple_recall': metrics_simple[1],
                'simple_f1': metrics_simple[2],
                'simple_accuracy': metrics_simple[3],
                'extended_precision': metrics_extended[0],
                'extended_recall': metrics_extended[1],
                'extended_f1': metrics_extended[2],
                'extended_accuracy': metrics_extended[3],
                'precision_ratio': ratios[0],
                'recall_ratio': ratios[1],
                'f1_ratio': ratios[2],
                'accuracy_ratio': ratios[3]
            })

    return pd.DataFrame(results)


def plot_results_grid(results_df):
    """Visualize the experiment results with heatmaps."""
    # Set up the figure
    plt.style.use('seaborn-v0_8')

    # Create a figure with subplots for ratios
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Define metrics and titles
    ratio_metrics = ['precision_ratio', 'recall_ratio', 'f1_ratio', 'accuracy_ratio']
    ratio_titles = ['Precision Ratio (Extended/Simple)',
                    'Recall Ratio (Extended/Simple)',
                    'F1-Score Ratio (Extended/Simple)',
                    'Accuracy Ratio (Extended/Simple)']

    # Plot ratio heatmaps
    for ax, metric, title in zip(axes.flatten(), ratio_metrics, ratio_titles):
        pivot_table = results_df.pivot(index='cluster_std', columns='cluster_distance', values=metric)

        # Sort y-axis in ascending order
        pivot_table = pivot_table.sort_index(ascending=False)

        # Get data range for colormap
        vmin = round(pivot_table.min().min(), 2)
        vmax = round(pivot_table.max().max(), 2)

        sns.heatmap(pivot_table,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    cbar=True,
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    center=1.0,
                    annot_kws={"size": 12})

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Cluster Distance', fontsize=12)
        ax.set_ylabel('Cluster Std', fontsize=12)

        # Format y-axis ticks to 1 decimal place
        yticks = [f"{float(y.get_text()):.1f}" for y in ax.get_yticklabels()]
        ax.set_yticklabels(yticks)

    plt.tight_layout()
    plt.savefig('performance_ratios_grid.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a figure with subplots for absolute metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    abs_metrics = ['extended_f1', 'simple_f1', 'extended_accuracy', 'simple_accuracy']
    abs_titles = ['Extended F1-Score', 'Simple F1-Score',
                  'Extended Accuracy', 'Simple Accuracy']

    # Plot absolute metric heatmaps
    for ax, metric, title in zip(axes.flatten(), abs_metrics, abs_titles):
        pivot_table = results_df.pivot(index='cluster_std', columns='cluster_distance', values=metric)

        # Sort y-axis in ascending order
        pivot_table = pivot_table.sort_index(ascending=False)

        # Get data range for colormap
        vmin = round(pivot_table.min().min(), 2)
        vmax = round(pivot_table.max().max(), 2)

        sns.heatmap(pivot_table,
                    annot=True,
                    fmt=".2f",
                    cmap="viridis",
                    cbar=True,
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    annot_kws={"size": 12})

        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Cluster Distance', fontsize=12)
        ax.set_ylabel('Cluster Std', fontsize=12)

        # Format y-axis ticks to 1 decimal place
        yticks = [f"{float(y.get_text()):.1f}" for y in ax.get_yticklabels()]
        ax.set_yticklabels(yticks)

    plt.tight_layout()
    plt.savefig('absolute_performance_grid.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Run the experiment grid with specified ranges
    print("Running experiment grid with:")
    print("Cluster distances: 0 to 3.5, step 0.5")
    print("Cluster stds: 0.2 to 0.8, step 0.1")

    results_df = run_experiment_grid(
        cluster_distances=np.arange(0, 4.0, 0.5),  # 1.0, 1.5, ..., 4.0
        cluster_stds=np.arange(0.2, 0.8, 0.1)  # 0.2, 0.3, ..., 0.6
    )

    # Save results with 3 decimal places
    results_df.round(3).to_csv('cluster_performance_grid.csv', index=False)
    print("Results saved to cluster_performance_grid.csv")

    # Plot results
    print("Generating visualizations...")
    plot_results_grid(results_df)
    print("Visualizations saved as PNG files")