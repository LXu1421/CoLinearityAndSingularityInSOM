import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tabulate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from minisom import MiniSom
from scipy.ndimage import gaussian_filter

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)     # For Python random operations

# Update the style to use a valid matplotlib style
plt.style.use('seaborn-v0_8')  # This works for newer matplotlib versions
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


## Model 1: Spherical Clusters with Noise and Collinear Features
def generate_spherical_clusters(n_samples=1000):
    # Generate two more distinct spherical clusters
    cluster_center1 = np.array([-1, -1])  # Cluster 1 center
    cluster_center2 = np.array([1, 1])  # Cluster 2 center (increased distance)
    cluster_std = 0.4  # Reduced from 0.5 to make clusters more compact

    # Generate clusters with reduced variance
    cluster1 = np.random.randn(n_samples // 2, 2) * cluster_std + cluster_center1
    cluster2 = np.random.randn(n_samples // 2, 2) * cluster_std + cluster_center2

    # Labels (0 for cluster1, 1 for cluster2)
    labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    # Original features (PC1 and PC2)
    X = np.vstack([cluster1, cluster2])

    # Add collinear feature (PC1b is a noisy version of PC1)
    PC1b = X[:, 0] + np.random.normal(0, 0.02, n_samples)  # Reduced noise from 0.1 to 0.05

    # Add noise features
    noise3 = np.random.randn(n_samples)
    noise4 = np.random.randn(n_samples)
    noise5 = np.random.randn(n_samples)

    # Combine all features
    X_extended = np.column_stack([X, PC1b, noise3, noise4, noise5])

    return X, X_extended, labels


def plot_pairwise_scatter(X, X_extended, labels):
    feature_names = ['PC1', 'PC2', 'PC1b', 'Noise3', 'Noise4', 'Noise5']
    n_features = X_extended.shape[1]

    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                axes[i, j].hist(X_extended[:, i], bins=30, alpha=0.7)
            else:
                axes[i, j].scatter(X_extended[labels == 0, j], X_extended[labels == 0, i],
                                   c='blue', s=10, alpha=0.5, label='Cluster 1')
                axes[i, j].scatter(X_extended[labels == 1, j], X_extended[labels == 1, i],
                                   c='red', s=10, alpha=0.5, label='Cluster 2')

            if i == n_features - 1:
                axes[i, j].set_xlabel(feature_names[j])
            if j == 0:
                axes[i, j].set_ylabel(feature_names[i])

    plt.suptitle('Pairwise Scatter Plots of Features (Cluster 1 vs Cluster 2)')
    plt.tight_layout()
    plt.savefig('figure1_pairwise_scatter.png')
    plt.close()


def plot_correlation_matrix(X_extended):
    feature_names = ['PC1', 'PC2', 'PC1b', 'Noise3', 'Noise4', 'Noise5']
    corr_matrix = np.corrcoef(X_extended.T)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names)
    ax.set_yticklabels(feature_names)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_title("Correlation Matrix of Input Features")
    plt.tight_layout()
    plt.savefig('figure3_correlation_matrix.png')
    plt.close()


def train_som(X, som_shape=(20, 20), n_iter=1000):
    som = MiniSom(som_shape[0], som_shape[1], X.shape[1],
                 sigma=0.8, learning_rate=0.3, random_seed=42)  # Add random_seed
    som.random_weights_init(X)
    som.train_random(X, n_iter, verbose=False)
    return som


def plot_u_matrix(som, title, filename):
    u_matrix = som.distance_map()

    plt.figure(figsize=(6, 5))
    plt.pcolor(u_matrix.T, cmap='bone_r')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def evaluate_som_performance(som, X, labels):
    winner_coordinates = np.array([som.winner(x) for x in X])
    winner_indices = winner_coordinates[:, 0] * som._weights.shape[1] + winner_coordinates[:, 1]

    # Use majority voting for each neuron to assign labels
    neuron_labels = {}
    for neuron in np.unique(winner_indices):
        neuron_mask = winner_indices == neuron
        if sum(neuron_mask) > 0:
            neuron_labels[neuron] = np.argmax(np.bincount(labels[neuron_mask].astype(int)))

    # Predict labels based on neuron assignments
    pred_labels = np.array([neuron_labels[neuron] for neuron in winner_indices])

    # Calculate metrics
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    accuracy = accuracy_score(labels, pred_labels)

    return precision, recall, f1, accuracy


def model1_experiment():
    # Generate data
    X, X_extended, labels = generate_spherical_clusters()

    # Plot pairwise scatter (Figure 1)
    plot_pairwise_scatter(X, X_extended, labels)

    # Plot correlation matrix (Figure 3)
    plot_correlation_matrix(X_extended)

    # Train SOMs and plot U-Matrices
    som_simple = train_som(X)
    som_extended = train_som(X_extended)

    # Plot U-Matrices (Figure 2)
    plot_u_matrix(som_simple, "U-Matrix: PC1 + PC2", "figure2_umap_pc1_pc2.png")
    plot_u_matrix(som_extended, "U-Matrix: PC1 + PC2 + PC1b + Noise", "figure2_umap_extended.png")

    # Evaluate performance
    metrics_simple = evaluate_som_performance(som_simple, X, labels)
    metrics_extended = evaluate_som_performance(som_extended, X_extended, labels)

    # Create performance table
    performance_df = pd.DataFrame({
        'Dataset': ['PC1 + PC2', 'PC1 + PC2 + PC1b + Noise'],
        'Precision': [metrics_simple[0], metrics_extended[0]],
        'Recall': [metrics_simple[1], metrics_extended[1]],
        'F1-Score': [metrics_simple[2], metrics_extended[2]],
        'Accuracy': [metrics_simple[3], metrics_extended[3]]
    })

    print("Model 1 Performance:")
    print(performance_df.to_markdown(index=False))

    return performance_df


## Model 2 & 3: TMI Data with Derivatives
def generate_tmi_data(size=100, gradual=True, noise_level=0.05):
    # Create grid
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))

    # Create circular intrusion
    center = (0.5, 0.5)
    radius = 0.15
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    if gradual:
        # Gradual boundary (Model 2)
        tmi = np.exp(-distance ** 2 / (2 * (radius / 2) ** 2))
    else:
        # Crisp boundary (Model 3)
        tmi = np.where(distance <= radius, 1.0, 0.0)

    # Add noise
    tmi_noisy = tmi + np.random.normal(0, noise_level, (size, size))

    # Calculate derivatives
    # 1st vertical derivative (1VD)
    grad_y, grad_x = np.gradient(tmi_noisy)
    vd = -grad_y  # Negative for downward continuation

    # Tilt angle
    tilt = np.arctan2(vd, grad_x)

    # Analytical signal amplitude (ASA)
    asa = np.sqrt(grad_x ** 2 + grad_y ** 2 + vd ** 2)

    return tmi_noisy, vd, tilt, asa, distance <= radius


def plot_tmi_data(tmi, vd, tilt, asa, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Function to plot with correct y-axis
    def plot_with_correct_axis(ax, data, title, cmap):
        im = ax.imshow(data, cmap=cmap, origin='lower', extent=[0, 1, 0, 1])
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        return im

    # Plot TMI
    plot_with_correct_axis(axes[0, 0], tmi, 'TMI (Raw + Noise)', 'magma')

    # Plot 1VD
    plot_with_correct_axis(axes[0, 1], vd, '1st Vertical Derivative', 'seismic')

    # Plot Tilt
    plot_with_correct_axis(axes[1, 0], tilt, 'Tilt Angle', 'RdBu')

    # Plot ASA
    plot_with_correct_axis(axes[1, 1], asa, 'Analytical Signal Amplitude', 'viridis')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def som_classification(X, labels, som_shape=(20, 20), n_iter=1000):
    # Standardize data
    scaler = StandardScaler()
    # Reshape input data if it's 2D (like TMI data)
    if len(X.shape) > 2:
        original_shape = X.shape[:2]
        X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[2]))
    else:
        original_shape = None
        X_scaled = scaler.fit_transform(X)

    # Train SOM
    som = MiniSom(som_shape[0], som_shape[1], X_scaled.shape[1], sigma=1.5, learning_rate=0.5)
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, n_iter, verbose=False)

    # Get predictions
    winner_coordinates = np.array([som.winner(x) for x in X_scaled])
    winner_indices = winner_coordinates[:, 0] * som._weights.shape[1] + winner_coordinates[:, 1]

    # Assign labels based on majority voting in each neuron
    neuron_labels = {}
    flat_labels = labels.flatten()
    for neuron in np.unique(winner_indices):
        neuron_mask = winner_indices == neuron
        if sum(neuron_mask) > 0:
            neuron_labels[neuron] = np.argmax(np.bincount(flat_labels[neuron_mask].astype(int)))

    # Create prediction map
    pred_map = np.array([neuron_labels[neuron] for neuron in winner_indices])
    if original_shape is not None:
        pred_map = pred_map.reshape(original_shape)

    # Calculate metrics - ensure both are flattened
    flat_pred = pred_map.flatten()
    flat_labels = labels.flatten()
    precision = precision_score(flat_labels, flat_pred)
    recall = recall_score(flat_labels, flat_pred)
    f1 = f1_score(flat_labels, flat_pred)
    accuracy = accuracy_score(flat_labels, flat_pred)

    return pred_map, (precision, recall, f1, accuracy)


def smooth_labels(labels, sigma=1):
    return gaussian_filter(labels.astype(float), sigma=sigma) > 0.5


def model23_experiment():
    # Generate data for both models
    # Model 2: Gradual intrusion
    tmi_gradual, vd_gradual, tilt_gradual, asa_gradual, labels_gradual = generate_tmi_data(gradual=True)

    # Model 3: Crisp intrusion
    tmi_crisp, vd_crisp, tilt_crisp, asa_crisp, labels_crisp = generate_tmi_data(gradual=False)

    # Plot TMI data (Figures 4 and 5)
    plot_tmi_data(tmi_gradual, vd_gradual, tilt_gradual, asa_gradual,
                  "Model 2: TMI Data with Derivatives (Gradual Boundary)", "figure4_tmi_gradual.png")
    plot_tmi_data(tmi_crisp, vd_crisp, tilt_crisp, asa_crisp,
                  "Model 3: TMI Data with Derivatives (Crisp Boundary)", "figure5_tmi_crisp.png")

    # Prepare datasets for classification
    # Raw data only
    raw_data_gradual = tmi_gradual.reshape(-1, 1)
    raw_data_crisp = tmi_crisp.reshape(-1, 1)

    # Augmented data (raw + derivatives)
    augmented_data_gradual = np.dstack([tmi_gradual, vd_gradual, tilt_gradual, asa_gradual])
    augmented_data_crisp = np.dstack([tmi_crisp, vd_crisp, tilt_crisp, asa_crisp])

    # PCA-augmented data
    pca = PCA(n_components=2)
    pca_data_gradual = pca.fit_transform(augmented_data_gradual.reshape(-1, 4))
    pca_data_crisp = pca.fit_transform(augmented_data_crisp.reshape(-1, 4))

    # Prepare all data configurations with their names
    data_configurations = [
        (raw_data_gradual, labels_gradual, 'Gradual Raw'),
        (augmented_data_gradual, labels_gradual, 'Gradual Augmented'),
        (pca_data_gradual, labels_gradual, 'Gradual PCA'),
        (raw_data_crisp, labels_crisp, 'Crisp Raw'),
        (augmented_data_crisp, labels_crisp, 'Crisp Augmented'),
        (pca_data_crisp, labels_crisp, 'Crisp PCA')
    ]

    # Perform classification for all configurations
    results = []
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for i, (X, y, model_name) in enumerate(data_configurations):
        y_flat = y.flatten()

        # SOM classification
        pred_map, metrics = som_classification(X, y)

        # Smooth predictions
        smoothed_pred = smooth_labels(pred_map)

        # Store results
        results.append({
            'Model': model_name,
            'Type': model_name.split()[0],
            'Features': model_name.split()[1],
            'Precision': metrics[0],
            'Recall': metrics[1],
            'F1-Score': metrics[2],
            'Accuracy': metrics[3]
        })

        # Plot results (only gradual for figure 6)
        # In the plotting loop within model23_experiment():
        # Plot results (only gradual for figure 6)
        if 'Gradual' in model_name:
            row = ['Raw', 'Augmented', 'PCA'].index(model_name.split()[1])

            # Common settings for all subplots
            extent = [0, 100, 0, 100]
            xticks = np.arange(0, 101, 25)
            yticks = np.arange(0, 101, 25)
            grid_style = {'linestyle': ':', 'alpha': 0.5, 'color': 'grey'}

            # True labels (first column)
            im = axes[row, 0].imshow(y, cmap='binary', origin='lower', extent=extent)
            axes[row, 0].set_title('True Labels' if row == 0 else '', pad=10)
            axes[row, 0].set_ylabel('Y (m)', labelpad=10)
            axes[row, 0].set_xlabel('X (m)', labelpad=10)
            axes[row, 0].set_xticks(xticks)
            axes[row, 0].set_yticks(yticks)
            axes[row, 0].grid(**grid_style)

            # SOM predictions (second column)
            im = axes[row, 1].imshow(pred_map.reshape(100, 100) if len(pred_map.shape) == 1 else pred_map,
                                     cmap='binary', origin='lower', extent=extent)
            axes[row, 1].set_title('SOM Prediction' if row == 0 else '', pad=10)
            axes[row, 1].set_ylabel('Y (m)', labelpad=10)
            axes[row, 1].set_xlabel('X (m)', labelpad=10)
            axes[row, 1].set_xticks(xticks)
            axes[row, 1].set_yticks(yticks)
            axes[row, 1].grid(**grid_style)

            # Smoothed predictions (third column)
            im = axes[row, 2].imshow(
                smoothed_pred.reshape(100, 100) if len(smoothed_pred.shape) == 1 else smoothed_pred,
                cmap='binary', origin='lower', extent=extent)
            axes[row, 2].set_title('Smoothed SOM' if row == 0 else '', pad=10)
            axes[row, 2].set_ylabel('Y (m)', labelpad=10)
            axes[row, 2].set_xlabel('X (m)', labelpad=10)
            axes[row, 2].set_xticks(xticks)
            axes[row, 2].set_yticks(yticks)
            axes[row, 2].grid(**grid_style)

            # Add row labels on the left
            if row == 0:
                axes[row, 0].text(-0.35, 0.5, 'Raw Data', rotation=90,
                                  va='center', ha='center', transform=axes[row, 0].transAxes, fontsize=12)
            elif row == 1:
                axes[row, 0].text(-0.35, 0.5, 'Raw+Derivatives', rotation=90,
                                  va='center', ha='center', transform=axes[row, 0].transAxes, fontsize=12)
            elif row == 2:
                axes[row, 0].text(-0.35, 0.5, 'PCA Processed', rotation=90,
                                  va='center', ha='center', transform=axes[row, 0].transAxes, fontsize=12)

        # Adjust layout to prevent overlap
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(left=0.15)  # Make space for row labels


    plt.suptitle('SOM Predictions for Gradual Intrusion with Noise', y=0.98)
    plt.tight_layout()
    plt.savefig('figure6_som_predictions.png')
    plt.close()

    # Create performance table
    results_df = pd.DataFrame(results)
    print("\nModel 2 & 3 Performance:")
    print(results_df.to_markdown(index=False))

    return results_df


## Run all experiments
if __name__ == '__main__':
    print("Running Model 1 Experiment...")
    model1_results = model1_experiment()

    print("\nRunning Model 2 & 3 Experiments...")
    model23_results = model23_experiment()

    # Save performance tables with 3 decimal places
    model1_results.round(3).to_csv('model1_performance.csv', index=False)
    model23_results.round(3).to_csv('model23_performance.csv', index=False)

    print("\nPerformance tables saved as CSV files with 3 decimal places")