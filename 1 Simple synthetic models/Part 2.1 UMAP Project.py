import numpy as np
import matplotlib.pyplot as plt
import umap

# -----------------------------
# Data generation (as defined)
# -----------------------------
def generate_spherical_clusters1(
    n_samples=1000,
    cluster_distance=2.0,
    cluster_std=0.4
):
    # Cluster centers separated along PC1 only
    cluster_center1 = np.array([-cluster_distance / 2, 0])
    cluster_center2 = np.array([ cluster_distance / 2, 0])

    cluster1 = np.random.randn(n_samples // 2, 2) * cluster_std + cluster_center1
    cluster2 = np.random.randn(n_samples // 2, 2) * cluster_std + cluster_center2

    labels = np.concatenate([
        np.zeros(n_samples // 2),
        np.ones(n_samples // 2)
    ])

    # Raw feature set: PC1, PC2
    X = np.vstack([cluster1, cluster2])

    # Collinear feature
    PC1b = X[:, 0] + np.random.normal(0, 0.02, n_samples)

    # Noise features
    noise_features = np.random.randn(n_samples, 3)

    # Extended feature set
    X_extended = np.column_stack([X, PC1b, noise_features])

    return X, X_extended, labels

def generate_spherical_clusters2(
    n_samples=1000,
    cluster_distance=0.5,
    cluster_std=0.6
):
    # Cluster centers separated along PC1 only
    cluster_center1 = np.array([-cluster_distance / 2, 0])
    cluster_center2 = np.array([ cluster_distance / 2, 0])

    cluster1 = np.random.randn(n_samples // 2, 2) * cluster_std + cluster_center1
    cluster2 = np.random.randn(n_samples // 2, 2) * cluster_std + cluster_center2

    labels = np.concatenate([
        np.zeros(n_samples // 2),
        np.ones(n_samples // 2)
    ])

    # Raw feature set: PC1, PC2
    X = np.vstack([cluster1, cluster2])

    # Collinear feature
    PC1b = X[:, 0] + np.random.normal(0, 0.02, n_samples)

    # Noise features
    noise_features = np.random.randn(n_samples, 3)

    # Extended feature set
    X_extended = np.column_stack([X, PC1b, noise_features])

    return X, X_extended, labels


# -----------------------------
# Generate data
# -----------------------------
X_raw, X_extended1, labels = generate_spherical_clusters1()
X_raw1, X_extended, labels = generate_spherical_clusters2()

# -----------------------------
# UMAP configuration
# -----------------------------
umap_params = dict(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

umap_model = umap.UMAP(**umap_params)

# -----------------------------
# Fit UMAP
# -----------------------------
embedding_raw = umap_model.fit_transform(X_raw)
embedding_extended = umap_model.fit_transform(X_extended)


# -----------------------------
# Plotting
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (a) Raw features
axes[0].scatter(
    embedding_raw[:, 0],
    embedding_raw[:, 1],
    c=labels,
    s=10
)
axes[0].set_title("(a) UMAP: raw feature set (PC1, PC2)")
axes[0].set_xlabel("UMAP-1")
axes[0].set_ylabel("UMAP-2")
axes[0].text(0.05, 0.95, "cluster_distance=2.0,\ncluster_std=0.4",
             transform=axes[0].transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# (b) Extended features
axes[1].scatter(
    embedding_extended[:, 0],
    embedding_extended[:, 1],
    c=labels,
    s=10
)
axes[1].set_title("(b) UMAP: extended feature set (PC1, PC2, PC1b, Noise)")
axes[1].set_xlabel("UMAP-1")
axes[1].set_ylabel("UMAP-2")
axes[1].text(0.05, 0.95, "cluster_distance=0.5,\ncluster_std=0.6",
             transform=axes[1].transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
outname = f"UMAP.png"
plt.savefig(outname, dpi=300, bbox_inches="tight")
plt.close()
