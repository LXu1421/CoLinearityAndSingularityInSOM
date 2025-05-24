import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
import matplotlib.ticker as ticker



class ClusterEvaluator:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.methods = {
            'SOM': self.evaluate_som,
            'K-Means (Exclusive)': self.evaluate_kmeans,
            'DBSCAN (Density)': self.evaluate_dbscan,
            'Agglomerative (Hierarchical)': self.evaluate_agglo,
            'GMM (Probabilistic)': self.evaluate_gmm
        }

    def generate_data(self, cluster_distance=2.0, cluster_std=0.4):
        """Generate synthetic data with specified separation and variance"""
        centers = np.array([[-cluster_distance / 2, -cluster_distance / 2],
                            [cluster_distance / 2, cluster_distance / 2]])

        # Generate clusters
        cluster1 = np.random.randn(self.n_samples // 2, 2) * cluster_std + centers[0]
        cluster2 = np.random.randn(self.n_samples // 2, 2) * cluster_std + centers[1]

        # Labels and base features
        labels = np.concatenate([np.zeros(self.n_samples // 2), np.ones(self.n_samples // 2)])
        X = np.vstack([cluster1, cluster2])

        # Extended features
        PC1b = X[:, 0] + np.random.normal(0, 0.02, self.n_samples)
        noise = np.random.randn(self.n_samples, 3)
        X_extended = np.column_stack([X, PC1b, noise])

        return X, X_extended, labels

    def evaluate_method(self, method_func, X, labels):
        """Generic evaluation wrapper for clustering methods"""
        try:
            pred_labels = method_func(X)
            return {
                'precision': precision_score(labels, pred_labels, zero_division=0),
                'recall': recall_score(labels, pred_labels, zero_division=0),
                'f1': f1_score(labels, pred_labels, zero_division=0),
                'accuracy': accuracy_score(labels, pred_labels)
            }
        except Exception as e:
            print(f"Error in {method_func.__name__}: {str(e)}")
            return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}

    # Clustering method implementations
    def evaluate_som(self, X):
        som = MiniSom(5, 5, X.shape[1], sigma=1.0, learning_rate=0.5)
        som.random_weights_init(X)
        som.train_random(X, 100)
        winners = np.array([som.winner(x) for x in X])
        unique_winners = np.unique(winners, axis=0)
        pred_labels = np.zeros(len(X))
        for i, winner in enumerate(winners):
            pred_labels[i] = np.where((unique_winners == winner).all(axis=1))[0][0]
        return pred_labels % 2  # Convert to binary

    def evaluate_kmeans(self, X):
        return KMeans(n_clusters=2, random_state=42).fit_predict(X)

    def evaluate_dbscan(self, X):
        X_scaled = StandardScaler().fit_transform(X)
        return DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled) % 2

    def evaluate_agglo(self, X):
        return AgglomerativeClustering(n_clusters=2).fit_predict(X)

    def evaluate_gmm(self, X):
        return GaussianMixture(n_components=2, random_state=42).fit(X).predict(X)

    def run_experiments(self, cluster_distances, cluster_stds):
        """Run experiments across parameter grid for all methods"""
        results = []

        for dist in cluster_distances:
            for std in cluster_stds:
                X, X_extended, labels = self.generate_data(dist, std)

                for method_name, method_func in self.methods.items():
                    # Evaluate on simple and extended features
                    simple_metrics = self.evaluate_method(method_func, X, labels)
                    extended_metrics = self.evaluate_method(method_func, X_extended, labels)

                    # Calculate ratios
                    ratios = {
                        f'{metric}_ratio': extended_metrics[metric] / simple_metrics[metric]
                        if simple_metrics[metric] != 0 else 0
                        for metric in ['precision', 'recall', 'f1', 'accuracy']
                    }

                    results.append({
                        'method': method_name,
                        'cluster_distance': dist,
                        'cluster_std': std,
                        **{f'simple_{k}': v for k, v in simple_metrics.items()},
                        **{f'extended_{k}': v for k, v in extended_metrics.items()},
                        **ratios
                    })

        return pd.DataFrame(results)

    def plot_results(self, results_df):
        """Visualize results for all methods"""
        sns.set_theme(style="whitegrid")
        metrics = ['precision_ratio', 'recall_ratio', 'f1_ratio', 'accuracy_ratio']

        for metric in metrics:
            plt.figure(figsize=(12, 8))
            pivot = results_df.pivot_table(index=['cluster_std', 'cluster_distance'],
                                           columns='method',
                                           values=metric)
            cluster_stds = results_df['cluster_std'].unique()
            cluster_stds.sort()  # Ensure values are in ascending order

            # Unstack the multi-index for visualization
            pivot = pivot.unstack(level=0)

            # Plot each method's performance
            for i, method in enumerate(self.methods.keys()):
                plt.subplot(2, 3, i + 1)
                ax = sns.heatmap(pivot[method],
                            annot=True,
                            fmt=".0f",
                            cmap="coolwarm",
                            center=1.0,
                            vmin=0.5,
                            vmax=1.5)
                plt.title(f"{method}\n{metric.replace('_', ' ').title()}")
                plt.xlabel('Cluster Distance')
                plt.ylabel('Cluster Std')
                # Format x-axis and y-axis labels to show only one decimal place
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

                # Set y-axis ticks to match cluster_stds and ensure correct order
                ax.set_yticks(range(len(cluster_stds)))  # Ensure correct number of ticks
                ax.set_yticklabels([f"{std:.1f}" for std in cluster_stds])  # Format labels
                # Ensure the y-axis increases from bottom to top (if needed)
                ax.invert_yaxis()

            plt.tight_layout()
            plt.savefig(f'{metric}_comparison.png', dpi=300)
            plt.close()


if __name__ == '__main__':
    evaluator = ClusterEvaluator(n_samples=1000)

    # Run experiments with your specified ranges
    results_df = evaluator.run_experiments(
        cluster_distances=np.arange(0, 3.5, 0.5),
        cluster_stds=np.arange(0.2, 0.8, 0.1)
    )

    # Save and plot results
    results_df.to_csv('clustering_comparison_results_UML.csv', index=False)
    evaluator.plot_results(results_df)