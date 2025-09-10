#!/usr/bin/env python3
# Aggregate_cluster_postprocess with NPZ support
import argparse
import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd


def load_npz(path):
    """Load data from NPZ file."""
    data = np.load(path)
    return data['data']


def save_npz(path, arr):
    """Save data to NPZ file."""
    np.savez_compressed(path, data=arr)


def aggregate_clusters(clusters, geology, features, weights, target_k, min_sim):
    unique = np.unique(clusters[clusters > 0])
    mapping = {u: u for u in unique}
    merges = []

    # Calculate cluster statistics first
    cluster_stats = {}
    for cluster_id in unique:
        mask = clusters == cluster_id
        cluster_stats[cluster_id] = {
            'geology_values': geology[mask],
            'feature_means': [np.mean(f[mask]) for f in features] if features else []
        }

    # Calculate similarities
    sims = {}
    for i in unique:
        for j in unique:
            if i >= j:
                continue

            # Get cluster statistics
            stats_i = cluster_stats[i]
            stats_j = cluster_stats[j]

            # Calculate overlap similarity based on geology distribution
            geo_i = stats_i['geology_values']
            geo_j = stats_j['geology_values']

            # Use histogram intersection for geology similarity
            all_geology = np.unique(np.concatenate([geo_i, geo_j]))
            hist_i = np.histogram(geo_i, bins=all_geology, density=True)[0]
            hist_j = np.histogram(geo_j, bins=all_geology, density=True)[0]
            overlap = np.minimum(hist_i, hist_j).sum()

            # Calculate feature similarities
            feat_sims = []
            if features:
                for idx, (mean_i, mean_j) in enumerate(zip(stats_i['feature_means'], stats_j['feature_means'])):
                    # Normalize feature similarity to 0-1 range
                    feat_range = np.max(features[idx]) - np.min(features[idx])
                    if feat_range > 0:
                        feat_sim = 1 - abs(mean_i - mean_j) / feat_range
                    else:
                        feat_sim = 1.0
                    feat_sims.append(feat_sim)

            # Weighted similarity
            if features:
                sim = weights[0] * overlap + sum(w * s for w, s in zip(weights[1:], feat_sims))
            else:
                sim = overlap

            sims[(i, j)] = sim

    # Merge clusters
    while len(set(mapping.values())) > target_k:
        if not sims:
            break

        pair, best = max(sims.items(), key=lambda kv: kv[1])
        if best < min_sim:
            break

        i, j = pair
        for k, v in mapping.items():
            if v == j:
                mapping[k] = i

        merges.append((i, j, best))

        # Remove merged cluster from similarities
        sims = {k: v for k, v in sims.items() if j not in k}

    return mapping, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--geology", required=True)
    ap.add_argument("--features", nargs="*")
    ap.add_argument("--weights", nargs="*", type=float, default=[0.6, 0.3, 0.1])
    ap.add_argument("--target_k", type=int, required=True)
    ap.add_argument("--min_sim", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log")
    ap.add_argument("--map")
    ap.add_argument("--report")
    args = ap.parse_args()

    clusters = load_npz(args.clusters)
    geology = load_npz(args.geology)
    feats = [load_npz(f) for f in args.features] if args.features else []

    # Ensure all arrays have the same shape
    if feats:
        for i, feat in enumerate(feats):
            if feat.shape != clusters.shape:
                print(f"Warning: Feature {i} shape {feat.shape} doesn't match clusters shape {clusters.shape}")
                # Resize feature to match clusters shape if needed
                feats[i] = np.resize(feat, clusters.shape)

    mapping, merges = aggregate_clusters(clusters, geology, feats, args.weights,
                                         args.target_k, args.min_sim)
    merged = np.vectorize(lambda x: mapping.get(x, 0))(clusters)
    save_npz(args.out, merged)

    if args.log:
        pd.DataFrame(merges, columns=["keep", "drop", "similarity"]).to_csv(args.log, index=False)
    if args.map:
        pd.DataFrame(list(mapping.items()), columns=["cluster", "merged"]).to_csv(args.map, index=False)
    if args.report:
        with open(args.report, "w") as f:
            f.write(f"Merged to {len(set(mapping.values()))} clusters\n")
            f.write(f"Original clusters: {len(np.unique(clusters[clusters > 0]))}\n")


if __name__ == "__main__":
    main()