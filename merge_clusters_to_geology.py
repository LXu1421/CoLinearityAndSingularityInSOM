#!/usr/bin/env python3
# merge_clusters_to_geology.py with NPZ support
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage


def load_npz(path):
    """Load data from NPZ file."""
    data = np.load(path)
    return data['data']


def save_npz(path, arr):
    """Save data to NPZ file."""
    np.savez_compressed(path, data=arr)


def sieve_clusters(clusters, min_size):
    """Remove small clusters."""
    labeled, num_features = ndimage.label(clusters > 0)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes >= min_size
    mask_sizes[0] = 0  # keep background
    return mask_sizes[labeled]


def merge_clusters_to_geology(clusters, geology, purity_threshold=0.6, min_size=25):
    """Merge clusters based on geological purity."""
    # Sieve small clusters first
    sieved = sieve_clusters(clusters, min_size)

    unique_clusters = np.unique(sieved[sieved > 0])
    mapping = {}

    for cluster_id in unique_clusters:
        cluster_mask = sieved == cluster_id
        geology_in_cluster = geology[cluster_mask]

        if len(geology_in_cluster) == 0:
            continue

        # Find dominant geology in cluster
        unique_geology, counts = np.unique(geology_in_cluster[geology_in_cluster > 0], return_counts=True)

        if len(unique_geology) == 0:
            continue

        dominant_geology = unique_geology[np.argmax(counts)]
        purity = np.max(counts) / np.sum(counts)

        if purity >= purity_threshold:
            mapping[cluster_id] = dominant_geology
        else:
            # If not pure enough, keep original cluster ID
            mapping[cluster_id] = cluster_id

    # Apply mapping
    result = np.zeros_like(sieved)
    for old_id, new_id in mapping.items():
        result[sieved == old_id] = new_id

    return result, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--geology", required=True)
    ap.add_argument("--out_merged", required=True)
    ap.add_argument("--out_map")
    ap.add_argument("--report")
    ap.add_argument("--purity", type=float, default=0.6)
    ap.add_argument("--sieve_pixels", type=int, default=25)
    ap.add_argument("--fix-ambiguous", action="store_true")
    args = ap.parse_args()

    clusters = load_npz(args.clusters)
    geology = load_npz(args.geology)

    merged, mapping = merge_clusters_to_geology(
        clusters, geology, args.purity, args.sieve_pixels
    )

    save_npz(args.out_merged, merged)

    if args.out_map:
        pd.DataFrame(list(mapping.items()), columns=["cluster", "geology"]).to_csv(args.out_map, index=False)

    if args.report:
        with open(args.report, "w") as f:
            f.write(f"Merged clusters report:\n")
            f.write(f"Original clusters: {len(np.unique(clusters[clusters > 0]))}\n")
            f.write(f"Merged clusters: {len(np.unique(merged[merged > 0]))}\n")
            f.write(f"Purity threshold: {args.purity}\n")
            f.write(f"Min sieve size: {args.sieve_pixels}\n")


if __name__ == "__main__":
    main()