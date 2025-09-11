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
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
from sklearn.decomposition import PCA


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

# Define a mapping from lithology names to BCOLOR indices
LITHOLOGY_COLOR_MAP = {
    # Map lithology names to specific BCOLOR indices
    'Regional metamorphic': 1,
    'Contact metamorphic': 2,
    'Felsic volcanic': 3,
    'I/m/u volcanic': 4,
    'Felsic intrusive': 5,
    'I/m/u intrusive': 6,
    'Psammitic sediment': 7,
    'Pelitic sediment': 8,
    'Carbonaceous rock': 9,
    'Chert': 10,
    'Ironstone': 11,
    'Massive sulphide': 12,
    'Sedimentary cover': 13,
    # Add mappings for your specific lithology names
    'Mafic volcanic': 4,  # Map to I/m/u volcanic
    'Metamorphic rocks': 1,  # Map to Regional metamorphic
    'Intermediate intrusive': 6,  # Map to I/m/u intrusive
}

# Define colors for rock types (using Bcolor from previous experiments)
BCOLOR = np.array([
    [1, 1, 1],  # 0 Pad/Background
    [0.4940, 0.1840, 0.5560],  # 1 Regional metamorphic
    [1, 0, 1],  # 2 Contact metamorphic
    [0.4660, 0.6740, 0.1880],  # 3 Felsic volcanic
    [0, 0.5, 0],  # 4 I/m/u volcanic
    [1, 0, 0],  # 5 Felsic intrusive
    [0.6350, 0.0780, 0.1840],  # 6 I/m/u intrusive
    [1, 1, 0],  # 7 Psammitic sediment
    [0.9290, 0.6940, 0.1250],  # 8 Pelitic sediment
    [0.75, 0.75, 0],  # 9 Carbonaceous rock
    [0.8500, 0.3250, 0.0980],  # 10 Chert
    [0.3010, 0.7450, 0.9330],  # 11 Ironstone
    [0, 0, 0],  # 12 Massive sulphide (using black as placeholder)
    [0.7, 0.7, 0.7],  # 13 Sedimentary Cover
])

# Create a custom colormap from the BCOLOR array
from matplotlib.colors import ListedColormap

lithology_cmap = ListedColormap(BCOLOR)


def get_color_index_for_lithology(lithology_name, lithology_mapping, q_number):
    """Get the color index for a lithology name based on the mapping."""
    # First, try to get the color index from our predefined mapping
    if lithology_name in LITHOLOGY_COLOR_MAP:
        return LITHOLOGY_COLOR_MAP[lithology_name]

    # If not found, try to find a similar name
    for mapped_name, color_index in LITHOLOGY_COLOR_MAP.items():
        if mapped_name.lower() in lithology_name.lower() or lithology_name.lower() in mapped_name.lower():
            return color_index

    # If still not found, use a default based on the lithology ID
    # Find the lithology ID for this name
    lithology_id = None
    for lid, name in lithology_mapping.items():
        if name == lithology_name:
            lithology_id = lid
            break

    # Use the lithology ID modulo the number of colors as a fallback
    if lithology_id is not None:
        return lithology_id % len(BCOLOR)

    # Ultimate fallback
    return 0


def create_high_dpi_figures(final_labels, lithology_map, geophys_data, q_number, output_dir, dpi=300):
    """Create high-resolution figures at 300+ DPI with improved visualization."""
    lith_mapping = lithology_mappings.get(q_number, {})

    # Create figure directory
    fig_dir = os.path.join(output_dir, "High_DPI_Figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Create a larger figure to accommodate all subplots
    fig, axes = plt.subplots(2, 3, figsize=(28, 14))
    fig.suptitle(f'Lithology Classification Results - {q_number}', fontsize=20, fontweight='bold')

    # Common settings for all subplots
    for ax in axes.flat:
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylim(lithology_map.shape[0] - 0.5, -0.5)  # Y increases upwards
        ax.set_aspect('equal')  # Keep square pixels

    # Get unique lithology IDs for consistent coloring
    unique_lithologies = np.unique(lithology_map)
    unique_lithologies = unique_lithologies[unique_lithologies > 0]  # Remove background if any

    # Use the custom lithology colormap
    n_colors = len(BCOLOR)
    cmap = lithology_cmap

    # Create a mapping from lithology ID to color index
    lith_id_to_color_index = {}
    for lith_id in unique_lithologies:
        lith_name = lith_mapping.get(int(lith_id), f'Unknown_{lith_id}')
        color_index = get_color_index_for_lithology(lith_name, lith_mapping, q_number)
        lith_id_to_color_index[lith_id] = color_index

    # Apply the color mapping to the lithology maps
    lithology_map_colored = np.zeros_like(lithology_map, dtype=int)
    final_labels_colored = np.zeros_like(final_labels, dtype=int)

    for lith_id in unique_lithologies:
        if lith_id in lith_id_to_color_index:
            color_index = lith_id_to_color_index[lith_id]
            lithology_map_colored[lithology_map == lith_id] = color_index
            final_labels_colored[final_labels == lith_id] = color_index

    # Original lithology map
    im1 = axes[0, 0].imshow(lithology_map_colored, cmap=cmap, interpolation='nearest',
                            vmin=0, vmax=n_colors - 1)
    axes[0, 0].set_title('Original Lithology Map', fontsize=14, pad=10)
    axes[0, 0].set_xticks(range(0, lithology_map.shape[1], 100))
    axes[0, 0].set_yticks(range(0, lithology_map.shape[0], 100))
    axes[0, 0].grid(True, color='lightgray', linestyle='--', linewidth=0.3)
    # Flip y-axis so it increases upwards
    axes[0, 0].set_ylim(lithology_map.shape[0] - 0.5, -0.5)
    axes[0, 0].set_aspect('equal')

    # Final classified map
    im2 = axes[0, 1].imshow(final_labels_colored, cmap=cmap, interpolation='nearest',
                            vmin=0, vmax=n_colors - 1)
    axes[0, 1].set_title('SOM Classification Result', fontsize=14, pad=10)
    axes[0, 1].set_xticks(range(0, final_labels.shape[1], 100))
    axes[0, 1].set_yticks(range(0, final_labels.shape[0], 100))
    axes[0, 1].grid(True, color='lightgray', linestyle='--', linewidth=0.3)
    # Flip y-axis so it increases upwards
    axes[0, 1].set_ylim(lithology_map.shape[0] - 0.5, -0.5)
    axes[0, 1].set_aspect('equal')

    # Enhanced difference map - show specific misclassifications
    diff_map = np.zeros_like(lithology_map, dtype=float)

    # Create a more detailed difference representation
    for i in range(lithology_map.shape[0]):
        for j in range(lithology_map.shape[1]):
            if lithology_map[i, j] != final_labels[i, j]:
                # Encode both true and predicted values in the difference map
                # Use a formula that creates unique values for each (true, predicted) pair
                diff_map[i, j] = lithology_map[i, j] * 10 + final_labels[i, j]
            else:
                diff_map[i, j] = 0  # No difference

    # Create a custom colormap for the difference visualization
    diff_cmap = plt.colormaps['viridis'].resampled(20)
    im3 = axes[0, 2].imshow(diff_map, cmap=diff_cmap, interpolation='nearest')
    axes[0, 2].set_title('Detailed Classification Differences', fontsize=14, pad=10)
    axes[0, 2].set_xticks(range(0, diff_map.shape[1], 100))
    axes[0, 2].set_yticks(range(0, diff_map.shape[0], 100))
    axes[0, 2].grid(True, color='lightgray', linestyle='--', linewidth=0.3)
    # Flip y-axis so it increases upwards
    axes[0, 2].set_ylim(lithology_map.shape[0] - 0.5, -0.5)
    axes[0, 2].set_aspect('equal')

    # Geophysical data
    grav_key = f"{q_number}-Grav.bmp"
    mag_key = f"{q_number}-Mag.bmp"

    if grav_key in geophys_data:
        # Use origin='lower' to ensure y-axis increases upward
        im4 = axes[1, 0].imshow(geophys_data[grav_key]['raw'], cmap='RdBu_r', origin='lower')
        axes[1, 0].set_title('Gravity Data', fontsize=14, pad=10)
        axes[1, 0].set_xticks(range(0, geophys_data[grav_key]['raw'].shape[1], 100))
        axes[1, 0].set_yticks(range(0, geophys_data[grav_key]['raw'].shape[0], 100))
        # Add colorbar for gravity data
        plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)

    if mag_key in geophys_data:
        # Use origin='lower' to ensure y-axis increases upward
        im5 = axes[1, 1].imshow(geophys_data[mag_key]['raw'], cmap='RdBu_r', origin='lower')
        axes[1, 1].set_title('Magnetic Data', fontsize=14, pad=10)
        axes[1, 1].set_xticks(range(0, geophys_data[mag_key]['raw'].shape[1], 100))
        axes[1, 1].set_yticks(range(0, geophys_data[mag_key]['raw'].shape[0], 100))
        # Add colorbar for magnetic data
        plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)

    # Legend for lithology
    axes[1, 2].axis('off')  # Turn off the axis for the legend
    legend_elements = []
    for lith_id in unique_lithologies:
        name = lith_mapping.get(int(lith_id), f'Class_{lith_id}')
        color_index = lith_id_to_color_index.get(lith_id, 0)
        color = BCOLOR[color_index] if color_index < len(BCOLOR) else [0, 0, 0]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color,
                                             edgecolor='black', label=f'{lith_id}: {name}'))

    axes[1, 2].legend(handles=legend_elements, loc='center',
                      title='Lithology Legend', fontsize=10,
                      title_fontsize=12, framealpha=1)

    # Remove colorbars for the first two panels (im1 and im2)
    # Only add colorbar for the difference map
    diff_cbar = fig.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    diff_cbar.set_label('Difference Encoding (True*10 + Predicted)', rotation=270, labelpad=20)

    # Adjust layout to accommodate suptitle
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])  # Removed space for the bottom text

    # Save at high DPI
    output_path = os.path.join(fig_dir, f'{q_number}_som_classification_results_{dpi}dpi.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"High-DPI figure saved: {output_path}")

    # Also create a separate detailed difference analysis figure
    create_detailed_difference_analysis(lithology_map, final_labels, q_number, lith_mapping, output_dir, dpi)

    return output_path

def create_detailed_difference_analysis(true_labels, pred_labels, q_number, lith_mapping, output_dir, dpi=300):
    """Create a detailed analysis of classification differences with comprehensive ID diagnostics."""
    fig_dir = os.path.join(output_dir, "High_DPI_Figures")

    # DIAGNOSTIC: Print comprehensive information about the data
    print(f"\n{'=' * 60}")
    print(f"DIAGNOSTIC INFO FOR {q_number}")
    print(f"{'=' * 60}")

    # Check the lithology mapping first
    print(f"Expected lithology mapping for {q_number}:")
    expected_mapping = lithology_mappings.get(q_number, {})
    for lid, name in expected_mapping.items():
        print(f"  ID {lid}: {name}")
    print(f"Expected number of classes: {len(expected_mapping)}")

    # Analyze true labels
    print(f"\nTrue labels analysis:")
    print(f"  Shape: {true_labels.shape}")
    print(f"  Data type: {true_labels.dtype}")
    print(f"  Min value: {np.min(true_labels)}")
    print(f"  Max value: {np.max(true_labels)}")

    unique_true_all = np.unique(true_labels)
    print(f"  All unique values: {unique_true_all}")

    for val in unique_true_all:
        count = np.sum(true_labels == val)
        percentage = count / true_labels.size * 100
        print(f"    Value {val}: {count} pixels ({percentage:.2f}%)")

    # Analyze predicted labels
    print(f"\nPredicted labels analysis:")
    print(f"  Shape: {pred_labels.shape}")
    print(f"  Data type: {pred_labels.dtype}")
    print(f"  Min value: {np.min(pred_labels)}")
    print(f"  Max value: {np.max(pred_labels)}")

    unique_pred_all = np.unique(pred_labels)
    print(f"  All unique values: {unique_pred_all}")

    for val in unique_pred_all:
        count = np.sum(pred_labels == val)
        percentage = count / pred_labels.size * 100
        print(f"    Value {val}: {count} pixels ({percentage:.2f}%)")

    # Check for background handling
    print(f"\nBackground analysis:")
    background_true = np.sum(true_labels == 0)
    background_pred = np.sum(pred_labels == 0)
    print(f"  Background pixels in true: {background_true}")
    print(f"  Background pixels in pred: {background_pred}")

    # Decide how to handle unique values
    # Option 1: Include ALL unique values (including 0 if it exists)
    all_unique_ids_including_zero = np.unique(np.concatenate([unique_true_all, unique_pred_all]))
    print(f"  All unique IDs (including 0): {all_unique_ids_including_zero}")

    # Option 2: Exclude 0 (traditional background exclusion)
    all_unique_ids_excluding_zero = all_unique_ids_including_zero[all_unique_ids_including_zero > 0]
    print(f"  All unique IDs (excluding 0): {all_unique_ids_excluding_zero}")

    # Option 3: Use the expected mapping keys
    expected_ids = list(expected_mapping.keys())
    print(f"  Expected IDs from mapping: {expected_ids}")

    # Let's use a hybrid approach: include all IDs that appear in the data OR are expected
    all_relevant_ids = np.unique(np.concatenate([
        all_unique_ids_including_zero,
        expected_ids
    ]))

    # Remove 0 only if it's clearly a background (has >50% of pixels)
    total_pixels = true_labels.size
    zero_percentage = np.sum(true_labels == 0) / total_pixels

    if zero_percentage > 0.5:
        print(f"  Treating 0 as background ({zero_percentage:.1%} of pixels)")
        all_relevant_ids = all_relevant_ids[all_relevant_ids > 0]
    else:
        print(f"  Including 0 as a valid class ({zero_percentage:.1%} of pixels)")

    print(f"  Final IDs to analyze: {all_relevant_ids}")

    # Create confusion matrix using all relevant IDs
    n_classes = len(all_relevant_ids)
    confusion_matrix = np.zeros((n_classes, n_classes))

    # Create mapping from lithology ID to matrix index
    id_to_idx = {lid: idx for idx, lid in enumerate(all_relevant_ids)}

    print(f"\nID to matrix index mapping:")
    for lid, idx in id_to_idx.items():
        name = expected_mapping.get(lid, f'Unknown_{lid}')
        print(f"  ID {lid} -> Index {idx}: {name}")

    # Fill confusion matrix
    print(f"\nBuilding confusion matrix...")
    for true_id in all_relevant_ids:
        for pred_id in all_relevant_ids:
            true_idx = id_to_idx[true_id]
            pred_idx = id_to_idx[pred_id]
            count = np.sum((true_labels == true_id) & (pred_labels == pred_id))
            confusion_matrix[true_idx, pred_idx] = count
            if count > 0:
                print(f"  True {true_id} -> Pred {pred_id}: {count} pixels")

    # Check if we have empty classes
    row_sums = confusion_matrix.sum(axis=1)
    col_sums = confusion_matrix.sum(axis=0)

    print(f"\nMatrix validation:")
    print(f"  Matrix shape: {confusion_matrix.shape}")
    print(f"  Total pixels in matrix: {np.sum(confusion_matrix)}")
    print(f"  Row sums (true class totals): {row_sums}")
    print(f"  Col sums (pred class totals): {col_sums}")

    # Identify empty classes
    empty_true_classes = [all_relevant_ids[i] for i in range(n_classes) if row_sums[i] == 0]
    empty_pred_classes = [all_relevant_ids[i] for i in range(n_classes) if col_sums[i] == 0]

    if empty_true_classes:
        print(f"  WARNING: Empty true classes: {empty_true_classes}")
    if empty_pred_classes:
        print(f"  WARNING: Empty predicted classes: {empty_pred_classes}")

    # Normalize by row to show percentage of each true class
    confusion_matrix_norm = np.zeros_like(confusion_matrix)
    for i in range(n_classes):
        if row_sums[i] > 0:
            confusion_matrix_norm[i, :] = confusion_matrix[i, :] / row_sums[i]

    # Create lithology labels with both ID and name
    lithology_labels = []
    for lid in all_relevant_ids:
        lith_name = expected_mapping.get(lid, f'Unknown_{lid}')
        # Truncate long names for better display
        if len(lith_name) > 15:
            lith_name = lith_name[:12] + '...'
        lithology_labels.append(f'ID{lid}\n{lith_name}')

    print(f"\nFinal lithology labels: {lithology_labels}")
    print(f"{'=' * 60}")

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 14))
    fig.suptitle(f'Detailed Classification Analysis - {q_number}\n'
                 f'({n_classes} classes analyzed)', fontsize=16, fontweight='bold')

    # Manually break long labels into multiple lines
    def break_labels(labels, max_len=12):
        cleaned = []
        for label in labels:
            # Extract only the 'ID' followed by digits (e.g., 'ID2')
            match = re.search(r'ID\d+', label)
            short_label = match.group(0) if match else label  # fallback to original if no match
            # Apply line breaking if needed
            broken = '\n'.join([short_label[i:i + max_len] for i in range(0, len(short_label), max_len)])
            cleaned.append(broken)
        return cleaned

    broken_labels = break_labels(lithology_labels)

    # Plot raw counts
    im1 = ax1.imshow(confusion_matrix, cmap='turbo', aspect='equal')  # Changed to 'equal' for same scale
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14)
    ax1.set_xlabel('Predicted Lithology', fontsize=12)
    ax1.set_ylabel('True Lithology', fontsize=12)
    ax1.set_xticks(range(n_classes))
    ax1.set_yticks(range(n_classes))
    ax1.set_xticklabels(broken_labels, rotation=0, ha='center', fontsize=12)
    ax1.set_yticklabels(broken_labels, rotation=90, fontsize=12)
    ax1.invert_yaxis()

    # Add colorbar for counts
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Number of Pixels', rotation=270, labelpad=20)

    # Add text annotations for counts
    for i in range(n_classes):
        for j in range(n_classes):
            count = int(confusion_matrix[i, j])
            if count > 0:
                color = 'white' if confusion_matrix[i, j] > np.max(confusion_matrix) / 2 else 'white'
                ax1.text(j, i, str(count), ha='center', va='center',
                         color=color, fontsize=8, weight='bold')

    # Plot normalized percentages
    im2 = ax2.imshow(confusion_matrix_norm, cmap='turbo', aspect='equal', vmin=0, vmax=1)  # Changed to 'equal'
    ax2.set_title('Confusion Matrix (Normalized by True Class)', fontsize=14)
    ax2.set_xlabel('Predicted Lithology', fontsize=12)
    ax2.set_ylabel('True Lithology', fontsize=12)
    ax2.set_xticks(range(n_classes))
    ax2.set_yticks(range(n_classes))
    ax2.set_xticklabels(broken_labels, rotation=0, ha='center', fontsize=12)
    ax2.set_yticklabels(broken_labels, rotation=90, fontsize=12)
    ax2.invert_yaxis()

    # Add colorbar for percentages
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Proportion of True Class', rotation=270, labelpad=20)

    # Add text annotations for percentages
    for i in range(n_classes):
        for j in range(n_classes):
            percentage = confusion_matrix_norm[i, j]
            if percentage > 0.01:
                color = 'white' if percentage > 0.5 else 'black'
                ax2.text(j, i, f'{percentage:.2f}', ha='center', va='center',
                         color=color, fontsize=8, weight='bold')

    # Adjust layout with more space for labels
    plt.subplots_adjust(bottom=0.45, top=0.9, wspace=0.1, left=0.2, right=0.95)

    # Save the detailed analysis figure
    analysis_path = os.path.join(fig_dir, f'{q_number}_detailed_difference_analysis_{dpi}dpi.png')
    plt.savefig(analysis_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    # Create a lithology-specific difference map with BCOLOR
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create a mapping from lithology ID to color index
    lith_id_to_color_index = {}
    for lith_id in all_relevant_ids:
        lith_name = expected_mapping.get(int(lith_id), f'Unknown_{lith_id}')
        color_index = get_color_index_for_lithology(lith_name, expected_mapping, q_number)
        lith_id_to_color_index[lith_id] = color_index

    # Create a difference map where each pixel shows the true lithology if misclassified, and 0 if correct
    diff_map = np.zeros_like(true_labels)
    for i in range(true_labels.shape[0]):
        for j in range(true_labels.shape[1]):
            if true_labels[i, j] != pred_labels[i, j] and true_labels[i, j] in all_relevant_ids:
                diff_map[i, j] = lith_id_to_color_index.get(true_labels[i, j], 0)

    # Plot with BCOLOR colormap
    im = ax.imshow(diff_map, cmap=lithology_cmap, vmin=0, vmax=len(BCOLOR) - 1, interpolation='nearest')
    ax.set_title(f'Misclassification Map - {q_number}\n(Color shows true lithology where misclassified)', fontsize=14)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Add grid and styling
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(0, diff_map.shape[1], 100))
    ax.set_yticks(np.arange(0, diff_map.shape[0], 100))
    ax.set_ylim(diff_map.shape[0] - 0.5, -0.5)  # Y increases upward

    # Calculate comprehensive statistics
    total_relevant_pixels = np.sum(np.isin(true_labels, all_relevant_ids))
    correct_pixels = np.sum((true_labels == pred_labels) & np.isin(true_labels, all_relevant_ids))
    accuracy = correct_pixels / total_relevant_pixels * 100 if total_relevant_pixels > 0 else 0

    # Calculate per-class statistics
    class_stats = []
    for lid in all_relevant_ids:
        true_mask = true_labels == lid
        pred_mask = pred_labels == lid

        # True positives, false positives, false negatives
        tp = np.sum(true_mask & pred_mask)
        fp = np.sum(~true_mask & pred_mask)
        fn = np.sum(true_mask & ~pred_mask)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        lith_name = expected_mapping.get(int(lid), f'Unknown_{lid}')
        class_stats.append({
            'ID': lid,
            'Name': lith_name,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'True_Count': np.sum(true_mask),
            'Pred_Count': np.sum(pred_mask)
        })

    # Add comprehensive text box with statistics
    stats_text = f'Overall Statistics:\n'
    stats_text += f'Relevant Pixels: {total_relevant_pixels:,}\n'
    stats_text += f'Correct: {correct_pixels:,}\n'
    stats_text += f'Accuracy: {accuracy:.2f}%\n'
    stats_text += f'Classes Found: {len(all_relevant_ids)}\n\n'

    stats_text += 'Per-Class Performance:\n'
    for stat in class_stats:
        stats_text += f"ID{stat['ID']} ({stat['Name'][:8]}):\n"
        stats_text += f"  F1: {stat['F1']:.3f}\n"
        stats_text += f"  Pixels: {stat['True_Count']}\n"

    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()

    # Save the lithology-specific difference map
    lith_diff_path = os.path.join(fig_dir, f'{q_number}_lithology_difference_map_{dpi}dpi.png')
    plt.savefig(lith_diff_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    # Create and save detailed diagnostic report
    report_path = os.path.join(fig_dir, f'{q_number}_diagnostic_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Diagnostic Report for {q_number}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Expected classes: {len(expected_mapping)}\n")
        f.write(f"Found classes: {len(all_relevant_ids)}\n")
        f.write(f"Expected IDs: {list(expected_mapping.keys())}\n")
        f.write(f"Found IDs: {list(all_relevant_ids)}\n")
        f.write(f"Missing IDs: {set(expected_mapping.keys()) - set(all_relevant_ids)}\n")
        f.write(f"Extra IDs: {set(all_relevant_ids) - set(expected_mapping.keys())}\n\n")

        for stat in class_stats:
            f.write(f"ID {stat['ID']}: {stat['Name']}\n")
            f.write(f"  True pixels: {stat['True_Count']}\n")
            f.write(f"  Pred pixels: {stat['Pred_Count']}\n")
            f.write(f"  F1 score: {stat['F1']:.4f}\n\n")

    print(f"Detailed difference analysis saved: {analysis_path}")
    print(f"Lithology difference map saved: {lith_diff_path}")
    print(f"Diagnostic report saved: {report_path}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Classes analyzed: {len(all_relevant_ids)} (expected: {len(expected_mapping)})")


def load_processed_data(data_dir):
    """Load processed data from NPZ files created by the first script."""
    geophys_data = {}
    planview_data = {}
    lithology_data = {}

    # Load geophysical data
    for i in range(1, 7):
        q_number = f"Q{i:03d}"
        for field_type in ['Grav', 'Mag']:
            file_name = f"{q_number}-{field_type}_processed.npz"
            file_path = os.path.join(data_dir, file_name)

            if os.path.exists(file_path):
                try:
                    data = np.load(file_path)
                    geophys_data[f"{q_number}-{field_type}.bmp"] = dict(data)
                    data.close()
                    print(f"Loaded geophysical data: {file_name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

    # Load lithology data directly from the properly indexed files
    for i in range(1, 7):
        q_number = f"Q{i:03d}"
        file_name = f"{q_number}-planview_processed.npz"
        file_path = os.path.join(data_dir, file_name)

        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                lithology_map = data['expanded']
                lithology_data[f"{q_number}-lithology"] = lithology_map
                data.close()
                print(f"Loaded lithology data: {file_name}")
            except Exception as e:
                print(f"Error loading lithology map for {q_number}: {str(e)}")

                # Fallback: try to load from planview data
                planview_file = f"{q_number}-planview_processed.npz"
                planview_path = os.path.join(data_dir, planview_file)

                if os.path.exists(planview_path):
                    try:
                        pv_data = np.load(planview_path)
                        planview_img = pv_data['expanded']
                        planview_data[f"{q_number}-planview.bmp"] = planview_img

                        # Convert planview to lithology map (this is the old method)
                        lithology_map = rgb_to_lithology_id(planview_img, q_number)
                        lithology_data[f"{q_number}-lithology"] = lithology_map

                        pv_data.close()
                        print(f"Converted planview to lithology map for {q_number}")
                    except Exception as e2:
                        print(f"Error converting planview to lithology map for {q_number}: {str(e2)}")
        else:
            print(f"Warning: Could not find lithology map for {q_number}")

    return geophys_data, planview_data, lithology_data


def load_lithology_map(q_number, data_dir):
    """Load lithology map from NPZ file."""
    file_path = os.path.join(data_dir, f"{q_number}-lithology.npz")
    if os.path.exists(file_path):
        try:
            data = np.load(file_path)
            lithology_map = data['lithology']
            data.close()
            print(f"Loaded lithology map for {q_number}")
            return lithology_map
        except Exception as e:
            print(f"Error loading lithology map for {q_number}: {str(e)}")

    # Fallback: try to load from planview data
    file_path = os.path.join(data_dir, f"{q_number}-planview_processed.npz")
    if os.path.exists(file_path):
        try:
            data = np.load(file_path)
            planview_data = data['expanded']
            data.close()
            # Convert RGB to lithology ID (old method)
            lithology_map = rgb_to_lithology_id(planview_data, q_number)
            print(f"Converted planview to lithology map for {q_number}")
            return lithology_map
        except Exception as e:
            print(f"Error converting planview to lithology map for {q_number}: {str(e)}")

    print(f"Warning: Could not load lithology map for {q_number}")
    return None


def rgb_to_lithology_id(rgb_array, q_number):
    """Convert RGB values to lithology IDs based on the mapping for the specific Q model."""
    if rgb_array.ndim != 3:
        return rgb_array  # Already a lithology ID map

    # Get the lithology mapping for this Q number
    lith_mapping = lithology_mappings.get(q_number, {})

    # Create a unique identifier for each RGB color
    color_ids = (rgb_array[..., 0].astype(np.uint32) << 16 |
                 rgb_array[..., 1].astype(np.uint32) << 8 |
                 rgb_array[..., 2].astype(np.uint32))

    # Get unique colors in the image
    unique_colors = np.unique(color_ids)

    # Create a mapping from color ID to lithology ID
    color_to_lithology = {}
    for i, color_id in enumerate(unique_colors):
        if color_id == 0:  # Black background
            color_to_lithology[color_id] = 0
        else:
            # Assign lithology IDs sequentially starting from 1
            lithology_id = min(i, len(lith_mapping))
            color_to_lithology[color_id] = lithology_id

    # Apply the mapping
    lithology_map = np.zeros_like(color_ids, dtype=np.uint8)
    for color_id, lith_id in color_to_lithology.items():
        lithology_map[color_ids == color_id] = lith_id

    return lithology_map


def prepare_features_for_som(geophys_data, q_number, nn_model=None, nn_scaler=None, nn_le=None):
    """Prepare integrated 2D features for SOM training, including NN predictions."""
    features_list = []
    feature_names = []

    # Extract features from both gravity and magnetic data
    grav_key = f"{q_number}-Grav.bmp"
    mag_key = f"{q_number}-Mag.bmp"

    if grav_key in geophys_data:
        grav_data = geophys_data[grav_key]
        features_list.extend([grav_data['raw'].flatten()])
        feature_names.extend(['grav_raw'])

    if mag_key in geophys_data:
        mag_data = geophys_data[mag_key]
        features_list.extend([mag_data['raw'].flatten()])
        feature_names.extend(['mag_raw'])

    if not features_list:
        return None, None, None

    # Stack features and standardize
    features = np.column_stack(features_list)

    # Add NN predictions as features if available
    if nn_model is not None and nn_scaler is not None:
        # First, get NN predictions using the original 8 features
        # Re-extract all 8 features for NN prediction
        nn_features_list = []

        grav_key = f"{q_number}-Grav.bmp"
        mag_key = f"{q_number}-Mag.bmp"

        if grav_key in geophys_data:
            grav_data = geophys_data[grav_key]
            nn_features_list.extend([
                grav_data['raw'].flatten(),
                grav_data['1VD'].flatten(),
                grav_data['tile'].flatten(),
                grav_data['analytical_signal'].flatten()
            ])

        if mag_key in geophys_data:
            mag_data = geophys_data[mag_key]
            nn_features_list.extend([
                mag_data['raw'].flatten(),
                mag_data['1VD'].flatten(),
                mag_data['tile'].flatten(),
                mag_data['analytical_signal'].flatten()
            ])

        # Create the full feature set for NN prediction
        features_for_nn = np.column_stack(nn_features_list)

        # Standardize features using NN scaler
        features_scaled_nn = nn_scaler.transform(features_for_nn)

        # Get NN predictions (probabilities)
        nn_probs = nn_model.predict(features_scaled_nn, verbose=0)

        # Add NN probabilities as additional features
        features = np.column_stack([features, nn_probs])

        # Add feature names for NN probabilities
        for i in range(nn_probs.shape[1]):
            feature_names.append(f'nn_class_{i}')

    # Standardize all features for SOM
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, feature_names, scaler


def load_nn_model(model_path="NN_Results/lithology_model.h5"):
    """Load the trained neural network model and associated preprocessing objects."""
    model = keras.models.load_model(model_path)

    # Load label encoder and scaler
    le_classes = np.load("NN_Results/label_encoder_classes.npy", allow_pickle=True)
    scaler_mean = np.load("NN_Results/scaler_mean.npy")
    scaler_scale = np.load("NN_Results/scaler_scale.npy")

    # Recreate the scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale

    # Recreate the label encoder
    le = LabelEncoder()
    le.classes_ = le_classes

    return model, scaler, le


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


def kmeans_cluster_aggregation(clusters, lithology_map, target_k):
    """Use K-means to aggregate SOM clusters to target number of classes."""
    print(f"Aggregating {len(np.unique(clusters))} clusters to {target_k} using K-means...")

    # Reshape clusters to 2D for processing
    clusters_2d = clusters.reshape(lithology_map.shape)

    # Get unique cluster IDs
    unique_clusters = np.unique(clusters_2d)

    # Calculate cluster centroids based on their position in the grid
    centroids = []
    for cluster_id in unique_clusters:
        positions = np.argwhere(clusters_2d == cluster_id)
        if len(positions) > 0:
            centroid = np.mean(positions, axis=0)
            centroids.append(centroid)
        else:
            centroids.append([0, 0])

    centroids = np.array(centroids)

    # Apply K-means to cluster centroids
    kmeans = KMeans(n_clusters=target_k, random_state=42)
    kmeans.fit(centroids)

    # Create mapping from original cluster to new aggregated cluster
    cluster_mapping = {}
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mapping[cluster_id] = kmeans.labels_[i] + 1  # +1 to avoid 0

    # Apply mapping to all clusters
    aggregated_clusters = np.zeros_like(clusters_2d)
    for cluster_id, new_id in cluster_mapping.items():
        aggregated_clusters[clusters_2d == cluster_id] = new_id

    return aggregated_clusters.flatten(), cluster_mapping


def assign_final_lithology_ids(merged_clusters, lithology_priors, q_number):
    """Assign final lithology IDs using argmax rule with data likelihood and section prior."""
    lith_mapping = lithology_mappings.get(q_number, {})

    final_labels = np.zeros_like(merged_clusters)
    mapping_table = {}

    unique_clusters = np.unique(merged_clusters)
    unique_clusters = unique_clusters[unique_clusters > 0]

    # Flatten the lithology_priors to 1D to ensure consistent indexing
    lithology_priors_flat = lithology_priors.ravel()

    for cluster_id in unique_clusters:
        mask = merged_clusters == cluster_id
        prior_values = lithology_priors_flat[mask]

        # Find most common lithology in this cluster region
        if len(prior_values) > 0:
            # Count occurrences of each lithology (excluding 0)
            lithology_counts = Counter(prior_values[prior_values > 0])
            if lithology_counts:
                # Get the most common lithology
                most_common_lith = lithology_counts.most_common(1)[0][0]
                final_labels[mask] = most_common_lith
                # Convert uint8 key to string for JSON serialization
                mapping_table[str(cluster_id)] = {
                    'lithology_id': int(most_common_lith),  # Ensure int for JSON
                    'lithology_name': lith_mapping.get(int(most_common_lith), f'Unknown_{most_common_lith}'),
                    'pixels': int(np.sum(mask)),  # Ensure int for JSON
                    'confidence': lithology_counts[most_common_lith] / np.sum(mask) if np.sum(mask) > 0 else 0
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


def process_single_model_with_som(q_number, geophys_data, lithology_data, output_dir, nn_model=None, nn_scaler=None,
                                  nn_le=None):
    """Process a single geological model using SOM with lithology priors and NN constraints."""
    print(f"\n{'=' * 60}")
    print(f"Processing {q_number} with SOM, lithology priors, and NN constraints")
    print(f"{'=' * 60}")

    # Get lithology data
    lith_key = f"{q_number}-lithology"
    if lith_key not in lithology_data:
        print(f"No lithology data for {q_number}")
        return None

    # Create temporary directory for intermediate files
    temp_dir = os.path.join(output_dir, "temp", q_number)
    os.makedirs(temp_dir, exist_ok=True)

    # Process lithology map
    lithology_map = lithology_data[lith_key]
    print(f"Lithology map shape: {lithology_map.shape}")
    print(f"Unique values in lithology map: {np.unique(lithology_map)}")

    # Check if we need to adjust the lithology IDs (if they start from 0 instead of 1)
    unique_values = np.unique(lithology_map)
    if 0 in unique_values and len(unique_values) > 1:
        # Check if the values are 0-indexed (0, 1, 2, 3) but should be 1-indexed (1, 2, 3, 4)
        max_value = np.max(lithology_map)
        expected_max = len(lithology_mappings.get(q_number, {}))

        if max_value == expected_max - 1:
            print(f"Adjusting lithology IDs from 0-indexed to 1-indexed")
            # Add 1 to all non-zero values
            lithology_map = lithology_map + 1
            print(f"Adjusted unique values: {np.unique(lithology_map)}")

    # Prepare features for SOM with NN constraints
    features_scaled, feature_names, scaler = prepare_features_for_som(
        geophys_data, q_number, nn_model, nn_scaler, nn_le
    )

    if features_scaled is None:
        print(f"No geophysical data available for {q_number}")
        return None

    # Check if features are valid
    if features_scaled.shape[0] == 0 or features_scaled.shape[1] == 0:
        print(f"Invalid features shape: {features_scaled.shape}")
        return None

    # Train SOM with lithology priors and NN constraints
    som_shape = (25, 25)
    som, initial_clusters = train_som_with_priors(
        features_scaled, lithology_map.flatten(), q_number,
        som_shape=som_shape, n_iterations=15000
    )

    print(f"Initial clusters shape: {initial_clusters.shape}")

    # Save initial cluster map as NPZ
    initial_clusters_path = os.path.join(temp_dir, "initial_som_clusters.npz")
    save_npz_file(initial_clusters_path, initial_clusters, lithology_map.shape)

    # Save lithology map for processing as NPZ
    lithology_path = os.path.join(temp_dir, "lithology_map.npz")
    save_npz_file(lithology_path, lithology_map.flatten(), lithology_map.shape)

    # Step 1: Aggregate clusters to target K using K-means
    target_k = len(lithology_mappings.get(q_number, {}))
    aggregated_clusters, cluster_mapping = kmeans_cluster_aggregation(
        initial_clusters, lithology_map, target_k
    )

    print(f"Aggregated clusters shape: {aggregated_clusters.shape}")

    # Save aggregated clusters
    aggregated_path = os.path.join(temp_dir, "aggregated_clusters.npz")
    save_npz_file(aggregated_path, aggregated_clusters, lithology_map.shape)

    # Step 2: Assign final lithology IDs
    print("\nStep 2: Assigning final lithology IDs...")
    final_labels, mapping_table = assign_final_lithology_ids(
        aggregated_clusters, lithology_map, q_number
    )

    print(f"Final labels shape: {final_labels.shape}")

    # Reshape final_labels to match the original lithology map dimensions
    final_labels_2d = final_labels.reshape(lithology_map.shape)
    print(f"Reshaped final labels to: {final_labels_2d.shape}")

    # Step 3: Export results
    print("\nStep 3: Exporting results...")

    # Save final label map as NPZ
    final_labels_path = os.path.join(output_dir, f"{q_number}_final_lithology_map.npz")
    save_npz_file(final_labels_path, final_labels_2d.flatten(), final_labels_2d.shape)

    # Save mapping table
    mapping_df = pd.DataFrame.from_dict(mapping_table, orient='index')
    mapping_df.index.name = 'cluster_id'
    mapping_path = os.path.join(output_dir, f"{q_number}_lithology_mapping.csv")
    mapping_df.to_csv(mapping_path)

    # Calculate metrics - use flattened arrays for comparison
    metrics = calculate_metrics(lithology_map.flatten(), final_labels, features_scaled)

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{q_number}_classification_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Create high-DPI figures - use the 2D array for visualization
    fig_path = create_high_dpi_figures(
        final_labels_2d, lithology_map, geophys_data, q_number, output_dir, dpi=300
    )

    # Also save final labels as PNG for visualization
    png_path = os.path.join(output_dir, f"{q_number}_final_lithology_map.png")
    plt.figure(figsize=(12, 9))
    plt.imshow(final_labels_2d, cmap='tab10')
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
        'final_labels': final_labels_2d,
        'mapping_table': mapping_table
    }


def enhanced_analysis_with_som(geophys_data, lithology_data, output_dir):
    """Enhanced analysis using SOM with lithology priors and NN constraints for all available models."""
    print("\n" + "=" * 80)
    print("ENHANCED SOM-BASED LITHOLOGY CLASSIFICATION WITH NN CONSTRAINTS")
    print("=" * 80)

    # Load the neural network model
    try:
        nn_model, nn_scaler, nn_le = load_nn_model()
        print("Loaded neural network model for lithology constraints")
    except Exception as e:
        print(f"Could not load neural network model: {str(e)}")
        nn_model, nn_scaler, nn_le = None, None, None

    results = {}

    # Process each available Q model
    available_q_numbers = []
    for lith_key in lithology_data.keys():
        if lith_key.startswith('Q') and lith_key.endswith('-lithology'):
            q_number = lith_key.split('-')[0]
            available_q_numbers.append(q_number)

    print(f"Found {len(available_q_numbers)} models to process: {available_q_numbers}")

    for q_number in sorted(available_q_numbers):
        try:
            result = process_single_model_with_som(
                q_number, geophys_data, lithology_data, output_dir,
                nn_model, nn_scaler, nn_le
            )
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
    print("Starting SOM-enhanced processing with NN constraints...")

    # Create output directory
    output_dir = "SyntheticNoddy_SOM_NN"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")

    # Load processed data from first script
    data_dir = r"SyntheticNoddy\Data"
    print(f"\nLoading processed data from: {data_dir}")

    geophys_data, planview_data, lithology_data = load_processed_data(data_dir)

    if geophys_data or lithology_data:
        print(f"\nLoaded {len(geophys_data)} geophysical datasets and {len(lithology_data)} lithology datasets")
        print("\nPerforming enhanced SOM-based analysis with NN constraints...")
        enhanced_analysis_with_som(geophys_data, lithology_data, output_dir)
        print("\nEnhanced SOM analysis with NN constraints complete!")
    else:
        print("\nNo processed data found. Please run the first script first.")


if __name__ == "__main__":
    main()