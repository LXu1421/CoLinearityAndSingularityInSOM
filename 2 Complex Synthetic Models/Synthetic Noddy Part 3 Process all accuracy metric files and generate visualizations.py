import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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

# Define colors for plotting
Bcolor = np.array([
    [0.4940, 0.1840, 0.5560],  # Regional metamorphic
    [1, 0, 1],  # Contact metamorphic
    [0.4660, 0.6740, 0.1880],  # Felsic volcanic
    [0, 0.5, 0],  # I/m/u volcanic
    [1, 0, 0],  # Felsic intrusive
    [0.6350, 0.0780, 0.1840],  # I/m/u intrusive
    [1, 1, 0],  # Psammitic sediment
    [0.9290, 0.6940, 0.1250],  # Pelitic sediment
    [0.75, 0.75, 0],  # Carbonaceous rock
    [0.8500, 0.3250, 0.0980],  # Chert
    [0.3010, 0.7450, 0.9330],  # Ironstone
    [0, 0, 0],  # Massive sulphide
    [0.7, 0.7, 0.7]  # Sedimentary cover
])

# Create a mapping from lithology names to colors
lithology_colors = {
    'Mafic volcanic': Bcolor[3],  # I/m/u volcanic
    'Sedimentary cover': Bcolor[12],  # Psammitic sediment
    'Psammetic sediment': Bcolor[6],  # Psammitic sediment
    'Psammitic sediment': Bcolor[6],  # Psammitic sediment
    'Psammatic sediment': Bcolor[6],  # Psammitic sediment
    'Felsic intrusive': Bcolor[4],  # Felsic intrusive
    'Pelitic intrusive': Bcolor[7],  # Pelitic sediment
    'Pelitic sediment': Bcolor[7],  # Pelitic sediment
    'Metamorphic rocks': Bcolor[0],  # Regional metamorphic
    'Chert': Bcolor[9],  # Chert
    'Carbonaceous rock': Bcolor[8],  # Carbonaceous rock
    'Intermediate intrusive': Bcolor[5]  # I/m/u intrusive
}

som_shape=7

def process_accuracy_files(input_dir, output_dir):
    """Process all accuracy metric files and generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionaries to store all data for comparisons
    accuracy_comparison = {exp: [] for exp in ['Raw', 'All Features', 'PCA']}
    lithology_accuracies = {}  # To store accuracies by lithology
    q_numbers = []

    # Initialize lithology accuracy storage
    all_lithology_names = set()
    for mapping in lithology_mappings.values():
        all_lithology_names.update(mapping.values())

    for lith_name in all_lithology_names:
        lithology_accuracies[lith_name] = {
            'Raw': [],
            'All Features': [],
            'PCA': []
        }

    for q in range(1, 7):
        q_num = f'Q{q:03d}'
        q_numbers.append(q_num)
        file_path = os.path.join(input_dir, f'{q_num}_accuracy_metrics.csv')

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        lith_mapping = lithology_mappings[q_num]

        # Create a subfolder for this Q number
        q_output_dir = os.path.join(output_dir, q_num)
        os.makedirs(q_output_dir, exist_ok=True)

        # Process each experiment type
        for exp in ['Raw', 'All Features', 'PCA']:
            exp_data = df[df['Experiment'] == exp]
            if exp_data.empty:
                continue

            # Plot lithology centers with standard deviations
            plot_lithology_centers(exp_data, q_num, exp, lith_mapping, q_output_dir)

            # Store overall accuracy data for comparison
            accuracy_comparison[exp].append(exp_data['Overall_Accuracy'].values[0])

            # Store lithology-specific accuracies
            for lith_id, lith_name in lith_mapping.items():
                acc_col = f'Lithology_{lith_id}_Accuracy'
                if acc_col in exp_data.columns:
                    acc_value = exp_data[acc_col].values[0]
                    lithology_accuracies[lith_name][exp].append(acc_value)
                else:
                    lithology_accuracies[lith_name][exp].append(np.nan)

    # Plot overall accuracy comparison
    plot_accuracy_comparison(accuracy_comparison, q_numbers, output_dir)

    # Plot lithology-specific accuracy comparisons
    plot_lithology_accuracy_comparisons(lithology_accuracies, q_numbers, output_dir)


def plot_lithology_accuracy_comparisons(lithology_accuracies, q_numbers, output_dir):
    """Plot accuracy comparison for each lithology across experiments."""
    # Create a subfolder for lithology accuracy plots
    lith_acc_dir = os.path.join(output_dir, "Lithology_Accuracy_Comparisons")
    os.makedirs(lith_acc_dir, exist_ok=True)

    # Define line styles and markers for each experiment
    styles = {
        'Raw': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 8},
        'All Features': {'color': 'green', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'markersize': 8},
        'PCA': {'color': 'red', 'marker': '^', 'linestyle': '-.', 'linewidth': 2, 'markersize': 8}
    }

    # Plot for each lithology
    for lith_name, acc_data in lithology_accuracies.items():
        plt.figure(figsize=(12, 6))

        # Get the color for this lithology from our predefined colors
        lith_color = lithology_colors.get(lith_name, [0.5, 0.5, 0.5])

        # Plot each experiment's accuracy trend for this lithology
        for exp, accuracies in acc_data.items():
            # Filter out NaN values and get corresponding Q numbers
            valid_indices = [i for i, acc in enumerate(accuracies) if not np.isnan(acc)]
            if valid_indices:  # Only plot if we have valid data
                valid_q_numbers = [q_numbers[i] for i in valid_indices]
                valid_accuracies = [accuracies[i] for i in valid_indices]
                plt.plot(valid_q_numbers, valid_accuracies,
                         label=exp,
                         **styles[exp])

        plt.xlabel('Q Number', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Accuracy Comparison for {lith_name}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)  # Accuracy ranges from 0 to 1

        # Add a colored background patch for the lithology
        plt.gca().add_patch(plt.Rectangle((0, 0), 1, 1, fc=lith_color, alpha=0.1,
                                          transform=plt.gca().transAxes, zorder=-1))

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        safe_lith_name = lith_name.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(lith_acc_dir, f'{safe_lith_name}_accuracy_comparison.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved lithology accuracy plot to: {output_path}")


def plot_lithology_centers(exp_data, q_num, exp, lith_mapping, output_dir):
    """Plot lithology centers with standard deviations in PCA space."""
    plt.figure(figsize=(10, 8))

    # Get the number of lithologies from the mapping
    num_lithologies = len(lith_mapping)

    # Plot each lithology center with standard deviation ellipse
    for lith_id in range(1, num_lithologies + 1):
        lith_name = lith_mapping[lith_id]
        pc1 = exp_data[f'Lithology_{lith_id}_Center_PC1'].values[0]
        pc2 = exp_data[f'Lithology_{lith_id}_Center_PC2'].values[0]
        std_dev = exp_data[f'Lithology_{lith_id}_Std_Dev'].values[0]

        # Skip if we have NaN values
        if np.isnan(pc1) or np.isnan(pc2) or np.isnan(std_dev):
            continue

        # Get color for this lithology
        color = lithology_colors.get(lith_name, [0.5, 0.5, 0.5])

        # Plot center point
        plt.scatter(pc1, pc2, color=color, s=100, label=lith_name, edgecolor='black', linewidth=1)

        # Add standard deviation ellipse
        ellipse = Ellipse((pc1, pc2), width=std_dev, height=std_dev,
                          angle=0, color=color, alpha=0.2)
        plt.gca().add_patch(ellipse)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'{q_num} {exp} - Lithology Centers in PCA Space')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(output_dir, f'{q_num}_{exp}_lithology_centers.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved lithology centers plot to: {output_path}")


def plot_accuracy_comparison(accuracy_comparison, q_numbers, output_dir):
    """Plot accuracy comparison across all experiments and Q files."""
    plt.figure(figsize=(12, 6))

    # Define line styles and markers for each experiment
    styles = {
        'Raw': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'All Features': {'color': 'green', 'marker': 's', 'linestyle': '--'},
        'PCA': {'color': 'red', 'marker': '^', 'linestyle': '-.'}
    }

    # Plot each experiment's accuracy trend
    for exp, accuracies in accuracy_comparison.items():
        if accuracies:  # Only plot if we have data
            plt.plot(q_numbers, accuracies,
                     label=exp,
                     **styles[exp],
                     linewidth=2,
                     markersize=8)

    plt.xlabel('Q Number')
    plt.ylabel('Overall Accuracy')
    plt.title('Accuracy Comparison Across Experiments')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1

    # Save the plot
    output_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved accuracy comparison plot to: {output_path}")


if __name__ == "__main__":
    input_dir = "SyntheticNoddy/SOM_Results"
    output_dir = "SyntheticNoddy/Accuracy_Analysis"
    process_accuracy_files(input_dir, output_dir)