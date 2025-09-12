import os
import pandas as pd
import re
from pathlib import Path


def extract_overall_accuracy(txt_file_path):
    """Extract overall accuracy from diagnostic report txt file"""
    try:
        with open(txt_file_path, 'r') as file:
            content = file.read()

        # Look for F1 scores in the content
        f1_scores = []
        lines = content.split('\n')
        for line in lines:
            if 'F1 score:' in line:
                f1_match = re.search(r'F1 score:\s*([\d.]+)', line)
                if f1_match:
                    f1_scores.append(float(f1_match.group(1)))

        # Calculate overall accuracy as average of F1 scores
        if f1_scores:
            return sum(f1_scores) / len(f1_scores)
        else:
            return 0.0
    except Exception as e:
        print(f"Error reading {txt_file_path}: {e}")
        return 0.0


def extract_lithology_data(csv_file_path):
    """Extract lithology data from mapping CSV file"""
    try:
        df = pd.read_csv(csv_file_path)
        lithology_data = {}

        for _, row in df.iterrows():
            lithology_name = row['lithology_name']
            confidence = row['confidence']
            lithology_data[lithology_name] = confidence

        return lithology_data
    except Exception as e:
        print(f"Error reading {csv_file_path}: {e}")
        return {}


def extract_f1_scores(txt_file_path):
    """Extract F1 scores from diagnostic report txt file"""
    try:
        with open(txt_file_path, 'r') as file:
            content = file.read()

        f1_scores = {}
        lines = content.split('\n')

        for line in lines:
            if 'F1 score:' in line and ':' in line:
                # Extract lithology name and F1 score
                parts = line.split(':')
                if len(parts) >= 3:
                    lithology_name = parts[0].strip()
                    f1_match = re.search(r'F1 score:\s*([\d.]+)', line)
                    if f1_match:
                        f1_scores[lithology_name] = float(f1_match.group(1))

        return f1_scores
    except Exception as e:
        print(f"Error reading {txt_file_path}: {e}")
        return {}


def process_folders():
    """Main function to process all folders and generate summary"""
    # Define the folders to process
    folders = {
        'Raw': 'SyntheticNoddy_SOM_NN_Raw',
        'All': 'SyntheticNoddy_SOM_NN_All',
        'PCA': 'SyntheticNoddy_SOM_NN_PCA'
    }

    # Define all possible lithology units (you may need to adjust this based on your actual data)
    lithology_units = [
        'Sedimentary cover',
        'Psammitic sediment',
        'Felsic intrusive',
        'Pelitic sediment',
        # Add more units as needed based on your complete dataset
    ]

    # Create result dataframe
    results = []

    for input_type, folder_name in folders.items():
        if not os.path.exists(folder_name):
            print(f"Warning: Folder {folder_name} not found, skipping...")
            continue

        # Process each Q file (Q001 to Q006)
        for q_num in range(1, 7):
            model_number = f"Q{q_num:03d}"

            # Construct file paths
            csv_filename = f"{model_number}_lithology_mapping.csv"
            txt_filename = f"{model_number}_diagnostic_report.txt"

            csv_path = os.path.join(folder_name, csv_filename)
            txt_path = os.path.join(folder_name, "High_DPI_Figures", txt_filename)

            # Check if files exist
            if not os.path.exists(csv_path) or not os.path.exists(txt_path):
                print(f"Warning: Files for {model_number} in {input_type} not found, skipping...")
                continue

            # Extract data
            overall_accuracy = extract_overall_accuracy(txt_path)
            lithology_confidences = extract_lithology_data(csv_path)
            f1_scores = extract_f1_scores(txt_path)

            # Create result row
            result_row = {
                'Model Number': model_number,
                'Input': input_type,
                'Overall Accuracy': overall_accuracy
            }

            # Add F1 scores and confidences for each lithology unit
            for unit in lithology_units:
                f1_key = f"{unit} (F1 score)"
                conf_key = f"{unit} (confidence)"

                result_row[f1_key] = f1_scores.get(unit, '-')
                result_row[conf_key] = lithology_confidences.get(unit, '-')

            results.append(result_row)

    # Create DataFrame and save to CSV
    if results:
        df_results = pd.DataFrame(results)

        # Reorder columns for better readability
        columns = ['Model Number', 'Input', 'Overall Accuracy']
        for unit in lithology_units:
            columns.extend([f"{unit} (F1 score)", f"{unit} (confidence)"])

        df_results = df_results[columns]
        df_results.to_csv('SOM_NN_result_book.CSV', index=False)
        print("Results saved to SOM_NN_result_book.CSV")

        return df_results
    else:
        print("No data found to process")
        return pd.DataFrame()


def get_all_lithology_units():
    """Scan all files to get complete list of lithology units"""
    lithology_units = set()
    folders = ['SyntheticNoddy_SOM_NN_Raw', 'SyntheticNoddy_SOM_NN_All', 'SyntheticNoddy_SOM_NN_PCA']

    for folder in folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith('_lithology_mapping.csv'):
                    try:
                        df = pd.read_csv(os.path.join(folder, file))
                        lithology_units.update(df['lithology_name'].unique())
                    except:
                        continue

    return sorted(list(lithology_units))


# Alternative: If you want to automatically detect all lithology units
# lithology_units = get_all_lithology_units()

# Manual list based on your example - you should expand this with all 9 units
lithology_units = [
    'Sedimentary cover',
    'Psammitic sediment',
    'Felsic intrusive',
    'Pelitic sediment',
    # Add the other 5 units that appear in your complete dataset
]

# Run the processing
if __name__ == "__main__":
    result_df = process_folders()
    if not result_df.empty:
        print(result_df.head())