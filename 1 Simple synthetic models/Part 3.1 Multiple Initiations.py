import subprocess
import shutil
from pathlib import Path

# Base directory (current folder)
BASE_DIR = Path.cwd()

SCRIPT = BASE_DIR / "1 Simple synthetic models\Part 3 Other UML.py"

# Folder names
result_folders = [f"Results - I{i}" for i in range(2, 11)]

# Files produced by Part 3
OUTPUT_FILES = [
    "clustering_comparison_results_UML.csv",
    "precision_ratio_comparison.png",
    "recall_ratio_comparison.png",
    "f1_ratio_comparison.png",
    "accuracy_ratio_comparison.png",
]

for folder in result_folders:
    out_dir = BASE_DIR / folder
    out_dir.mkdir(exist_ok=True)

    print(f"Running Part 3 for {folder}...")

    # Run the script
    subprocess.run(
        ["python", SCRIPT.name],
        cwd=BASE_DIR,
        check=True
    )

    # Move outputs into the instance folder
    for file in OUTPUT_FILES:
        src = BASE_DIR / file
        if src.exists():
            shutil.move(src, out_dir / file)

print("All runs completed.")
