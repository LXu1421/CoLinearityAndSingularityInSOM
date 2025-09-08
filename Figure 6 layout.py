import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

# 🔧 Define input and output directories
input_dir = 'SyntheticNoddy\Edited'     # ← Change this to your actual input folder
output_dir = 'SyntheticNoddy\Edited'   # ← Change this to your desired output folder

# Create figure with 3 rows and 4 columns of subplots
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
fig.subplots_adjust(wspace=0.05, hspace=0.05)  # Tighten spacing

# Define the letters for each set
set_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# Process each of the 6 sets
for i in range(6):
    row = i // 2
    col = (i % 2) * 2
    set_num = f"{i + 1:03d}"

    # 🔄 Construct full image paths
    img1_path = os.path.join(input_dir, f"Q{set_num}_PCA_PCA.png")
    img2_path = os.path.join(input_dir, f"Q{set_num}_PCA_lithology_centers.png")

    # Load images or use placeholder
    if os.path.exists(img1_path):
        img1 = mpimg.imread(img1_path)
    else:
        img1 = np.ones((100, 100, 3)) * (i / 6)

    if os.path.exists(img2_path):
        img2 = mpimg.imread(img2_path)
    else:
        img2 = np.ones((100, 100, 3)) * (1 - i / 6)

    # Display images with preserved aspect ratio
    ax1 = axes[row, col]
    ax1.imshow(img1)
    ax1.set_aspect('equal')  # Preserve aspect ratio
    ax1.axis('off')

    ax2 = axes[row, col + 1]
    ax2.imshow(img2)
    ax2.set_aspect('equal')  # Preserve aspect ratio
    ax2.axis('off')

    # Add label above image pair
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    center_x = (pos1.x0 + pos2.x1) / 2
    top_y = max(pos1.y1, pos2.y1) + 0.01  # Slightly tighter

    fig.text(center_x, top_y, set_labels[i],
             fontfamily='Arial', fontsize=12,
             ha='center', va='bottom')

# 💾 Save figure with tight bounding box
output_path = os.path.join(output_dir, 'figure_layout.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()