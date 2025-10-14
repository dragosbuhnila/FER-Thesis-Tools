import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.join(".", "saliency_maps")
HUMAN_SALIENCIES_DIR = os.path.join(BASE_DIR, "human_Results")
BASEFACE = os.path.join(BASE_DIR, "basefaces", "baseface_neutral_reshaped.png")

def overlay_saliency_with_nans(baseface_path, saliency_path):
    # Load the baseface image
    baseface = np.array(Image.open(baseface_path).convert("L"))  # Convert to grayscale
    baseface = baseface / 255.0  # Normalize to [0, 1]

    # Load the saliency map
    saliency = np.load(saliency_path)

    # Create a mask for NaN values in the saliency map
    nan_mask = np.isnan(saliency)

    # Plot the baseface and overlay the saliency map
    plt.figure(figsize=(8, 8))
    plt.imshow(baseface, cmap="gray", interpolation="nearest")  # Display the baseface
    plt.imshow(nan_mask, cmap="Reds", alpha=0.5, interpolation="nearest")  # Overlay NaN mask in red
    plt.title(f"Saliency Map with NaNs: {os.path.basename(saliency_path)}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Collect saliency files
    some_saliencies = [file for file in os.listdir(os.path.join(HUMAN_SALIENCIES_DIR, "MARFRO/heatmaps")) if file.endswith(".npy")]
    some_saliencies = [os.path.join(HUMAN_SALIENCIES_DIR, os.path.join("MARFRO", "heatmaps"), file) for file in some_saliencies]
    some_other_saliencies = [file for file in os.listdir(os.path.join(HUMAN_SALIENCIES_DIR, "MATVIN/heatmaps")) if file.endswith(".npy")]
    some_other_saliencies = [os.path.join(HUMAN_SALIENCIES_DIR, os.path.join("MATVIN", "heatmaps"), file) for file in some_other_saliencies]
    some_saliencies = some_saliencies + some_other_saliencies

    # Display each saliency map
    for saliency_file in some_saliencies:
        print(f"Displaying saliency file: {saliency_file}")
        overlay_saliency_with_nans(BASEFACE, saliency_file)