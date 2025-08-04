import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.visualize import plot_matrix

# =====================================================================================

some_saliency_map_path = "saliency_maps/human_Results/AGAPIG/heatmaps/ANGRY_canonical.npy"
save_path = "saliency_maps/"

# =====================================================================================

if __name__ == "__main__":
    saliency_map = np.load(some_saliency_map_path)
    saliency_map_shape = saliency_map.shape
    print(saliency_map_shape)

    if len(saliency_map_shape) != 2:
        raise ValueError(f"Expected 2D saliency map, got shape {saliency_map_shape}")
    else:
        print(f"Saliency map shape will be: {saliency_map_shape}")

    # make a (shape) numpy array with
    std_saliency_map_zeroes = np.zeros(saliency_map_shape)
    std_saliency_map_ones = np.ones(saliency_map_shape)

    # add the nans
    std_saliency_map_zeroes[np.isnan(saliency_map)] = np.nan
    std_saliency_map_ones[np.isnan(saliency_map)] = np.nan

    # plot the map
    plot_matrix(std_saliency_map_zeroes, title="Standard Saliency Map - Zeroes", block=False, vmin=0, vmax=1)
    plot_matrix(std_saliency_map_ones, title="Standard Saliency Map - Ones", block=False, vmin=0, vmax=1)
    plt.show()

    # save the maps
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, "std_saliency_map_zeroes.npy"), std_saliency_map_zeroes)
    np.save(os.path.join(save_path, "std_saliency_map_ones.npy"), std_saliency_map_ones)
    print(f"Saved standard saliency maps to {save_path}")