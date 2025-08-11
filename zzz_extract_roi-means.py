import os

import numpy as np
from modules.compare_saliency_maps import compute_heatmap_statistics
from modules.roi_statistics import roi_mean

INPUT_FOLDER = "zzz_input_saliency_maps"
OUTPUT_FOLDER = "zzz_output_means_vectors"

DEBUG = False

if __name__ == "__main__":
    for heatmap_fname in os.listdir(INPUT_FOLDER):
        heatmap_relpath = os.path.join(INPUT_FOLDER, heatmap_fname)
        heatmap = np.load(heatmap_relpath)
        
        print("==================================================")
        print(f"Computing mean-vectors for {heatmap_fname}")
        stats, _, _ = compute_heatmap_statistics(heatmap, heatmap_relpath, roi_mean, "mean", separate_lr=True,
                                   weigh_roi_overlap=False, debug=DEBUG, force_recalculate=True, dont_save=True)

        # Save the stats to the output folder
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        output_path = os.path.join(OUTPUT_FOLDER, f"{heatmap_fname.split('.')[0]}_mean-vector.npy")
        np.save(output_path, stats, allow_pickle=True)