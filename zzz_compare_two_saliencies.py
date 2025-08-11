
import os

import numpy as np

from modules.compare_saliency_maps import compute_heatmap_statistics
from modules.roi_statistics import roi_mean, compare_difmean


INPUT_FOLDER = "zzz_input_two_saliency_maps"
OUTPUT_FOLDER = "zzz_output_two_saliency_maps"

DEBUG = False

if __name__ == "__main__":
    heatmap_fnames = os.listdir(INPUT_FOLDER)
    if len(heatmap_fnames) != 2:
        raise ValueError(f"Expected exactly 2 heatmap files in {INPUT_FOLDER}, found {len(heatmap_fnames)}.")

    two_stats = []

    for heatmap_fname in heatmap_fnames:

        heatmap_relpath = os.path.join(INPUT_FOLDER, heatmap_fname)
        heatmap = np.load(heatmap_relpath)

        print("==================================================")
        print(f"Computing mean-vectors for {heatmap_fname}")
        stats, _, _ = compute_heatmap_statistics(heatmap, heatmap_relpath, roi_mean, "mean", separate_lr=False,
                                                   weigh_roi_overlap=False, debug=DEBUG, force_recalculate=True, dont_save=True, roi_type="aus")

        two_stats.append(stats)

    if len(two_stats) != 2:
        raise ValueError("Expected exactly 2 statistics dictionaries.")

    diff_stats = compare_difmean(two_stats[0], two_stats[1])
    print("Differences in mean values between the two heatmaps:")
    print(diff_stats)