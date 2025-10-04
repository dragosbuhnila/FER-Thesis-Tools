import os
import numpy as np

from modules.compare_saliency_maps import compute_heatmap_statistics
from modules.roi_statistics import roi_mean

INPUT_FOLDER = "zzz_input_saliency_maps"
OUTPUT_FOLDER = "zzz_output_means_vectors"

CANONICAL_ONLY = True

DEBUG = False

if __name__ == "__main__":
    # > Clear the output folder
    if os.path.exists(OUTPUT_FOLDER):
        print(f"Emptying output folder: {OUTPUT_FOLDER}")
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    # <

    # > Process each heatmap in the input folder
    # for heatmap_fname in os.listdir(INPUT_FOLDER):
    #     heatmap_relpath = os.path.join(INPUT_FOLDER, heatmap_fname)
    #     heatmap = np.load(heatmap_relpath)
        
    #     print("==================================================")
    #     print(f"Computing mean-vectors for {heatmap_fname}")
    #     stats, _, _ = compute_heatmap_statistics(heatmap, heatmap_relpath, roi_mean, "mean", separate_lr=True,
    #                                weigh_roi_overlap=False, debug=DEBUG, force_recalculate=True, dont_save=True, roi_type="faceparts")

    #     # Save the stats to the output folder
    #     if not os.path.exists(OUTPUT_FOLDER):
    #         os.makedirs(OUTPUT_FOLDER)
    #     output_path = os.path.join(OUTPUT_FOLDER, f"{heatmap_fname.split('.')[0]}_mean-vector.npy")
    #     np.save(output_path, stats, allow_pickle=True)
    # <

    # Or
    # > Process each heatmap and folder of heatmaps in the input folder
    things_in_input_folder = os.listdir(INPUT_FOLDER)

    # >> Process files first
    files = [thing for thing in things_in_input_folder if os.path.isfile(os.path.join(INPUT_FOLDER, thing))]
    files = [f for f in files if f.endswith('canonical.npy')] if CANONICAL_ONLY else [f for f in files if f.endswith('.npy')]

    for heatmap_fname in files:
        heatmap_relpath = os.path.join(INPUT_FOLDER, heatmap_fname)
        heatmap = np.load(heatmap_relpath)
        
        stats, _, _ = compute_heatmap_statistics(heatmap, heatmap_relpath, roi_mean, "mean", separate_lr=True,
                                weigh_roi_overlap=False, debug=DEBUG, force_recalculate=True, dont_save=True, roi_type="faceparts")

        # Save the stats to the output folder
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        output_path = os.path.join(OUTPUT_FOLDER, f"{heatmap_fname.split('.')[0]}_mean-vector.npy")
        np.save(output_path, stats, allow_pickle=True)
    # <<

    # >> Then process directories
    dirs = [thing for thing in things_in_input_folder if os.path.isdir(os.path.join(INPUT_FOLDER, thing))]

    for d in dirs:
        dirpath = os.path.join(INPUT_FOLDER, d)
        heatmap_fnames = os.listdir(dirpath)
        heatmap_fnames = [f for f in heatmap_fnames if f.endswith('canonical.npy')] if CANONICAL_ONLY else [f for f in heatmap_fnames if f.endswith('.npy')]

        for heatmap_fname in heatmap_fnames:
            heatmap_relpath = os.path.join(dirpath, heatmap_fname)
            heatmap = np.load(heatmap_relpath)
            
            stats, _, _ = compute_heatmap_statistics(heatmap, heatmap_relpath, roi_mean, "mean", separate_lr=True,
                                    weigh_roi_overlap=False, debug=DEBUG, force_recalculate=True, dont_save=True, roi_type="faceparts")

            # Save the stats to the output folder inside a folder named like d
            output_subfolder = os.path.join(OUTPUT_FOLDER, d)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            output_path = os.path.join(output_subfolder, f"{heatmap_fname.split('.')[0]}_mean-vector.npy")
            np.save(output_path, stats, allow_pickle=True)
    # <<
    # <

    # > Show the first output file you find
    # >> Try files first
    found = False
    for heatmap_fname in files:
        output_path = os.path.join(OUTPUT_FOLDER, f"{heatmap_fname.split('.')[0]}_mean-vector.npy")
        if os.path.exists(output_path):
            print(f"Example output file: {output_path}")
            example_stats = np.load(output_path, allow_pickle=True).item()
            print(example_stats)
            found = True
            break
    # <<

    # >> Then try directories
    if not found:
        for d in dirs:
            output_subfolder = os.path.join(OUTPUT_FOLDER, d)
            heatmap_fnames = os.listdir(os.path.join(INPUT_FOLDER, d))
            heatmap_fnames = [f for f in heatmap_fnames if f.endswith('canonical.npy')] if CANONICAL_ONLY else [f for f in heatmap_fnames if f.endswith('.npy')]

            for heatmap_fname in heatmap_fnames:
                output_path = os.path.join(output_subfolder, f"{heatmap_fname.split('.')[0]}_mean-vector.npy")
                if os.path.exists(output_path):
                    print(f"Example output file: {output_path}")
                    example_stats = np.load(output_path, allow_pickle=True).item()
                    print(example_stats)
                    found = True
                    break
            if found:
                break  
    # <<
    # <

    

