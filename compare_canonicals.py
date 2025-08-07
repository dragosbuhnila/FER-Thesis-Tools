# Dragos Buhnila 2025 ========================================================================================

import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pickle

from modules.basefaces import get_base_face
from modules.filenames_utils import get_emotion_from_heatmap_relpath, try_extract_model_or_user_name

from modules.roi_statistics import roi_mean, compare_meandif
from modules.landmark_utils import AU_LANDMARKS, FACE_PARTS_LANDMARKS, FACE_PARTS_LANDMARKS_LRMERGED, get_all_AUs, get_all_face_parts, get_all_face_parts_lrmerged
from modules.mask_n_heatmap_utils import compute_pixel_repetition_heatmap, get_roi_matrix, invert_heatmap
from modules.save_load_utils import load_statistics, save_statistics
from modules.visualize import make_comparison_grid, plot_matrix, show_grid_matplotlib, show_heatmaps

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

# >======================================================================================
# >=========== MACROS ===================================================================
# >======================================================================================

COMPARE_CHOSEN = compare_meandif    # e.g. choose between: compare_meandif, compare_hist
ROI_STAT_CHOSEN = roi_mean          # e.g. choose between: roi_mean, roi_hist
STAT_NAMES = {"roi_mean": "mean"}[ROI_STAT_CHOSEN.__name__] # e.g. "mean", "mean-std", "hist", etc.

DO_DEBUG = False  
DEBUG_ROIs = True  # If you use this, reweighting to see overlaps between ROIs will be enabled, as well as separate_lr for left/right distinctions, and forced recalculation of stats but with saving disabled
DEBUG_ROIs_SAVEONLY = False # If True, won't show the two plots (masked_heatmaps and repetition)
FORCE_RECALCULATE_STATS = False

ROI_DEBUG_FOLDER = "ROI_debug" if DEBUG_ROIs else None  

SALIENCY_MAPS_DIR = os.path.join(".", "saliency_maps")
HUMAN_RESULTS_DIR = os.path.join(SALIENCY_MAPS_DIR, "human_Results")

if DO_DEBUG == True:
    print(f"chosen macros are:")
    print(f"COMPARE_CHOSEN: {COMPARE_CHOSEN.__name__}")
    print(f"ROI_STAT_CHOSEN: {ROI_STAT_CHOSEN.__name__}")
    print(f"STAT_NAMES: {STAT_NAMES}")
    print(f"DO_DEBUG: {DO_DEBUG}")
    print(f"HUMAN_RESULTS_DIR: {HUMAN_RESULTS_DIR}")

# <======================================================================================
# <=========== END OF MACROS ============================================================
# <======================================================================================


# >======================================================================================
# >=========== GLOBAL VARIABLES =========================================================
# >======================================================================================

# 2) Load testers' ranking
testers_ranking_path = "./saliency_maps/human_Results/testers_ranking.pkl"
if os.path.exists(testers_ranking_path):
    with open(testers_ranking_path, "rb") as f:
        testers_ranking = pickle.load(f)
        testers_ranking = sorted(testers_ranking, key=lambda x: x[1], reverse=True)  # resort to be sure
        testers_ranking = [tester_name for tester_name, _ in testers_ranking]  # keep only names

# print(f"Loaded testers' ranking from {testers_ranking_path}. Found {len(testers_ranking)} testers:")
# for rank, tester in enumerate(testers_ranking, 1):
#     print(f"{rank:2d}. {tester}")


# <======================================================================================
# <=========== END OF GLOBAL VARIABLES =================================================
# <======================================================================================


# >======================================================================================
# >=========== FUNCTIONS ================================================================
# >======================================================================================

def compute_masked_heatmaps(heatmap, heatmap_fname, roi_type, emotion, debug=False, separate_lr=False):
    if roi_type.lower() == "aus":
        ROIs = get_all_AUs() 
        roi_landmarks = AU_LANDMARKS  
    elif roi_type.lower() == "faceparts":
        if separate_lr:
            ROIs = get_all_face_parts()
            roi_landmarks = FACE_PARTS_LANDMARKS
        else:
            ROIs = get_all_face_parts_lrmerged()
            roi_landmarks = FACE_PARTS_LANDMARKS_LRMERGED
    else:
        raise ValueError(f"Invalid roi_type: {roi_type}. Expected 'aus' or 'faceparts'.")

    # WARNING this is a hotfix (tapullissimo)
    if emotion.upper() == "ONES" or emotion.upper() == "ZEROES":
        emotion = "NEUTRAL"  # treat "ones" and "zeroes" as neutral for masking purposes

    # a BaseFace object contains: filename, image, shape, landmarks
    baseFace = get_base_face(emotion)
    all_landmarks = baseFace.landmarks
    base_face_shape = baseFace.shape

    # Create a dictionary to hold masked heatmaps for each AU: a masked heatmap is a saliency heatmap cut to only the region of interest (ROI) defined by the AU landmarks
    masked_heatmaps = {}
    for roi in ROIs:
        # Collect all masked heatmaps for this ROI
        roi_masks = []

        # this extra step is  needed bc I coded AUs that are lists of lists for organizing
        for landmark_set in roi_landmarks[roi]: 
            landmark_coordinates = []
            for landmark_idx in landmark_set:
                landmark_coordinates.append(all_landmarks[landmark_idx])

            is_closed_loop = True if landmark_set[0] != landmark_set[-1] else False
            roi_matrix = get_roi_matrix(base_face_shape, landmark_coordinates, fill=is_closed_loop, expansion=33, debug=debug)
            roi_matrix = cv2.resize(roi_matrix, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_LINEAR)
            roi_matrix[roi_matrix > 0] = 1  # Ensure binary mask
            roi_matrix = roi_matrix.astype(float)
            roi_matrix[roi_matrix != 1] = np.nan

            masked_heatmap = heatmap * roi_matrix
            roi_masks.append(masked_heatmap)

            # old debug
            # if debug:
            #     resized_baseface = cv2.resize(baseFace.image, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_LINEAR)
            #     plot_matrix(masked_heatmap, title=f"Masked Heatmap for {heatmap_fname} and ROI {roi}", background=resized_baseface)

        # Merge all masks for this ROI (pixelwise max, so any covered pixel is included)
        if len(roi_masks) == 1:
            masked_heatmaps[roi] = roi_masks[0]
        else:
            if separate_lr:
                if len(roi_masks) != 2:
                    raise ValueError(f"Expected 2 masks for separate left/right processing, got {len(roi_masks)}")
                masked_heatmaps[f"{roi}_left"] = roi_masks[0]
                masked_heatmaps[f"{roi}_right"] = roi_masks[1]
            else:
                stacked = np.stack(roi_masks)
                merged_masked_heatmap = np.nanmax(stacked, axis=0)
                masked_heatmaps[roi] = merged_masked_heatmap

    if debug:
        print(f"Computed masked heatmaps for {heatmap_fname} with {len(masked_heatmaps)} ROIs (separate_lr={separate_lr} and roi_type={roi_type}).")
        resized_baseface = cv2.resize(baseFace.image, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_LINEAR)
        # for roi, masked_heatmap in masked_heatmaps.items():
            # plot_name = f"{heatmap_fname} - {roi}".replace("/", "_")
            # plot_matrix(masked_heatmap, title=plot_name, background=resized_baseface, save_folder=ROI_DEBUG_FOLDER, save_only=DEBUG_ROIs_SAVEONLY)
        
        stacked_roi_masks = np.stack(list(masked_heatmaps.values()))
        merged_masked_heatmap = np.nanmax(stacked_roi_masks, axis=0)
        plot_name = f"{heatmap_fname} - Merged Masks With Heatmap".replace("/", "_")
        plot_matrix(merged_masked_heatmap, title=plot_name, background=resized_baseface, save_folder=ROI_DEBUG_FOLDER, save_only=DEBUG_ROIs_SAVEONLY)

    return masked_heatmaps

def compute_heatmap_statistics(heatmap, heatmap_relpath, compute_stat, stat_names,
                               weigh_roi_overlap: bool, debug=False, force_recalculate=False, separate_lr=False, roi_type="aus"):
    """
    Computes statistics for a given heatmap.
    Args:
        heatmap (np.ndarray): The heatmap to compute statistics for.
        heatmap_relpath (str): The relative path of the heatmap file.
            Example: "saliency_maps/results/subject/heatmaps/heatmap_1.npy"
        compute_stat (function): The function to compute the statistic.
            Example: roi_mean, roi_mean_std, roi_hist, etc.
        debug (bool): If True, will print debug information and plot the heatmap.
        stat_names (str): A string representing the names of the statistics to compute.
            Example: "mean", "mean-std", "hist", etc.
    Returns:
        tuple: A tuple containing:
            - statistics (dict): The computed statistics for each AU.
                Example: {"mean": 0.1234} or {"mean": 0.1234, "std": 0.5678} or {"AU1": {"hist": [0.1, 0.2, 0.3], "mean": 0.1234}, ...}
            - subject (str): The subject name extracted from the heatmap path.
                Example: "AGAPIG"
            - emotion (str): The emotion extracted from the heatmap filename.
                Example: "DISGUST"
    Raises:
        ValueError: If the heatmap is not a 2D array or if the heatmap filename does not follow the expected format.
    """
    subject = try_extract_model_or_user_name(heatmap_relpath)
    emotion = get_emotion_from_heatmap_relpath(heatmap_relpath)

    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap {subject}/{emotion} is not a 2D array. Please check the file format.")

    if DEBUG_ROIs:
        separate_lr = False
        weigh_roi_overlap = True 
        force_recalculate = True

    # 0) Try to load stat if already cached
    if not force_recalculate:
        cached_statistics = load_statistics(heatmap_relpath, stat_names=stat_names, weigh_roi_overlap=weigh_roi_overlap, separate_lr=separate_lr, roi_type=roi_type)
        if cached_statistics is not None:
            print(f"Cached stats loaded for {subject}/{emotion}, weigh_roi_overlap={weigh_roi_overlap}, separate_lr={separate_lr}, roi_type={roi_type}.")
            return cached_statistics, subject, emotion
        else:
            print(f"Cached stats unavailable for {subject}/{emotion}, weigh_roi_overlap={weigh_roi_overlap}, separate_lr={separate_lr}, roi_type={roi_type}.")
    else:
        print(f"Cached stats ignored, forced recalculation for {subject}/{emotion}, weigh_roi_overlap={weigh_roi_overlap}, separate_lr={separate_lr}, roi_type={roi_type}.")

    if debug:
        # print(f"Shape of heatmap {subject}/{emotion}: {heatmap.shape}")
        plot_matrix(heatmap, title=f"Heatmap for {subject}/{emotion}")

    # 1) Compute the masked heatmaps ROI by ROI
    masked_heatmaps = compute_masked_heatmaps(heatmap, f"{subject}/{emotion}", roi_type, emotion, debug=DEBUG_ROIs, separate_lr=separate_lr)
    print(f"Computed masked heatmaps for {subject}/{emotion} with {len(masked_heatmaps)} AUs (separate_lr={separate_lr}).")

    # 1.5) Based on the ROIs that we used, there may be some pixel repetition across AUs.
    if weigh_roi_overlap:
        repetition_map, amt_ovlp, amt_tot = compute_pixel_repetition_heatmap(masked_heatmaps, debug=False)
        if DEBUG_ROIs:
            baseface = get_base_face(emotion)
            resized_baseface = cv2.resize(baseface.image, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_LINEAR)
            plot_matrix(repetition_map, title=f"{subject}_{emotion} Pixel Repetition Heatmap ({amt_ovlp} of {amt_tot} overlapping)", background=resized_baseface, alpha=0.5, save_folder=ROI_DEBUG_FOLDER, save_only=DEBUG_ROIs_SAVEONLY)

        weightmap = invert_heatmap(repetition_map)  # Invert the heatmap to use it as a weightmap
        if debug:
            plot_matrix(weightmap, title=f"Weightmap for {subject}/{emotion}", background=resized_baseface, alpha=0.5)

        print(f"Pixel repetition heatmap not skipped for {subject}/{emotion}.")
    else:
        print(f"Pixel repetition heatmap skipped for {subject}/{emotion}.")
        weightmap = None

    # 2) Compute statistic
    statistics = {}
    for au, masked_heatmap in masked_heatmaps.items():
        stat = compute_stat(masked_heatmap, pxbypx_weightmap=weightmap, debug=debug)
        statistics[au] = stat
        if debug:
            print(f"Computed {stat_names} for AU {au}: {stat}")

    # 3) Cache the statistics (if debugging ROIs don't save as it may fuck up the stats size, since when debuggin I plot l/r ROIs together, while we usually don't do that with faceparts)
    if not DEBUG_ROIs:
        save_statistics(heatmap_relpath, statistics, weigh_roi_overlap=weigh_roi_overlap, separate_lr=separate_lr, roi_type=roi_type)

    return statistics, subject, emotion

def do_group_comparison(heatmaps_relpaths, debug=False, save_only=False, force_recalculate_stats=False):
    # 1) Compute statistics for all heatmaps
    unique_heatmap_names = []   
    heatmaps = {}
    # stats_list_raw = {}
    # stats_list_pxwtd = {}
    # stats_list_sep_lr = {}
    stats_list_faceparts = {}

    for heatmap_relpath in heatmaps_relpaths:
        heatmap = np.load(heatmap_relpath)

        # compute both raw and pixel-weighted statistics, so you can plot both for comparison
        # - pxwtd means that the statistics are weighted by the pixel repetition across AUs

        # stats_raw, _, _     = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=False, debug=debug, force_recalculate=force_recalculate_stats, separate_lr=False, roi_type="aus")
        # stats_pxwtd, _, _   = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=True, debug=debug, force_recalculate=force_recalculate_stats, separate_lr=False, roi_type="aus")
        # stats_sep_lr, _, _  = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=False, debug=debug, force_recalculate=force_recalculate_stats, separate_lr=True, roi_type="aus")
        stats_faceparts, subject, emotion = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=False, debug=debug, force_recalculate=force_recalculate_stats, separate_lr=False, roi_type="faceparts")
        unique_heatmap_name = f"{subject}/{emotion}"
        
        unique_heatmap_names.append(unique_heatmap_name)
        heatmaps[unique_heatmap_name] = heatmap

        # stats_list_raw[unique_heatmap_name] = stats_raw
        # stats_list_pxwtd[unique_heatmap_name] = stats_pxwtd
        # stats_list_sep_lr[unique_heatmap_name] = stats_sep_lr
        stats_list_faceparts[unique_heatmap_name] = stats_faceparts

    # 2) Make the grids
    # grid_raw        = make_comparison_grid(unique_heatmap_names, stats_list_raw,   COMPARE_CHOSEN)
    # grid_pxwtd      = make_comparison_grid(unique_heatmap_names, stats_list_pxwtd, COMPARE_CHOSEN)
    # grid_sep_lr     = make_comparison_grid(unique_heatmap_names, stats_list_sep_lr, COMPARE_CHOSEN)
    grid_faceparts  = make_comparison_grid(unique_heatmap_names, stats_list_faceparts, COMPARE_CHOSEN)

    # # 3) Calculate if there's any differences
    # some_difference = False
    # for i in unique_heatmap_names:
    #     for j in unique_heatmap_names:
    #         try:
    #             val_pxwtd = float(grid_pxwtd.loc[i, j])
    #             val_raw = float(grid_raw.loc[i, j])
    #             diff = abs(val_pxwtd - val_raw)
    #             if diff >= 0.0001:
    #                 # print(f"Difference between {i} and {j}: {diff:.4f} (pxwtd: {val_pxwtd:.4f}, raw: {val_raw:.4f})")
    #                 some_difference = True
    #             # else:
    #                 # print(f"Difference between {i} and {j} is negligible: {diff:.4f} (pxwtd: {val_pxwtd:.4f}, raw: {val_raw:.4f})")
    #         except Exception:
    #             diff = "NaN"
    # if not some_difference:
    #     print("No significant differences found between pixel-weighted and raw statistics.")    

    # 4) Show the grids and heatmaps
    if save_only:
        print(f"Not displaying plt for heatmaps: {', '.join(unique_heatmap_names)}")
    else:
        show_heatmaps(heatmaps)

        # show_grid_matplotlib(grid_pxwtd, title="Stat Comparison: AUs-reweighted, symmetric", cmap="viridis")
        # show_grid_matplotlib(grid_raw, title="Stat Comparison: AUs-not-reweighted, symmetric", cmap="viridis")
        # show_grid_matplotlib(grid_sep_lr, title="Stat Comparison: AUs-not-reweighted, l/r", cmap="viridis")
        show_grid_matplotlib(grid_faceparts, title="Stat Comparison: Faceparts, l/r", cmap="viridis")

        plt.show()
        

# <======================================================================================
# <=========== END OF FUNCTIONS =========================================================
# <======================================================================================

if __name__ == "__main__":
    testers = [
        d for d in os.listdir(HUMAN_RESULTS_DIR)
        if os.path.isdir(os.path.join(HUMAN_RESULTS_DIR, d))
    ]

    EMOTIONS = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]


    while True:
        print("\n=== Main Menu ===")
        print("0) Test metric between a full-blue and a full-red heatmap")
        print("1) Single Person")
        print("2) Top%, and Emotion")
        print("3) Top% + Gender, and Emotion (not implemented)")
        print("8) Recalculate statistics for all heatmaps")
        print("9) Quit")
        choice = input("Enter choice: ").strip()

        if choice == "0":
            heatmaps_relpaths = [
                os.path.join(SALIENCY_MAPS_DIR, "std_saliency_map_ones.npy"),
                os.path.join(SALIENCY_MAPS_DIR, "std_saliency_map_zeroes.npy")
            ]

            for heatmap_relpath in heatmaps_relpaths:
                if not os.path.isfile(heatmap_relpath):
                    print(f"  >> Error: heatmap file '{heatmap_relpath}' not found.")
                    continue

            print(f"Comparing heatmaps: {heatmaps_relpaths[0]} and {heatmaps_relpaths[1]}")
            do_group_comparison(heatmaps_relpaths, debug=DO_DEBUG, force_recalculate_stats=True)
            continue
        
        elif choice == "1":
            print(f"Available IDs: {', '.join(testers)}")
            tester = input("Enter Person ID: ").strip().upper()

            if tester not in testers:
                print(f"  >> Error: '{tester}' not found. Try again.")
                continue

            heatmaps_path = os.path.join(HUMAN_RESULTS_DIR, tester, "heatmaps")
            if not os.path.isdir(heatmaps_path):
                print(f"  >> Error: heatmaps folder '{heatmaps_path}' missing.")
                continue

            heatmap_fnames = [
                f for f in os.listdir(heatmaps_path)
                if f.endswith("canonical.npy")
            ]
            if not heatmap_fnames:
                print(f"  >> No canonical.npy files in '{heatmaps_path}'.")
                continue

            heatmaps_relpaths = [os.path.join(heatmaps_path, fname) for fname in heatmap_fnames]
            print(f"heatmaps_relpaths: {heatmaps_relpaths}")
            do_group_comparison(heatmaps_relpaths, debug=DO_DEBUG, force_recalculate_stats=FORCE_RECALCULATE_STATS)

        elif choice == "2":
            # Select emotion
            print("Choose emotion between ", ", ".join(EMOTIONS))
            selected_emotion = input("Enter Emotion: ").strip().upper()
            if selected_emotion not in EMOTIONS:
                print(f"  >> Error: '{selected_emotion}' not found. Try again.")
                continue

            # Select range of top performers
            print(f"Choose a range of top performers for {selected_emotion} (e.g., 0-10 for top 10%, 5-20 for from 5% to 20%):")
            range_input = input("Enter range (e.g., 0-10 or 15-50): ").strip()
            topX_performers = []
            try:
                start, end = map(int, range_input.split("-"))
                if start < 0 or end < 0 or start > end:
                    raise ValueError

                start = int(start / 100 * len(testers_ranking))
                end = int(np.ceil(end / 100 * len(testers_ranking)))
                if end == 0:
                    print(f"  >> Error: Range ending cannot be zero (start={start}, end={end}). Try again.")
                    continue
                
                topX_performers = testers_ranking[start:end]
            except ValueError:
                print(f"  >> Error: Invalid range '{range_input}'. Try again.")
                continue
            print(f"Selected performers {start} to {end} for {selected_emotion}: {', '.join(topX_performers)}")

            # Extract paths and fnames for heatmaps
            heatmaps_paths = []
            heatmap_fnames = []
            for tester in topX_performers:
                heatmaps_path = os.path.join(HUMAN_RESULTS_DIR, tester, "heatmaps")
                if not os.path.isdir(heatmaps_path):    
                    print(f"  >> Error: heatmaps folder '{heatmaps_path}' missing for tester '{tester}'.")
                    continue

                heatmaps_paths.append(heatmaps_path)
                heatmap_fnames.append(f"{selected_emotion}_canonical.npy")
                if not os.path.isfile(os.path.join(heatmaps_path, heatmap_fnames[-1])):
                    print(f"  >> Error: '{heatmap_fnames[-1]}' not found in '{heatmaps_path}' for tester '{tester}'.")
                    heatmaps_paths.pop()
                    heatmap_fnames.pop()

            heatmaps_relpaths = [os.path.join(path, fname) for path, fname in zip(heatmaps_paths, heatmap_fnames)]
            print(f"heatmaps_relpaths: {heatmaps_relpaths}")
            do_group_comparison(heatmaps_relpaths, debug=DO_DEBUG, force_recalculate_stats=FORCE_RECALCULATE_STATS)

        elif choice == "3":
            print("  >> Option not implemented yet.")
            continue

        elif choice == "8":
            for tester in testers:
                heatmaps_path = os.path.join(HUMAN_RESULTS_DIR, tester, "heatmaps")

                if not os.path.isdir(heatmaps_path):
                    print(f"  >> Error: heatmaps folder '{heatmaps_path}' missing.")
                    continue

                heatmap_fnames = [
                    f for f in os.listdir(heatmaps_path)
                    if f.endswith("canonical.npy")
                ]
                if not heatmap_fnames:
                    print(f"  >> No canonical.npy files in '{heatmaps_path}'.")
                    continue

                heatmaps_relpaths = [os.path.join(heatmaps_path, fname) for fname in heatmap_fnames]
                print(f"heatmaps_relpaths: {heatmaps_relpaths}")
                do_group_comparison(heatmaps_relpaths, debug=DO_DEBUG, save_only=True, force_recalculate_stats=True)

        elif choice == "9":
            print("Goodbye!")
            break

        else:
            print("  >> Invalid choice.")