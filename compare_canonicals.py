# Dragos Buhnila 2025 ========================================================================================

import itertools
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd
import pickle

from modules.basefaces import get_base_face
from modules.filenames_utils import get_emotion_from_heatmap_relpath, try_extract_model_or_user_name

from modules.roi_statistics import roi_mean, compare_meandif
from modules.landmark_utils import LANDMARKS, detect_facial_landmarks, EMOTION_AUS, get_all_AUs, load_landmark_coordinates, save_landmark_coordinates
from modules.mask_utils import compute_mask_alpha, get_roi_matrix
from modules.visualize import plot_matrix, print_aubyau_stats, show_grid_matplotlib, show_grid_tkinter, show_heatmaps


# >======================================================================================
# >=========== MACROS ===================================================================
# >======================================================================================

COMPARE_CHOSEN = compare_meandif    # e.g. choose between: compare_meandif, compare_hist
ROI_STAT_CHOSEN = roi_mean          # e.g. choose between: roi_mean, roi_hist
STAT_NAMES = {"roi_mean": "mean"}[ROI_STAT_CHOSEN.__name__] # e.g. "mean", "mean-std", "hist", etc.

DO_DEBUG = False  # Set to True to enable debug mode
DEEP_DEBUG = False  # Set to True to enable deep debug mode (e.g. plotting masked heatmaps)
FORCE_RECALCULATE_STATS = False

SALIENCY_MAPS_DIR = os.path.join(".", "saliency_maps")
HUMAN_RESULTS_DIR = os.path.join(SALIENCY_MAPS_DIR, "human_Results")

RUN_ON_ALL_TESTERS = False  # Set to False to skip the menu and run directly
SAVE_ONLY = False  # Set to True to save results without displaying plots
CHOICE = "1"  # Default choice if USE_MENU is False

if DO_DEBUG == True:
    print(f"chosen macros are:")
    print(f"COMPARE_CHOSEN: {COMPARE_CHOSEN.__name__}")
    print(f"ROI_STAT_CHOSEN: {ROI_STAT_CHOSEN.__name__}")
    print(f"STAT_NAMES: {STAT_NAMES}")
    print(f"DO_DEBUG: {DO_DEBUG}")
    print(f"SAVE_ONLY: {SAVE_ONLY}")
    print(f"HUMAN_RESULTS_DIR: {HUMAN_RESULTS_DIR}")
    print(f"RUN_ON_ALL_TESTERS: {RUN_ON_ALL_TESTERS}")
    print(f"CHOICE: {CHOICE}")

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

def compute_masked_heatmaps(heatmap, heatmap_fname, AUs, emotion, debug=False):
    if emotion.upper() == "ONES" or emotion.upper() == "ZEROES":
        emotion = "NEUTRAL"  # treat "ones" and "zeroes" as neutral for masking purposes
    baseFace = get_base_face(emotion)
    all_landmarks = baseFace.landmarks
    base_face_shape = baseFace.shape
    print(f"Computing masked heatmaps for {heatmap_fname} with emotion {emotion} and base face shape {base_face_shape}")

    masked_heatmaps = {}
    for au in AUs:
        for landmark_set in LANDMARKS[au]: # extra step needed bc I coded AUs are lists of lists for organizing
            landmark_coordinates = []
            for landmark_idx in landmark_set:
                landmark_coordinates.append(all_landmarks[landmark_idx])

            is_closed_loop = True if landmark_set[0] != landmark_set[-1] else False
            roi_matrix = get_roi_matrix(base_face_shape, landmark_coordinates, fill=is_closed_loop, debug=debug)
            roi_matrix = cv2.resize(roi_matrix, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_LINEAR)
            roi_matrix[roi_matrix > 0] = 1  # Ensure binary mask
            roi_matrix = roi_matrix.astype(float)
            roi_matrix[roi_matrix != 1] = np.nan

            masked_heatmap = heatmap * roi_matrix
            masked_heatmaps[au] = masked_heatmap

            if debug:
                plot_matrix(masked_heatmap, title=f"Masked Heatmap for {heatmap_fname} and AU {au}")

    return masked_heatmaps

def get_stats_cache_path(heatmap_relpath, stat_names: str):
    """
    Generates a path for caching statistics of a heatmap.
    Args:
        heatmap_relpath (str): The relative path of the heatmap file.
            Example: "saliency_maps/results/subject/heatmaps/heatmap_1.npy"
        stat_names (str): A string representing the names of the statistics to cache.
            Example: "mean", "mean-std", "hist", etc.
    Returns:
        str: The path where the statistics will be cached.
    """
    heatmap_fname = os.path.basename(heatmap_relpath)
    folder_path = os.path.dirname(heatmap_relpath)

    cache_dir = os.path.join(folder_path, "stats_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{heatmap_fname}_stat-{stat_names}.pkl")

def save_statistics(heatmap_relpath, statistics):
    stat_names = [stat_name for stat_name in next(iter(statistics.values())).keys()]
    print(f"stat_names are {stat_names}")
    path = get_stats_cache_path(heatmap_relpath, "-".join(stat_names))
    with open(path, "wb") as f:
        pickle.dump(statistics, f)

def load_statistics(heatmap_relpath, stat_names):
    path = get_stats_cache_path(heatmap_relpath, stat_names)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def compute_heatmap_statistics(heatmap, heatmap_relpath, compute_stat, stat_names, debug=False, force_recalculate=False):
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
    print(f"debug is {debug}, ")

    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap {subject}/{emotion} is not a 2D array. Please check the file format.")

    # 0) Try to load stat if already cached
    if not force_recalculate:
        cached_statistics = load_statistics(heatmap_relpath, stat_names=stat_names)
    else:
        cached_statistics = None
    if cached_statistics is not None:
        print(f"Loaded cached statistics for {subject}/{emotion}")
        return cached_statistics, subject, emotion
    print(f"Cache unavailable: computing statistics for heatmap {subject}/{emotion}.")

    if debug:
        print(f"Shape of heatmap {subject}/{emotion}: {heatmap.shape}")
        plot_matrix(heatmap, title=f"Heatmap for {subject}/{emotion}")

    # 1) Compute the masked heatmaps ROI by ROI
    masked_heatmaps = compute_masked_heatmaps(heatmap, f"{subject}/{emotion}", get_all_AUs(), emotion, debug=DEEP_DEBUG)
    print(f"Computed masked heatmaps for {subject}/{emotion} with {len(masked_heatmaps)} AUs.")

    # 2) Compute statistic
    statistics = {}
    for au, masked_heatmap in masked_heatmaps.items():
        # save the masked heatmap in the masked_heatmaps/subject/emotion/ folder
        masked_heatmap_path = os.path.join(os.path.dirname(heatmap_relpath), "masked_heatmaps", subject, emotion, f"{au}_masked.npy")
        os.makedirs(os.path.dirname(masked_heatmap_path), exist_ok=True)
        np.save(masked_heatmap_path, masked_heatmap)
        print(f"Saved masked heatmap for AU {au} at {masked_heatmap_path}")

        stat = compute_stat(masked_heatmap)
        statistics[au] = stat
        print(f"Computed {stat_names} for AU {au}: {stat}")

    # 3) Cache the statistics
    save_statistics(heatmap_relpath, statistics)

    return statistics, subject, emotion

def do_group_comparison(heatmaps_relpaths, debug=False, save_only=False, force_recalculate_stats=False):
    # Compute statistics for all heatmaps
    unique_heatmap_names = []   
    heatmaps = {}
    stats_list = {}
    for heatmap_relpath in heatmaps_relpaths:
        heatmap = np.load(heatmap_relpath)
        statistics, subject, emotion = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, debug=debug, force_recalculate=force_recalculate_stats)
        unique_heatmap_name = f"{subject}/{emotion}"
        
        unique_heatmap_names.append(unique_heatmap_name)
        heatmaps[unique_heatmap_name] = heatmap
        stats_list[unique_heatmap_name] = statistics

    # Create a grid using pandas DataFrame
    grid = pd.DataFrame(index=unique_heatmap_names, columns=unique_heatmap_names)

    for heatmap_name1, heatmap_name2 in itertools.combinations(unique_heatmap_names, 2):
        result = COMPARE_CHOSEN(stats_list[heatmap_name1], stats_list[heatmap_name2])
        # If result is a dict, get the first value
        if isinstance(result, dict):
            val = next(iter(result.values()))
        else:
            val = result
        grid.loc[heatmap_name1, heatmap_name2] = f"{val:.4f}"  # Fill the asymmetric cell
        grid.loc[heatmap_name2, heatmap_name1] = f"{val:.4f}"  # Fill the symmetric cell

    # Fill diagonal with "-"
    for heatmap_name in unique_heatmap_names:
        result = COMPARE_CHOSEN(stats_list[heatmap_name], stats_list[heatmap_name])
        # If result is a dict, get the first value
        if isinstance(result, dict):
            val = next(iter(result.values()))
        else:
            val = result
        grid.loc[heatmap_name, heatmap_name] = f"{val:.4f}"

    if save_only:
        print(f"Not displaying plt for heatmaps: {', '.join(unique_heatmap_names)}")
        return grid
    else:
        show_heatmaps(heatmaps)
        show_grid_matplotlib(grid)
        plt.show()
        return grid


# <======================================================================================
# <=========== END OF FUNCTIONS =========================================================
# <======================================================================================

if __name__ == "__main__":
    testers = [
        d for d in os.listdir(HUMAN_RESULTS_DIR)
        if os.path.isdir(os.path.join(HUMAN_RESULTS_DIR, d))
    ]

    EMOTIONS = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]


    for tester in testers:
        if not RUN_ON_ALL_TESTERS:
            print("\n=== Main Menu ===")
            print("0) Test metric between a full-blue and a full-red heatmap")
            print("1) Single Person")
            print("2) Top%, and Emotion")
            print("3) Top% + Gender, and Emotion (not implemented)")
            print("4) Quit")
            choice = input("Enter choice [1-4]: ").strip()
        else:
            choice = CHOICE

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
            if not RUN_ON_ALL_TESTERS:
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
            do_group_comparison(heatmaps_relpaths, debug=DO_DEBUG, save_only=SAVE_ONLY, force_recalculate_stats=FORCE_RECALCULATE_STATS)

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
            do_group_comparison(heatmaps_relpaths, debug=DO_DEBUG, save_only=SAVE_ONLY, force_recalculate_stats=FORCE_RECALCULATE_STATS)

        elif choice == "3":
            print("  >> Option not implemented yet.")
            continue

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("  >> Invalid choice, please enter 1-4.")