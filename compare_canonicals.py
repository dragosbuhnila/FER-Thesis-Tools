# Dragos Buhnila 2025 ========================================================================================

import itertools
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd
import pickle

from modules.models import BASEFACES_FOLDER, BaseFace
from modules.roi_statistics import roi_mean, compare_meandif
from modules.landmark_utils import LANDMARKS, detect_facial_landmarks, EMOTION_AUS, get_all_AUs, load_landmark_coordinates, save_landmark_coordinates
from modules.mask_utils import compute_mask_alpha, get_roi_matrix
from modules.visualize import plot_matrix, print_aubyau_stats, show_grid_matplotlib, show_grid_tkinter, show_heatmaps


# >======================================================================================
# >=========== MACROS ===================================================================
# >======================================================================================

# choose between: roi_mean, roi_hist
ROI_STAT_CHOSEN = roi_mean 
COMPARE_CHOSEN = compare_meandif

DO_DEBUG = True  # Set to True to enable debug mode
SAVE_ONLY = False  # Set to True to save results without displaying plots

# <======================================================================================
# <=========== END OF MACROS ============================================================
# <======================================================================================


# >======================================================================================
# >=========== GLOBAL VARIABLES =========================================================
# >======================================================================================

# 1) Load basefaces
baseface_fnames = [base_face_fname for base_face_fname in os.listdir(BASEFACES_FOLDER) if base_face_fname.endswith('reshaped.png')]
if len(baseface_fnames) != 7:
    raise ValueError(f"Expected 7 basefaces, found {len(baseface_fnames)}. Please check the basefaces directory.")

basefaces = {}
for baseface_fname in baseface_fnames:
    selected_emotion = baseface_fname.split('_')[1].upper()  # Assuming the filename format is like "baseface_emotion_reshaped.png"
    basefaces[selected_emotion] = BaseFace(baseface_fname)
    # print(f"Loaded base face for emotion {emotion}. {basefaces[emotion]}")

# 2) Load testers' ranking
testers_ranking_path = "./saliency_maps/human_Results/testers_ranking.pkl"
if os.path.exists(testers_ranking_path):
    with open(testers_ranking_path, "rb") as f:
        testers_ranking = pickle.load(f)
        testers_ranking = sorted(testers_ranking, key=lambda x: x[1], reverse=True)  # resort to be sure
        testers_ranking = [tester_name for tester_name, _ in testers_ranking]  # keep only names

print(f"Loaded testers' ranking from {testers_ranking_path}. Found {len(testers_ranking)} testers:")
for rank, tester in enumerate(testers_ranking, 1):
    print(f"{rank:2d}. {tester}")


# <======================================================================================
# <=========== END OF GLOBAL VARIABLES =================================================
# <======================================================================================


# >======================================================================================
# >=========== FUNCTIONS ================================================================
# >======================================================================================

def get_base_face(emotion):
    if emotion.upper() not in basefaces.keys():
        raise ValueError(f"Emotion {emotion} does not have a corresponding base face.")
    return basefaces[emotion.upper()]

def get_emotion_from_heatmap_fname(heatmap_fname):
    """
    Extracts the emotion from the heatmap filename.
    For now, assumes the filename format is like "DISGUST_Neutral.npy" or "DISGUST_canonical.npy".
    """
    parts = heatmap_fname.split('_')
    if len(parts) != 2:
        raise ValueError(f"Heatmap filename {heatmap_fname} does not follow the expected format.")
    if parts[0].upper() not in ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]:
        raise ValueError(f"Unknown emotion {parts[0]} in heatmap filename {heatmap_fname}.")
    return parts[0]

def compute_masked_heatmaps(heatmap, heatmap_fname, AUs, emotion, debug=False):
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

            masked_heatmap = heatmap * roi_matrix
            masked_heatmaps[au] = masked_heatmap

            if debug:
                plot_matrix(masked_heatmap, title=f"Masked Heatmap for {heatmap_fname} and AU {au}")

    return masked_heatmaps

def get_stats_cache_path(heatmap_fname, folder_path, stat_names: str):
    cache_dir = os.path.join(folder_path, "stats_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{heatmap_fname}_stat-{stat_names}.pkl")

def save_statistics(heatmap_fname, folder_path, statistics):
    stat_names = [stat_name for stat_name in next(iter(statistics.values())).keys()]
    print(f"stat_names are {stat_names}")
    path = get_stats_cache_path(heatmap_fname, folder_path, "-".join(stat_names))
    with open(path, "wb") as f:
        pickle.dump(statistics, f)

def load_statistics(heatmap_fname, folder_path, stat_names: str = "mean"):
    path = get_stats_cache_path(heatmap_fname, folder_path, stat_names)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def compute_heatmap_statistics(heatmap, heatmap_fname, folder_path, compute_stat, debug=False):
    fname_long = os.path.join(folder_path.split("/")[-1].split("\\")[1], heatmap_fname)
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap {fname_long} is not a 2D array. Please check the file format.")

    # 0) Try to load stat if already cached
    cached = load_statistics(heatmap_fname, folder_path)
    if cached is not None:
        print(f"Loaded cached statistics for {fname_long}")
        return cached

    if debug:
        print(f"Shape of heatmap {fname_long}: {heatmap.shape}")
        plot_matrix(heatmap, title=f"Heatmap for {fname_long}")

    # 1) Compute the masked heatmaps ROI by ROI
    emotion = get_emotion_from_heatmap_fname(heatmap_fname)
    masked_heatmaps = compute_masked_heatmaps(heatmap, fname_long, get_all_AUs(), emotion, debug=debug)

    # 2) Compute statistic
    statistics = {}
    for au, masked_heatmap in masked_heatmaps.items():
        stat = compute_stat(masked_heatmap)
        statistics[au] = stat

    # 3) Cache the statistics
    save_statistics(heatmap_fname, folder_path, statistics)

    return statistics

def do_group_comparison(heatmaps_folder_paths, heatmap_fnames, debug=False, save_only=False):
    fnames_long = [os.path.join(folder_path.split("/")[-1].split("\\")[1], fname) for folder_path, fname in zip(heatmaps_folder_paths, heatmap_fnames)]
    fnames_full = [os.path.join(folder_path, fname) for folder_path, fname in zip(heatmaps_folder_paths, heatmap_fnames)]

    print(f"fnames_long: {fnames_long}")
    print(f"fnames_full: {fnames_full}")

    heatmaps = {fname_long: np.load(fname_full) for fname_long, fname_full in zip(fnames_long, fnames_full)}

    # Compute statistics for all heatmaps
    stats_list = {fname_long: compute_heatmap_statistics(heatmaps[fname_long], fname, folder_path, ROI_STAT_CHOSEN, debug=debug) for fname, folder_path, fname_long in zip(heatmap_fnames, heatmaps_folder_paths, fnames_long)}

    # Create a grid using pandas DataFrame
    grid = pd.DataFrame(index=fnames_long, columns=fnames_long)

    for fname1, fname2 in itertools.combinations(fnames_long, 2):
        result = COMPARE_CHOSEN(stats_list[fname1], stats_list[fname2])
        # If result is a dict, get the first value
        if isinstance(result, dict):
            val = next(iter(result.values()))
        else:
            val = result
        grid.loc[fname1, fname2] = f"{val:.4f}"  # Fill the asymmetric cell
        grid.loc[fname2, fname1] = f"{val:.4f}"  # Fill the symmetric cell

    # Fill diagonal with "-"
    for fname in heatmap_fnames:
        result = COMPARE_CHOSEN(stats_list[fname1], stats_list[fname2])
        # If result is a dict, get the first value
        if isinstance(result, dict):
            val = next(iter(result.values()))
        else:
            val = result
        grid.loc[fname, fname] = f"{val:.4f}"

    if save_only:
        print(f"Not displaying plt for heatmaps: {', '.join(heatmap_fnames)}")
        return grid
    else:
        show_heatmaps(heatmaps)
        show_grid_matplotlib(grid)
        plt.show()
        return grid


# <======================================================================================
# <=========== END OF FUNCTIONS =========================================================
# <======================================================================================

USE_MENU = True  # Set to False to skip the menu and run directly
CHOICE = "1"  # Default choice if USE_MENU is False

if __name__ == "__main__":
    human_results_dir = "./saliency_maps/human_Results"

    testers = [
        d for d in os.listdir(human_results_dir)
        if os.path.isdir(os.path.join(human_results_dir, d))
    ]

    EMOTIONS = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]


    for tester in testers:
        if USE_MENU:
            print("\n=== Main Menu ===")
            print("1) Single Person")
            print("2) Top%, and Emotion")
            print("3) Top% + Gender, and Emotion (not implemented)")
            print("4) Quit")
            choice = input("Enter choice [1-4]: ").strip()
        else:
            choice = CHOICE

        if choice == "1":
            if USE_MENU:
                print(f"Available IDs: {', '.join(testers)}")
                tester = input("Enter Person ID: ").strip().upper()

            if tester not in testers:
                print(f"  >> Error: '{tester}' not found. Try again.")
                continue

            heatmaps_path = os.path.join(human_results_dir, tester, "heatmaps")
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

            do_group_comparison([heatmaps_path]*len(heatmap_fnames), heatmap_fnames, debug=DO_DEBUG, save_only=SAVE_ONLY)

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
                heatmaps_path = os.path.join(human_results_dir, tester, "heatmaps")
                if not os.path.isdir(heatmaps_path):    
                    print(f"  >> Error: heatmaps folder '{heatmaps_path}' missing for tester '{tester}'.")
                    continue

                heatmaps_paths.append(heatmaps_path)
                heatmap_fnames.append(f"{selected_emotion}_canonical.npy")
                if not os.path.isfile(os.path.join(heatmaps_path, heatmap_fnames[-1])):
                    print(f"  >> Error: '{heatmap_fnames[-1]}' not found in '{heatmaps_path}' for tester '{tester}'.")
                    heatmaps_paths.pop()
                    heatmap_fnames.pop()

            do_group_comparison(heatmaps_paths, heatmap_fnames, debug=DO_DEBUG, save_only=SAVE_ONLY)

        elif choice == "3":
            print("  >> Option not implemented yet.")
            continue

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("  >> Invalid choice, please enter 1-4.")