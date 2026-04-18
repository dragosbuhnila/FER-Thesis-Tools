import pickle
import os
import re
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.basefaces import get_base_face
from modules.filenames_utils import get_emotion_from_heatmap_relpath, get_emotion_full_from_path, reformat_bad_emotion_gtpred_name, try_extract_model_or_user_name, EMOTIONS
from modules.landmark_utils import AU_LANDMARKS, FACE_PARTS_LANDMARKS, FACE_PARTS_LANDMARKS_LRMERGED, get_all_AUs, get_all_face_parts, get_all_face_parts_lrmerged
from modules.mask_n_heatmap_utils import compute_pixel_repetition_heatmap, get_roi_matrix, invert_heatmap
from modules.save_load_utils import load_statistics, save_statistics
from modules.visualize import make_comparison_grid_combinations, make_comparison_grid_versus, plot_matrix, show_grid_matplotlib, show_heatmaps, show_heatmaps_grid
from modules.roi_statistics import compare_difmean, compare_meandif_pxbypx, convert_faceparts_roi_means_from_dict_to_vector, roi_mean, compare_meandif
from modules.saliencies_folders import HEATMAPS_ALE_SINGLE_HUMANS_DIR_PATH, saliencies_folders_rel_paths_phase1, saliencies_folders_rel_paths_phase2, testers_name_sets_phase1, testers_name_sets_phase2, saliencies_folders_rel_paths_phase2_subsets

# >======================================================================================
# >=========== MACROS ===================================================================
# >======================================================================================

COMPARE_CHOSEN = compare_meandif    # e.g. choose between: compare_meandif, compare_difmean
ROI_STAT_CHOSEN = roi_mean          # e.g. choose between: roi_mean, roi_hist
STAT_NAMES = {"roi_mean": "mean"}[ROI_STAT_CHOSEN.__name__] # e.g. "mean", "mean-std", "hist", etc.

# For a "correct" run, set debugs to false
DO_DEBUG = False  
DEBUG_ROIs = False  # If you use this, reweighting to see overlaps between ROIs will be enabled, as well as separate_lr for left/right distinctions, and forced recalculation of stats but with saving disabled
DEBUG_ROIs_SAVEONLY = False # If True, won't show the two plots (masked_heatmaps and repetition)
COMPARISONS_SAVEONLY = False
FORCE_RECALCULATE_STATS = False

ROI_DEBUG_FOLDER = "ROI_debug" if DEBUG_ROIs else None  

# WARNING: if you run this as main, the path will be relative to the current working directory, thus incorrect
SALIENCY_MAPS_DIR = os.path.join(".", "saliency_maps")
HUMAN_RESULTS_DIR = os.path.join(SALIENCY_MAPS_DIR, "HEATMAPS_humans_phase1", "Results")
OUTPUTS_DIR = os.path.join(SALIENCY_MAPS_DIR, "zzz_other_and_zips")
COMPARISON_GRID_FOLDER = os.path.join(OUTPUTS_DIR, "output_comparisons_meandif")


# <======================================================================================
# <=========== END OF MACROS ============================================================
# <======================================================================================

# ============ Extra Functions =================

def print_compare_saliency_maps_macros():
    print(f"COMPARE_CHOSEN: {COMPARE_CHOSEN.__name__}")
    print(f"ROI_STAT_CHOSEN: {ROI_STAT_CHOSEN.__name__}")
    print(f"STAT_NAMES: {STAT_NAMES}")
    print(f"DEBUG_ROIs: {DEBUG_ROIs}")
    print(f"DEBUG_ROIs_SAVEONLY: {DEBUG_ROIs_SAVEONLY}")
    print(f"FORCE_RECALCULATE_STATS: {FORCE_RECALCULATE_STATS}")
    print(f"ROI_DEBUG_FOLDER: {ROI_DEBUG_FOLDER}")
    print(f"COMPARISON_GRID_FOLDER: {COMPARISON_GRID_FOLDER}")
    print(f"SALIENCY_MAPS_DIR: {SALIENCY_MAPS_DIR}")
    print(f"HUMAN_RESULTS_DIR: {HUMAN_RESULTS_DIR}")

# ============ Main functions ==================

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

            # even older debug
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
        # old debug
        # for roi, masked_heatmap in masked_heatmaps.items():
        #     plot_name = f"{heatmap_fname} - {roi}".replace("/", "_")
        #     plot_matrix(masked_heatmap, title=plot_name, background=resized_baseface, save_folder=ROI_DEBUG_FOLDER, save_only=DEBUG_ROIs_SAVEONLY)
        
        stacked_roi_masks = np.stack(list(masked_heatmaps.values()))
        merged_masked_heatmap = np.nanmax(stacked_roi_masks, axis=0)
        plot_name = f"{heatmap_fname} - Merged Masks With Heatmap".replace("/", "_")
        plot_matrix(merged_masked_heatmap, title=plot_name, background=resized_baseface, save_folder=ROI_DEBUG_FOLDER, save_only=DEBUG_ROIs_SAVEONLY)

    return masked_heatmaps

def compute_heatmap_statistics(heatmap, heatmap_relpath, compute_stat, stat_names,
                               weigh_roi_overlap: bool, separate_lr, debug=False, force_recalculate=False, roi_type="faceparts", dont_save=False,
                               diagonal_only=True, subject_given=None):
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
        weigh_roi_overlap (bool): If True, will weigh the statistics by the pixel overlap between ROIs.
        diagonal_only (bool): If True, will only consider the diagonal saliency maps/heatmaps, i.e. the correct ones (Prediction same as Ground Truth).
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

    subject = subject_given if subject_given is not None else try_extract_model_or_user_name(heatmap_relpath)
    emotion_full = get_emotion_from_heatmap_relpath(heatmap_relpath) # e.g. => ANGRY_canonical

    if subject is None:
        subject = heatmap_relpath

    if emotion_full is None:
        emotion_full = ""

    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap {subject}/{emotion_full} is not a 2D array. Please check the file format.")

    if DEBUG_ROIs:
        separate_lr = False
        weigh_roi_overlap = True 
        force_recalculate = True

    # 0) Try to load stat if already cached
    print("------")
    if not force_recalculate:
        cached_statistics = load_statistics(heatmap_relpath, stat_names=stat_names, weigh_roi_overlap=weigh_roi_overlap, separate_lr=separate_lr, roi_type=roi_type)
        if cached_statistics is not None:
            print(f"Cached stats loaded for {subject}/{emotion_full}, weigh_roi_overlap={weigh_roi_overlap}, separate_lr={separate_lr}, roi_type={roi_type}, diagonal_only={diagonal_only}.")
            return cached_statistics, subject, emotion_full
        else:
            print(f"Cached stats unavailable for {subject}/{emotion_full}, weigh_roi_overlap={weigh_roi_overlap}, separate_lr={separate_lr}, roi_type={roi_type}, diagonal_only={diagonal_only}.")
    else:
        print(f"Cached stats ignored, forced recalculation for {subject}/{emotion_full}, weigh_roi_overlap={weigh_roi_overlap}, separate_lr={separate_lr}, roi_type={roi_type}, diagonal_only={diagonal_only}.")

    if debug:
        # print(f"Shape of heatmap {subject}/{emotion}: {heatmap.shape}")
        plot_matrix(heatmap, title=f"Heatmap for {subject}/{emotion_full}")

    # 1) Compute the masked heatmaps ROI by ROI
    masked_heatmaps = compute_masked_heatmaps(heatmap, f"{subject}/{emotion_full}", roi_type, emotion_full, debug=DEBUG_ROIs, separate_lr=separate_lr)
    print(f"Computed masked heatmaps for {subject}/{emotion_full} with {len(masked_heatmaps)} AUs (separate_lr={separate_lr}).")

    # 1.5) Based on the ROIs that we used, there may be some pixel repetition across AUs.
    if weigh_roi_overlap:
        repetition_map, amt_ovlp, amt_tot = compute_pixel_repetition_heatmap(masked_heatmaps, debug=False)
        if DEBUG_ROIs:
            baseface = get_base_face(emotion_full)
            resized_baseface = cv2.resize(baseface.image, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_LINEAR)
            plot_matrix(repetition_map, title=f"{subject}_{emotion_full} Pixel Repetition Heatmap ({amt_ovlp} of {amt_tot} overlapping)", background=resized_baseface, alpha=0.5, save_folder=ROI_DEBUG_FOLDER, save_only=DEBUG_ROIs_SAVEONLY)

        weightmap = invert_heatmap(repetition_map)  # Invert the heatmap to use it as a weightmap
        if debug:
            plot_matrix(weightmap, title=f"Weightmap for {subject}/{emotion_full}", background=resized_baseface, alpha=0.5)

        # print(f"Pixel repetition heatmap not skipped for {subject}/{emotion}.")
    else:
        # print(f"Pixel repetition heatmap skipped for {subject}/{emotion}.")
        weightmap = None

    # 2) Compute statistic
    statistics = {}
    for au, masked_heatmap in masked_heatmaps.items():
        stat = compute_stat(masked_heatmap, pxbypx_weightmap=weightmap, debug=debug)
        statistics[au] = stat
        if debug:
            print(f"Computed {stat_names} for AU {au}: {stat}")

    

    # 3) Cache the statistics (if debugging ROIs don't save as it may fuck up the stats size, since when debuggin I plot l/r ROIs together, while we usually don't do that with faceparts)
    if not DEBUG_ROIs and not dont_save:
        save_statistics(heatmap_relpath, statistics, weigh_roi_overlap=weigh_roi_overlap, separate_lr=separate_lr, roi_type=roi_type)

    if emotion_full and diagonal_only:
        emotion_name = emotion_full.split("_")[0]
    else:
        emotion_name = emotion_full

    return statistics, subject, emotion_name

def compute_group_of_heatmaps_statistics(heatmaps_relpaths, debug, force_recalculate_stats, diagonal_only=True, subject_given=None):
    unique_heatmap_names = []
    heatmaps = {}
    # stats_list_raw = {}
    # stats_list_pxwtd = {}
    # stats_list_sep_lr = {}
    stats_list_faceparts = {}

    for heatmap_relpath in heatmaps_relpaths:
        heatmap = np.load(heatmap_relpath)

        # if the heatmap is only nans then skip it
        # if np.all(np.isnan(heatmap)):
        #     continue

        # compute both raw and pixel-weighted statistics, so you can plot both for comparison
        # - pxwtd means that the statistics are weighted by the pixel repetition across AUs

        # stats_raw, _, _     = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=False, debug=debug, force_recalculate=force_recalculate_stats, separate_lr=False, roi_type="aus", diagonal_only=diagonal_only)
        # stats_pxwtd, _, _   = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=True, debug=debug, force_recalculate=force_recalculate_stats, separate_lr=False, roi_type="aus", diagonal_only=diagonal_only)
        # stats_sep_lr, _, _  = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=False, debug=debug, force_recalculate=force_recalculate_stats, separate_lr=True, roi_type="aus", diagonal_only=diagonal_only)
        stats_faceparts, subject, emotion = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=False, debug=debug, force_recalculate=force_recalculate_stats, separate_lr=True, roi_type="faceparts", diagonal_only=diagonal_only, subject_given=subject_given)
        unique_heatmap_name = f"{subject}/{emotion}"
        
        unique_heatmap_names.append(unique_heatmap_name)
        heatmaps[unique_heatmap_name] = heatmap

        # stats_list_raw[unique_heatmap_name] = stats_raw
        # stats_list_pxwtd[unique_heatmap_name] = stats_pxwtd
        # stats_list_sep_lr[unique_heatmap_name] = stats_sep_lr
        stats_list_faceparts[unique_heatmap_name] = stats_faceparts

    return unique_heatmap_names, heatmaps, stats_list_faceparts

def do_group_comparison_combinations(heatmaps_relpaths, debug, save_only=False, force_recalculate_stats=False):
    # 1) Compute statistics for all heatmaps
    unique_heatmap_names, heatmaps, stats_list_faceparts = compute_group_of_heatmaps_statistics(heatmaps_relpaths, debug, force_recalculate_stats)

    # 2) Make the grids
    # grid_raw        = make_comparison_grid(unique_heatmap_names, stats_list_raw,   COMPARE_CHOSEN)
    # grid_pxwtd      = make_comparison_grid(unique_heatmap_names, stats_list_pxwtd, COMPARE_CHOSEN)
    # grid_sep_lr     = make_comparison_grid(unique_heatmap_names, stats_list_sep_lr, COMPARE_CHOSEN)
    grid_faceparts  = make_comparison_grid_combinations(unique_heatmap_names, stats_list_faceparts, COMPARE_CHOSEN)

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
        print(f"Not displaying plt for Combinations comparison of heatmaps: {', '.join(unique_heatmap_names)}")
    else:
        show_heatmaps(heatmaps)

        # show_grid_matplotlib(grid_pxwtd, title="Stat Comparison: AUs-reweighted, symmetric")
        # show_grid_matplotlib(grid_raw, title="Stat Comparison: AUs-not-reweighted, symmetric")
        # show_grid_matplotlib(grid_sep_lr, title="Stat Comparison: AUs-not-reweighted, l/r")
        show_grid_matplotlib(grid_faceparts, title="Stat Comparison: Faceparts, l/r")

        plt.show()

def do_group_comparison_versus(heatmaps_relpaths_1, heatmaps_relpaths_2, debug, save_only, force_recalculate_stats, subject_1=None, subject_2=None, subset=None):
    # 1) Compute statistics for both groups
    unique_heatmap_names_1, heatmaps_1, stats_list_faceparts_1 = compute_group_of_heatmaps_statistics(heatmaps_relpaths_1, debug, force_recalculate_stats, diagonal_only=False, subject_given=subject_1)
    unique_heatmap_names_2, heatmaps_2, stats_list_faceparts_2 = compute_group_of_heatmaps_statistics(heatmaps_relpaths_2, debug, force_recalculate_stats, diagonal_only=False, subject_given=subject_2)



    subject_1_extracted = unique_heatmap_names_1[0].split("/")[0] if unique_heatmap_names_1 else ""
    if subject_1 is not None and subject_1_extracted != subject_1:
        print(f"Warning: subject_1 '{subject_1}' does not match extracted subject 1 '{subject_1_extracted}' from heatmap names.")
    else:
        subject_1 = subject_1_extracted
    subject_2_extracted = unique_heatmap_names_2[0].split("/")[0] if unique_heatmap_names_2 else ""
    if subject_2 is not None and subject_2_extracted != subject_2:
        print(f"Warning: subject_2 '{subject_2}' does not match extracted subject 2 '{subject_2_extracted}' from heatmap names.")
    else:
        subject_2 = subject_2_extracted

    # print(f"Unique heatmap names (Group 1): {unique_heatmap_names_1}")
    # print(f"Unique heatmap names (Group 2): {unique_heatmap_names_2}")

    # 2) Make the grids
    grid_faceparts = make_comparison_grid_versus(unique_heatmap_names_1, unique_heatmap_names_2, stats_list_faceparts_1, stats_list_faceparts_2, COMPARE_CHOSEN)
    grid_pxbypx_mean = make_comparison_grid_versus(unique_heatmap_names_1, unique_heatmap_names_2, heatmaps_1, heatmaps_2, compare_meandif_pxbypx)

    # check if some files were missed by comparing non NaNs in grid_faceparts and common names between unique_heatmap_names_1 and unique_heatmap_names_2
    non_nan_entries = grid_faceparts.stack().index.tolist()
    non_nan_entries = [(i, j) for i, j in non_nan_entries if grid_faceparts.loc[i, j].lower() != "nan"]
    # Extract only the emotion part (after the "/") for comparison
    emotions_1 = set(name.split("/", 1)[1] for name in unique_heatmap_names_1)
    emotions_2 = set(name.split("/", 1)[1] for name in unique_heatmap_names_2)
    common_names = emotions_1.intersection(emotions_2)
    if len(non_nan_entries) != len(common_names):
        raise ValueError(f"Some files were missed in the comparison! There are {len(common_names)} common names between the two groups, but only {len(non_nan_entries)} non-NaN entries in the comparison grid. Non-NaN entries: {non_nan_entries}, Common names: {common_names}")

    # 3) Show the grids and heatmaps
    if save_only:
        print(f"Not displaying plt for Versus comparison between {subject_1} and {subject_2}")
    else:
        # fix pattlite to mobilenet
        subject_1 = subject_1.replace("pattlite", "mobilenet") if "pattlite" in subject_1 else subject_1
        subject_2 = subject_2.replace("pattlite", "mobilenet") if "pattlite" in subject_2 else subject_2
        subject_1 = subject_1.replace("PATTLITE", "MOBILENET") if "PATTLITE" in subject_1 else subject_1
        subject_2 = subject_2.replace("PATTLITE", "MOBILENET") if "PATTLITE" in subject_2 else subject_2



        show_heatmaps_grid(heatmaps_1, title=f"Comparison {subject_1} VS {subject_2} - {subject_1}_heatmaps", alpha=0.5, save_folder=COMPARISON_GRID_FOLDER, save_only=save_only, block=False, subset=subset)
        show_heatmaps_grid(heatmaps_2, title=f"Comparison {subject_1} VS {subject_2} - {subject_2}_heatmaps", alpha=0.5, save_folder=COMPARISON_GRID_FOLDER, save_only=save_only, block=False, subset=subset)
        show_grid_matplotlib(grid_faceparts, title=f"Comparison {subject_1} VS {subject_2} with {COMPARE_CHOSEN.__name__}()", axis_names=("True", "Predicted"), save_folder=COMPARISON_GRID_FOLDER, save_only=save_only, block=False)
        show_grid_matplotlib(grid_pxbypx_mean, title=f"Comparison {subject_1} VS {subject_2} with compare_meandif_pxbypx()", axis_names=("True", "Predicted"), save_folder=COMPARISON_GRID_FOLDER, save_only=save_only, block=False)

        plt.show()

# ============ END OF MAIN FUNCTIONS ==================

# ============ Menu functions ==================

# 0.1) Doesn't directly do a comparison, it's useful as a tool
def recalculate_all_stats():
    TESTERS = [
        d for d in os.listdir(HUMAN_RESULTS_DIR)
        if os.path.isdir(os.path.join(HUMAN_RESULTS_DIR, d))
    ]

    for tester in TESTERS:
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
        print(f"    >> heatmaps_relpaths: {heatmaps_relpaths}")
        do_group_comparison_combinations(heatmaps_relpaths, debug=DO_DEBUG, save_only=True, force_recalculate_stats=True)

# 0.2) Again, doesn't directly do a comparison, it's useful as a tool
def test_comparison():
    print(f"DEBUG: {os.path.abspath(SALIENCY_MAPS_DIR)}")

    heatmaps_relpaths = [
        os.path.join(SALIENCY_MAPS_DIR, "std_saliency_map_ones.npy"),
        os.path.join(SALIENCY_MAPS_DIR, "std_saliency_map_zeroes.npy")
    ]

    for heatmap_relpath in heatmaps_relpaths:
        if not os.path.isfile(heatmap_relpath):
            print(f"  >> Error: heatmap file '{heatmap_relpath}' not found.")
            continue

    print(f"Comparing heatmaps: {heatmaps_relpaths[0]} and {heatmaps_relpaths[1]}")
    do_group_comparison_combinations(heatmaps_relpaths, debug=DO_DEBUG, save_only=False, force_recalculate_stats=True)

# 1) Compare single person's heatmaps, meaning that it will do all combinations of the 7 emotions for that subject (heatmaps used are the diagonal ones, i.e. only correct guesses)
def compare_single_person():
    TESTERS = [
        d for d in os.listdir(HUMAN_RESULTS_DIR)
        if os.path.isdir(os.path.join(HUMAN_RESULTS_DIR, d))
    ]
    print(f"Available IDs: {', '.join(TESTERS)}")
    tester = input("Enter Person ID: ").strip().upper()

    if tester not in TESTERS:
        print(f"  >> Error: '{tester}' not found. Try again.")
        return

    heatmaps_path = os.path.join(HUMAN_RESULTS_DIR, tester, "heatmaps")
    if not os.path.isdir(heatmaps_path):
        print(f"  >> Error: heatmaps folder '{heatmaps_path}' missing.")
        return

    heatmap_fnames = [
        f for f in os.listdir(heatmaps_path)
        if f.endswith("canonical.npy")
    ]
    if not heatmap_fnames:
        print(f"  >> No canonical.npy files in '{heatmaps_path}'.")
        return

    heatmaps_relpaths = [os.path.join(heatmaps_path, fname) for fname in heatmap_fnames]
    print(f"    >> heatmaps_relpaths: {heatmaps_relpaths}")
    do_group_comparison_combinations(heatmaps_relpaths, debug=DO_DEBUG, save_only=False, force_recalculate_stats=FORCE_RECALCULATE_STATS)

# 2) Compare top performers' heatmaps across a selected emotion (again, only uses the diagonal heatmap of the selected emotion, i.e. only correct guesses)
def compare_top_and_emotions():
    testers_ranking_path = "./saliency_maps/HEATMAPS_humans_phase1/Results/testers_ranking.pkl"
    if os.path.exists(testers_ranking_path):
        with open(testers_ranking_path, "rb") as f:
            testers_ranking = pickle.load(f)
            testers_ranking = sorted(testers_ranking, key=lambda x: x[1], reverse=True)  # resort to be sure
            testers_ranking = [tester_name for tester_name, _ in testers_ranking]  # keep only names

    # print(f"Loaded testers' ranking from {testers_ranking_path}. Found {len(testers_ranking)} testers:")
    # for rank, tester in enumerate(testers_ranking, 1):
    #     print(f"{rank:2d}. {tester}")

    # Select emotion
    print("Choose emotion between ", ", ".join(EMOTIONS))
    selected_emotion = input("Enter Emotion: ").strip().upper()
    if selected_emotion not in EMOTIONS:
        print(f"  >> Error: '{selected_emotion}' not found. Try again.")
        return

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
            return

        topX_performers = testers_ranking[start:end]
    except ValueError:
        print(f"  >> Error: Invalid range '{range_input}'. Try again.")
        return
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
    print(f"    >> heatmaps_relpaths: {heatmaps_relpaths}")
    do_group_comparison_combinations(heatmaps_relpaths, debug=DO_DEBUG, save_only=False, force_recalculate_stats=FORCE_RECALCULATE_STATS)


def get_saliencies_folders_rel_paths(phase, subset=None):
    if int(phase) == 1:
        saliencies_folders_rel_paths = saliencies_folders_rel_paths_phase1
    elif int(phase) == 2:
        if subset is not None:
            if subset in saliencies_folders_rel_paths_phase2_subsets:
                saliencies_folders_rel_paths = saliencies_folders_rel_paths_phase2_subsets[subset]
            else:
                raise ValueError(f"Invalid subset: {subset}. Available subsets: {', '.join(saliencies_folders_rel_paths_phase2_subsets.keys())}")
        else:
            saliencies_folders_rel_paths = saliencies_folders_rel_paths_phase2
    else:
        raise ValueError(f"Invalid phase: {phase}. Please choose either 1 or 2.")
    
    return saliencies_folders_rel_paths

def get_testers_name_sets(phase):
    if int(phase) == 1:
        testers_name_sets = testers_name_sets_phase1
        human_results_dir = HUMAN_RESULTS_DIR
    elif int(phase) == 2:
        testers_name_sets = testers_name_sets_phase2
        human_results_dir = HEATMAPS_ALE_SINGLE_HUMANS_DIR_PATH
    else:
        raise ValueError(f"Invalid phase: {phase}. Please choose either 1 or 2.")

    return human_results_dir, testers_name_sets


def compare_two_subjects():
    phase = input("Enter phase (1 or 2): ")
    saliencies_folders_rel_paths = get_saliencies_folders_rel_paths(phase)

    subject_list = saliencies_folders_rel_paths.keys()
    subject_1 = input(f"Enter first subject ({', '.join(subject_list)}): ").strip().lower()
    subject_2 = input(f"Enter second subject ({', '.join(subject_list)}): ").strip().lower()

    compare_two_subjects_cmd(phase, subject_1, subject_2)

def compare_two_subjects_cmd(phase, subject_1, subject_2, subset=None):
    saliencies_folders_rel_paths = get_saliencies_folders_rel_paths(phase, subset)

    folder_relpath_1 = saliencies_folders_rel_paths.get(subject_1)
    folder_relpath_2 = saliencies_folders_rel_paths.get(subject_2)

    if (not folder_relpath_1) or (not folder_relpath_2):
        print(f"  >> Error: One or both subjects not found in 'saliencies_folders_rel_paths' dictionary.")
        return

    folder_path_1 = os.path.join(SALIENCY_MAPS_DIR, folder_relpath_1)
    folder_path_2 = os.path.join(SALIENCY_MAPS_DIR, folder_relpath_2)

    if not os.path.exists(folder_path_1):
        print(f"  >> Warning: Folder path '{folder_path_1}' does not exist.")
        return
    
    if not os.path.exists(folder_path_2):
        print(f"  >> Warning: Folder path '{folder_path_2}' does not exist.")
        return

    print(f"Comparing subjects in:\n  {folder_path_1}\n  {folder_path_2}")

    folder_1_filenames = os.listdir(folder_path_1)
    folder_2_filenames = os.listdir(folder_path_2)

    folder_1_relpaths = [os.path.join(folder_path_1, fname) for fname in folder_1_filenames if fname.endswith(".npy")]
    folder_2_relpaths = [os.path.join(folder_path_2, fname) for fname in folder_2_filenames if fname.endswith(".npy")]

    print(f"  >> Folder 1: Found {len(folder_1_relpaths)} heatmaps_relpaths: {folder_1_relpaths}")
    print(f"  >> Folder 2: Found {len(folder_2_relpaths)} heatmaps_relpaths: {folder_2_relpaths}")

    do_group_comparison_versus(folder_1_relpaths, folder_2_relpaths, debug=DO_DEBUG, save_only=COMPARISONS_SAVEONLY, force_recalculate_stats=FORCE_RECALCULATE_STATS, subject_1=subject_1.upper(), subject_2=subject_2.upper(), subset=subset)

def create_organized_folders_aggr():
    phase = input("Enter phase (1 or 2): ")
    saliencies_folders_rel_paths = get_saliencies_folders_rel_paths(phase)

    subject_list = saliencies_folders_rel_paths.keys()
    subject_1 = input(f"Enter first subject ({', '.join(subject_list)}): ").strip().lower()
    subject_2 = input(f"Enter second subject ({', '.join(subject_list)}): ").strip().lower()

    create_organized_folders_aggr_cmd(phase, subject_1, subject_2)

def create_organized_folders_aggr_cmd(phase, subject_1, subject_2, subset=None):
    """ Extracts the meanvectors for all saliency maps of the aggregated subjects, and organizes the results as follows:
        (> is folder, - is file)
            > ANGRY_ANGRY
                - ANGRY_ANGRY_GROUP1.npy
                - ANGRY_ANGRY_GROUP2.npy
            > ANGRY_HAPPY
                - ANGRY_HAPPY_GROUP1.npy
                - ANGRY_HAPPY_GROUP2.npy
            > ...
    """
    saliencies_folders_rel_paths = get_saliencies_folders_rel_paths(phase, subset)

    folder_relpath_1 = saliencies_folders_rel_paths.get(subject_1)
    folder_relpath_2 = saliencies_folders_rel_paths.get(subject_2)
    comparison_subjects = f"{subject_1.upper()}_VS_{subject_2.upper()}"

    if (not folder_relpath_1) or (not folder_relpath_2):
        print(f"  >> Error: One or both subjects not found in 'saliencies_folders_rel_paths' dictionary.")
        return

    folder_path_1 = os.path.join(SALIENCY_MAPS_DIR, folder_relpath_1)
    folder_path_2 = os.path.join(SALIENCY_MAPS_DIR, folder_relpath_2)

    if not os.path.exists(folder_path_1):
        print(f"  >> Warning: Folder path '{folder_path_1}' does not exist.")
        return
    
    if not os.path.exists(folder_path_2):
        print(f"  >> Warning: Folder path '{folder_path_2}' does not exist.")
        return


    print(f"Comparing subjects in:\n  {folder_path_1}\n  {folder_path_2}")


    folder_1_filenames = os.listdir(folder_path_1)
    folder_2_filenames = os.listdir(folder_path_2)

    folder_1_relpaths = [os.path.join(folder_path_1, fname) for fname in folder_1_filenames if fname.endswith(".npy")]
    folder_2_relpaths = [os.path.join(folder_path_2, fname) for fname in folder_2_filenames if fname.endswith(".npy")]

    # > only keep the heatmaps that are present in both folders
    emotion_gt_pred_occurrences = {}
    for relpath in folder_1_relpaths + folder_2_relpaths:
        emotion_gt_pred = get_emotion_full_from_path(relpath)
        emotion_gt_pred_occurrences[emotion_gt_pred] = emotion_gt_pred_occurrences.get(emotion_gt_pred, 0) + 1

    folder_1_relpaths = [relpath for relpath in folder_1_relpaths if get_emotion_full_from_path(relpath) in emotion_gt_pred_occurrences and emotion_gt_pred_occurrences[get_emotion_full_from_path(relpath)] == 2]
    folder_2_relpaths = [relpath for relpath in folder_2_relpaths if get_emotion_full_from_path(relpath) in emotion_gt_pred_occurrences and emotion_gt_pred_occurrences[get_emotion_full_from_path(relpath)] == 2]

    print(f"  >> Folder 1: Found {len(folder_1_relpaths)} heatmaps_relpaths: {folder_1_relpaths}")
    print(f"  >> Folder 2: Found {len(folder_2_relpaths)} heatmaps_relpaths: {folder_2_relpaths}")
    # < 

    # > now compute the meanvectors for all heatmaps in both folders
    folder_1_relpaths.sort(key=lambda x: get_emotion_full_from_path(x))
    folder_2_relpaths.sort(key=lambda x: get_emotion_full_from_path(x))
    for relpath1, relpath2 in zip(folder_1_relpaths, folder_2_relpaths):
        emotion_gt_pred = get_emotion_full_from_path(relpath1)
        if emotion_gt_pred != get_emotion_full_from_path(relpath2):
            raise ValueError(f"Emotion mismatch between '{relpath1}' and '{relpath2}'")

        heatmap1 = np.load(relpath1)
        heatmap2 = np.load(relpath2)

        stats1, subject1, emotion1 = compute_heatmap_statistics(heatmap1, relpath1, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=False, debug=DO_DEBUG, force_recalculate=FORCE_RECALCULATE_STATS, separate_lr=True, roi_type="faceparts", diagonal_only=False, subject_given=subject_1.upper())
        stats2, subject2, emotion2 = compute_heatmap_statistics(heatmap2, relpath2, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=False, debug=DO_DEBUG, force_recalculate=FORCE_RECALCULATE_STATS, separate_lr=True, roi_type="faceparts", diagonal_only=False, subject_given=subject_2.upper())

        stats1 = convert_faceparts_roi_means_from_dict_to_vector(stats1)
        stats2 = convert_faceparts_roi_means_from_dict_to_vector(stats2)

        if subject1 != subject_1.upper() or subject2 != subject_2.upper() or emotion1 != emotion2 or emotion1 != emotion_gt_pred:
            print(f"  >> Warning: Mismatched subjects or emotions after extraction: '{subject1}/{emotion1}' and '{subject2}/{emotion2}' vs expected '{subject_1.upper()}/{emotion_gt_pred}' and '{subject_2.upper()}/{emotion_gt_pred}'")
            continue

        # Create output folder if it doesn't exist
        emotion_gt_pred = reformat_bad_emotion_gtpred_name(emotion_gt_pred)
        output_folder = os.path.join(OUTPUTS_DIR, f"mean-vectors_organized_aggregated_{subset}", comparison_subjects, f"{emotion_gt_pred}")
        os.makedirs(output_folder, exist_ok=True)

        # Save the statistics
        output_path_1 = os.path.join(output_folder, f"{emotion_gt_pred}_GROUP1.npy")
        output_path_2 = os.path.join(output_folder, f"{emotion_gt_pred}_GROUP2.npy")
        np.save(output_path_1, stats1)
        np.save(output_path_2, stats2)
        print(f"  >> Saved aggregated stats for '{emotion_gt_pred}' to '{output_path_1}' and '{output_path_2}'")

    # stats_faceparts, subject, emotion = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=False, debug=debug, force_recalculate=force_recalculate_stats, separate_lr=False, roi_type="faceparts", diagonal_only=diagonal_only, subject_given=subject_given)

def save_gran_vector(name, emotion_gt_and_pred, emotion_gt_and_pred_clean, comparison_subjects, group, human_results_dir, verbose=False):
    if group not in ["GROUP1", "GROUP2"]:
        raise ValueError(f"Group must be 'GROUP1' or 'GROUP2', got '{group}'")

    heatmap_relpath = os.path.join(human_results_dir, name, "heatmaps", f"{emotion_gt_and_pred}.npy")
    if not os.path.isfile(heatmap_relpath):
        return None

    heatmap = np.load(heatmap_relpath)
    stats, subject, emotion = compute_heatmap_statistics(heatmap, heatmap_relpath, ROI_STAT_CHOSEN, STAT_NAMES, weigh_roi_overlap=False, debug=DO_DEBUG, force_recalculate=FORCE_RECALCULATE_STATS, separate_lr=True, roi_type="faceparts", diagonal_only=False, subject_given=name.upper())
    stats = convert_faceparts_roi_means_from_dict_to_vector(stats)

    output_folder = os.path.join(OUTPUTS_DIR, "mean-vectors_organized_granular", comparison_subjects, f"{emotion_gt_and_pred_clean}", group)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{emotion_gt_and_pred_clean}_{name}.npy")

    np.save(output_path, stats)
    if verbose:
        print(f"  >> Saved granular stats for '{emotion_gt_and_pred_clean}' and tester '{name}' to '{output_path}'")

def create_organized_folders_gran():
    phase = input("Enter phase (1 or 2): ")
    subject_list_granular = ["men", "women", "best", "worst"]
    
    saliencies_folders_rel_paths = get_saliencies_folders_rel_paths(phase)
    subject_list_full = saliencies_folders_rel_paths.keys()

    for subject in subject_list_granular:
        if subject not in subject_list_full:
            raise ValueError(f"Subject eligible for granular comparison '{subject}' not found in 'saliencies_folders_rel_paths' dictionary (keys: {', '.join(subject_list_full)})")

    subject_1 = input(f"Enter first subject ({', '.join(subject_list_granular)}): ").strip().lower()
    subject_2 = input(f"Enter second subject ({', '.join(subject_list_granular)}): ").strip().lower()

    create_organized_folders_gran_cmd(phase, subject_1, subject_2)

def create_organized_folders_gran_cmd(phase, subject_1, subject_2):
    """ Extracts the meanvectors for all saliency maps of the two subjects, in a granular manner, and organizes the results as follows:
        (> is folder, - is file)
        > ANGRY_ANGRY
            > GROUP1
                - ANGRY_ANGRY_NAME1_mean-vector.npy
                - ANGRY_ANGRY_NAME2_mean-vector.npy
                - ANGRY_ANGRY_NAME3_mean-vector.npy
            > GROUP2
                - ...
        > ANGRY_HAPPY
            > GROUP1
                - ...
            > GROUP2
                - ...
        > ...
    """
    human_results_dir, testers_name_sets = get_testers_name_sets(phase)
    print(f"Testers name sets for phase {phase}:")
    print(testers_name_sets)

    # > get the names of the testers for both subjects (e.g. set for men and set for women)
    name_set_1 = testers_name_sets.get(subject_1)
    name_set_2 = testers_name_sets.get(subject_2)
    if (not name_set_1) or (not name_set_2):
        raise ValueError(f"One or both subjects not found in 'testers_name_sets' dictionary. Keys tried are '{subject_1}' and '{subject_2}'. Available keys are: {', '.join(testers_name_sets.keys())}")
    # < 

    comparison_subjects = f"{subject_1.upper()}_VS_{subject_2.upper()}"

    # > get the full list of emotions that are present in both sets of testers
    name_set_complete = name_set_1 | name_set_2
    emotions_gt_and_pred = set()
    for tester in name_set_complete:
        heatmaps_path = os.path.join(human_results_dir, tester, "heatmaps")
        if not os.path.isdir(heatmaps_path):
            raise ValueError(f"  >> Error: heatmaps folder '{heatmaps_path}' missing for tester '{tester}'.")
        
        heatmap_fnames = [f.split(".npy")[0] for f in os.listdir(heatmaps_path) if f.endswith(".npy")]
        emotions_gt_and_pred.update(heatmap_fnames)
    # < 

    # > for each emotion, extract the meanvectors for all testers in both sets, and save them in the organized folder structure
    for emotion_gt_and_pred in emotions_gt_and_pred:
        emotion_gt_and_pred_clean = reformat_bad_emotion_gtpred_name(emotion_gt_and_pred)
        
        for name in name_set_1:
            res = save_gran_vector(name, emotion_gt_and_pred, emotion_gt_and_pred_clean, comparison_subjects, group="GROUP1", human_results_dir=human_results_dir, verbose=True)
            if res is None:
                continue
        for name in name_set_2:
            res = save_gran_vector(name, emotion_gt_and_pred, emotion_gt_and_pred_clean, comparison_subjects, group="GROUP2", human_results_dir=human_results_dir, verbose=True)
            if res is None:
                continue

# ==================================================================

def merge_generated_images(new_paths, output_image_path, model_names, images_per_row=7, vertical=False):
    """
    Merges all generated images into a single image, grouped by XAI method, with model names above each image
    and XAI method names below the model names.
    Args:
        new_paths (dict): Dictionary of XAI method names and their corresponding image paths.
        output_image_path (str): Path to save the merged image.
        model_names (list): List of model names.
        images_per_row (int): Number of images per row (or column if vertical=True).
        vertical (bool): If True, display the image vertically (models as rows, XAI methods as columns).
    """
    # Open all images and group them by XAI method
    grouped_images = {}
    for old_path, image_path in new_paths.items():
        if "bubbles" in old_path.lower():
            xai_method = "Bubbles"
        elif "external" in old_path.lower() or "extpert" in old_path.lower():
            xai_method = "External"
        elif "gradcam" in old_path.lower():
            xai_method = "GradCAM"
        else: 
            raise ValueError(f"Unknown XAI method for path: {old_path}")

        for model_name in model_names:
            if model_name.lower() in old_path.lower():
                model = model_name
                break
        else:
            model = None

        if os.path.exists(image_path):
            img = Image.open(image_path)
            if xai_method not in grouped_images:
                grouped_images[xai_method] = []
            grouped_images[xai_method].append((img, model, xai_method))  # Store image, model name, and XAI method
        else:
            print(f"Image not found: {image_path}")

    # Determine the size of each image (assuming all images are the same size)
    first_image = next(iter(grouped_images.values()))[0][0]
    image_width, image_height = first_image.size

    # Add space for model names and XAI methods above each image
    text_height = 60  # Increase the height for the text area above each image
    total_image_height = image_height + text_height
    total_image_width = image_width + text_height

    # Calculate the size of the final merged image
    if vertical:
        num_rows = len(model_names)  # One row per model
        num_cols = len(grouped_images)  # One column per XAI method
        canvas_width = num_cols * total_image_width
        canvas_height = num_rows * total_image_height
    else:
        num_rows = len(grouped_images)  # One row per XAI method
        canvas_width = images_per_row * image_width
        canvas_height = num_rows * total_image_height

    # Create a blank canvas for the final image
    merged_image = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))

    # Create a drawing context for adding text
    draw = ImageDraw.Draw(merged_image)

    # Use a larger font size for the model names and XAI methods
    try:
        font_model = ImageFont.truetype("arial.ttf", 24)  # Font for model names
        font_xai = ImageFont.truetype("arial.ttf", 20)    # Font for XAI methods
    except IOError:
        font_model = ImageFont.load_default()  # Fallback to default font if "arial.ttf" is unavailable
        font_xai = ImageFont.load_default()

    # Paste images into the canvas with model names and XAI methods
    if vertical:
        x_offset = 0
        for model_name in model_names:
            y_offset = 0
            for xai_method, images in grouped_images.items():
                for img, img_model_name, img_xai_method in images:
                    if img_model_name == model_name and img_xai_method == xai_method:
                        # Paste the image
                        merged_image.paste(img, (x_offset + text_height, y_offset + text_height))

                        # Add the XAI method above the image
                        text_x = x_offset + total_image_width // 2
                        text_y_xai = y_offset + 10  # Position for the XAI method
                        draw.text((text_x, text_y_xai), xai_method, fill="gray", anchor="mm", font=font_xai)

                        # Add the model name to the left of the image
                        text_y = y_offset + total_image_height // 2
                        text_x_model = x_offset + 10  # Position for the model name
                        draw.text((text_x_model, text_y), model_name, fill="black", anchor="mm", font=font_model)

                y_offset += total_image_height
            x_offset += total_image_width
    else:
        y_offset = 0
        for xai_method, images in grouped_images.items():
            for idx, (img, model_name, xai_method) in enumerate(images):
                row = idx // images_per_row
                col = idx % images_per_row

                # Calculate the position to paste the image
                x_offset = col * image_width
                y_image_offset = y_offset + text_height

                # Paste the image
                merged_image.paste(img, (x_offset, y_image_offset))

                # Add the model name above the image
                if model_name:
                    text_x = x_offset + image_width // 2
                    text_y_model = y_offset + 10  # Position for the model name
                    draw.text((text_x, text_y_model), model_name, fill="black", anchor="mm", font=font_model)

                # Add the XAI method below the model name
                text_y_xai = y_offset + 40  # Position for the XAI method (below the model name)
                draw.text((text_x, text_y_xai), xai_method, fill="gray", anchor="mm", font=font_xai)

            # Move to the next row for the next XAI method
            y_offset += total_image_height

    # Save the final merged image
    merged_image.save(output_image_path)
    print(f"Merged image saved to: {output_image_path}")


def compute_left_right(source_folder_path, output_npy_path):
    """
    Computes left and right mean values for saliency maps in a folder and visualizes them in a confusion matrix.
    Args:
        folder_path (str): Path to the folder containing saliency maps.
        output_path (str): Path to the new png.
    """
    # List all .npy files in the folder
    heatmap_files = [f for f in os.listdir(source_folder_path) if f.endswith(".npy")]
    if not heatmap_files:
        print(f"No saliency maps found in folder: {source_folder_path}")
        return

    # Initialize confusion matrix
    num_emotions = len(EMOTIONS)
    confusion_matrix = np.zeros((num_emotions, num_emotions, 2))  # Shape: (num_emotions, num_emotions, 2)

    # Process each heatmap
    for heatmap_file in heatmap_files:
        heatmap_path = os.path.join(source_folder_path, heatmap_file)
        heatmap = np.load(heatmap_path)

        # Split heatmap into left and right halves
        mid_col = heatmap.shape[1] // 2
        left_half = heatmap[:, :mid_col]
        right_half = heatmap[:, mid_col:]

        # Compute mean values for left and right halves
        left_mean = np.nanmean(left_half)
        right_mean = np.nanmean(right_half)

        # if completely empty set to 0
        if np.isnan(left_mean):
            left_mean = 0
        if np.isnan(right_mean):
            right_mean = 0

        # Extract ground truth and predicted emotions from the filename
        emotion_gt_pred = get_emotion_full_from_path(heatmap_file)  # e.g., "ANGRY_HAPPY"
        if not emotion_gt_pred:
            raise ValueError(f"Could not extract emotion from filename: {heatmap_file}")
        emotion_gt_pred = reformat_bad_emotion_gtpred_name(emotion_gt_pred)

        emotion_gt, emotion_pred = emotion_gt_pred.split("_")
        if emotion_gt not in EMOTIONS or emotion_pred not in EMOTIONS:
            print(f"Invalid emotions in filename: {heatmap_file}")
            continue

        # Get indices for the confusion matrix
        gt_idx = EMOTIONS.index(emotion_gt)
        pred_idx = EMOTIONS.index(emotion_pred)

        # Store left and right means in the confusion matrix
        confusion_matrix[gt_idx, pred_idx, 0] += left_mean
        confusion_matrix[gt_idx, pred_idx, 1] += right_mean

    # Visualize the confusion matrix
    fig, ax = plt.subplots()

    # Initialize variables for global statistics
    blue_count = 0    # Count of blue cells (right > left)
    red_count = 0     # Count of red cells (left >= right)

    for i in range(num_emotions):
        for j in range(num_emotions):
            left_mean, right_mean = confusion_matrix[i, j]
            if left_mean == 0 and right_mean == 0:
                continue  # Skip empty cells

            # Determine color and transparency
            if right_mean > left_mean:
                color = "blue"
                blue_count += 1
            else:
                color = "red"
                red_count += 1
            alpha = abs(right_mean - left_mean) / max(left_mean, right_mean, 1e-5)  # Normalize transparency

            # Draw the square
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color, alpha=min(1, alpha)))

            # Add text for left and right means
            ax.text(j + 0.5, i + 0.7, f"L: {left_mean:.2f}", ha="center", va="center", fontsize=8, color="black")
            ax.text(j + 0.5, i + 0.3, f"R: {right_mean:.2f}", ha="center", va="center", fontsize=8, color="black")

    # Compute global statistics
    global_difference = blue_count - red_count

    # Set axis labels and ticks
    ax.set_xticks(np.arange(num_emotions) + 0.5)
    ax.set_yticks(np.arange(num_emotions) + 0.5)
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlim(0, num_emotions)
    ax.set_ylim(0, num_emotions)
    ax.invert_yaxis()

    # Add title with padding
    ax.set_title("Left vs Right Means (Blue is Right > Left)", fontsize=14, pad=13)  # Add padding to the title

    # Add global statistics below the title
    stats_text = f"Global Difference: {global_difference}"
    color = "red" if global_difference < 0 else "blue"
    ax.text(
        0.5, 1.02, stats_text, ha="center", va="center", fontsize=10, color=color, transform=ax.transAxes
    )

    # Ensure the matrix is perfectly rectangular
    ax.set_aspect("equal")  # Force square cells
    plt.tight_layout(pad=2.0)  # Adjust layout to add padding around the plot

    # Save the plot
    output_dir = os.path.dirname(output_npy_path)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_npy_path)
    plt.close()
    # plt.show()




def compute_left_right_wrapper(src_heatmaps_folder_basename="HEATMAPS_machines_phase2"):
    src_heatmaps_base_path = os.path.join(SALIENCY_MAPS_DIR, src_heatmaps_folder_basename)
    dest_heatmaps_folder_basename = src_heatmaps_folder_basename + "_left_right_analysis"
    dest_heatmaps_folder_path = src_heatmaps_base_path.replace(src_heatmaps_folder_basename, dest_heatmaps_folder_basename)

    if "fede" in src_heatmaps_base_path:
        MODEL_NAMES = ["convnext", "efficientnetb1", "inceptionv3", "pattlite", "resnet", "vgg19", "yolo"] # for fede
    elif "occft" in src_heatmaps_base_path:
        MODEL_NAMES = ["occft_convnext", "occft_efficientnetb1", "occft_inceptionv3", "occft_pattlite", "occft_resnet", "occft_vgg19", "occft_yolo"] # for occft
    else:
        raise ValueError(f"Invalid src_heatmaps_base_path {src_heatmaps_base_path}")

    # load all the paths to the desired folder, containing all the saliency maps, like:
    # > folder_name
    #   - ANGRY_canonical.py
    #   - ANGRY_disgust.py
    #   - DISGUST_canonical.py
    #   ...
    # Gradcam
    base_paths = {
        "Bubbles": os.path.join(src_heatmaps_base_path, "Bubbles"),
        "EXTERNAL": os.path.join(src_heatmaps_base_path, "EXTERNAL"),
        "GRADCAM": os.path.join(src_heatmaps_base_path, "GRADCAM"),
    }

    old_paths_to_new_paths = {}
    for xai_method_name, base_path in base_paths.items():
        for root, folders, files in os.walk(base_path):
            # when you meet a folder named heatmap that contains npy heatmaps
            if os.path.basename(root).lower() in MODEL_NAMES:
                # extract the upper path (just ./.. above heatmaps)
                new_path = root.replace(src_heatmaps_folder_basename, f"{dest_heatmaps_folder_basename}")
                old_paths_to_new_paths[root] = f"{new_path}.png"

    i = 1
    for old_path, new_path in old_paths_to_new_paths.items():
        print(f"Processing path {i}")
        print(f"Old path: {old_path}")
        print(f"New path: {new_path}")
        print()
        i += 1
        compute_left_right(old_path, new_path)

    # Merge all generated images into a single image
    output_image_path = os.path.join(dest_heatmaps_folder_path, "merged_image.png")
    merge_generated_images(old_paths_to_new_paths, output_image_path, MODEL_NAMES, images_per_row=7, vertical=False)
    output_image_path = os.path.join(dest_heatmaps_folder_path, "merged_image_vertical.png")
    merge_generated_images(old_paths_to_new_paths, output_image_path, MODEL_NAMES, images_per_row=7, vertical=True)
