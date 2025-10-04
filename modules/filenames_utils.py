import os
from pathlib import Path
import re

def try_extract_model_or_user_name(heatmap_relative_path):
    """
        If human side, the path is like: "saliency_maps/results/subject/heatmaps/heatmap_1.npy"
        If machine side, TODO
    """
    # std_relative_map case
    if "std_saliency_map" in heatmap_relative_path:
        return "std_saliency_map"

    heatmap_relative_path_parts = list(Path(heatmap_relative_path).parts)
    h = heatmap_relative_path_parts

    try:
        if re.fullmatch(r"[A-Z]{6}", h[-3]) and h[-2] == "heatmaps":
            return h[-3]
        else:
            return None
    except: 
        return None

def get_emotion_from_heatmap_relpath(heatmap_relative_path):
    """
    Extracts the emotion from the heatmap filename.
    For now, assumes the filename format is like "DISGUST_Neutral.npy" or "DISGUST_canonical.npy".
    """
    # Case 1) saliency_maps\std_saliency_map_ones.npy or saliency_maps\std_saliency_map_zeroes.npy
    if "std_saliency_map" in heatmap_relative_path:
        return heatmap_relative_path.split('/')[-1].split('_')[-1].split('.')[0].upper()

    # Case 2) e.g.: saliency_maps\human_Results\AGAPIG\heatmaps\ANGRY_Disgust.npy
    caps_heatmap_relative_path = heatmap_relative_path.upper()
    emotion_present = False
    if "ANGRY" in caps_heatmap_relative_path:
        emotion_present = True
    elif "DISGUST" in caps_heatmap_relative_path:
        emotion_present = True
    elif "FEAR" in caps_heatmap_relative_path:
        emotion_present = True
    elif "HAPPY" in caps_heatmap_relative_path:
        emotion_present = True
    elif "NEUTRAL" in caps_heatmap_relative_path:
        emotion_present = True
    elif "SAD" in caps_heatmap_relative_path:
        emotion_present = True
    elif "SURPRISE" in caps_heatmap_relative_path:
        emotion_present = True

    heatmap_relative_path_parts = list(Path(heatmap_relative_path).parts)
    heatmap_fname = heatmap_relative_path_parts[-1]
    emotion_full = heatmap_fname.split('.')[0]

    if emotion_present:
        return emotion_full

    parts = heatmap_fname.split('_')
    if len(parts) != 2:
        # raise ValueError(f"Heatmap filename {heatmap_fname} does not follow the expected format.")
        return None
    if parts[0].upper() not in ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]:
        # raise ValueError(f"Unknown emotion {parts[0]} in heatmap filename {heatmap_fname}.")
        return None

    return emotion_full

def get_emotion_full_from_path(heatmap_relative_path, crash_on_error=True):
    regex = r"^(([A-Z]+)_(\w+))\.npy$"
    emotion_gt_pred_match = re.match(regex, os.path.basename(heatmap_relative_path))

    if emotion_gt_pred_match:
        emotion_gt_pred = emotion_gt_pred_match.group(1)  # e.g. ANGRY_canonical
        return emotion_gt_pred
    else:
        error_msg = f"  >> Error: heatmap file '{heatmap_relative_path}' does not match expected naming convention."
        if crash_on_error:
            raise ValueError(error_msg)
        print(error_msg)

def reformat_bad_emotion_gtpred_name(emotion_gtpred_name):
    """
    Reformat known bad emotion names to correct ones. (e.g. ANGRY_canonical -> ANGRY_ANGRY; ANGRY_happiness -> ANGRY_HAPPY)
    Returns something like "ANGRY_ANGRY" or "DISGUST_HAPPY"
    """
    emotion_gtpred_name = emotion_gtpred_name.upper()

    gt = emotion_gtpred_name.split('_')[0]
    pred = emotion_gtpred_name.split('_')[1]

    if pred == "CANONICAL":
        pred = gt

    corrections = {
        "NEUTRAL": "NEUTRAL",
        "HAPPINESS": "HAPPY",
        "HAPPY": "HAPPY",
        "SADNESS": "SAD",
        "SAD": "SAD",
        "ANGER": "ANGRY",
        "ANGRY": "ANGRY",
        "FEAR": "FEAR",
        "DISGUST": "DISGUST",
        "SURPRISE": "SURPRISE"
    }

    if pred not in corrections:
        raise ValueError(f"Unknown predicted emotion '{pred}' in '{emotion_gtpred_name}'. Expected one of: {', '.join(sorted(corrections.keys()))}")

    pred = corrections[pred]
    return f"{gt}_{pred}"