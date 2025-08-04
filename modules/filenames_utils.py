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

    if re.fullmatch(r"[A-Z]{6}", h[-3]) and h[-2] == "heatmaps":
        return h[-3]
    else:
        raise ValueError(f"not developed machine part yet: {heatmap_relative_path}")

def get_emotion_from_heatmap_relpath(heatmap_relative_path):
    """
    Extracts the emotion from the heatmap filename.
    For now, assumes the filename format is like "DISGUST_Neutral.npy" or "DISGUST_canonical.npy".
    """
    if "std_saliency_map" in heatmap_relative_path:
        return heatmap_relative_path.split('/')[-1].split('_')[-1].split('.')[0].upper()

    heatmap_relative_path_parts = list(Path(heatmap_relative_path).parts)
    heatmap_fname = heatmap_relative_path_parts[-1]

    parts = heatmap_fname.split('_')
    if len(parts) != 2:
        raise ValueError(f"Heatmap filename {heatmap_fname} does not follow the expected format.")
    if parts[0].upper() not in ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]:
        raise ValueError(f"Unknown emotion {parts[0]} in heatmap filename {heatmap_fname}.")
    return parts[0]