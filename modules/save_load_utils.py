import os
import pickle


def get_stats_cache_path(heatmap_relpath, stat_names: str, weigh_roi_overlap: bool, separate_lr: bool, roi_type: str):
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

    filename = f"{heatmap_fname}_stat-{stat_names}"
    if weigh_roi_overlap:
        filename += "_weigh_roi_overlap"
    if separate_lr:
        filename += "_separate-lr"
    if roi_type not in ["aus", "faceparts"]:
        raise ValueError(f"Invalid roi_type: {roi_type}. Expected 'aus' or 'faceparts'.")
    if roi_type == "faceparts":
        filename += "_faceparts"
    filename += ".pkl"

    return os.path.join(cache_dir, filename)

def save_statistics(heatmap_relpath, statistics, weigh_roi_overlap: bool, separate_lr: bool, roi_type: str):
    stat_names = [stat_name for stat_name in next(iter(statistics.values())).keys()]
    # print(f"stat_names are {stat_names}")
    path = get_stats_cache_path(heatmap_relpath, "-".join(stat_names), weigh_roi_overlap, separate_lr, roi_type)
    with open(path, "wb") as f:
        pickle.dump(statistics, f)
    print(f"Statistics saved to {path}")

def load_statistics(heatmap_relpath, stat_names, weigh_roi_overlap: bool, separate_lr: bool, roi_type: str):
    path = get_stats_cache_path(heatmap_relpath, stat_names, weigh_roi_overlap, separate_lr, roi_type)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None