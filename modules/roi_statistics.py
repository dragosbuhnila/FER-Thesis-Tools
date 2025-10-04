import numpy as np

from modules.landmark_utils import ROI_ORDER_FACEPARTS

def roi_mean(masked_heatmap, pxbypx_weightmap=None, debug=False):
    """
    Compute the mean value of the masked heatmap.
    Args:
        masked_heatmap (np.ndarray): The masked heatmap array.
    Returns:
        dict: A dictionary containing the mean value.
        example: {"mean": 0.1234}
    """
    if pxbypx_weightmap is None:
        pxbypx_weightmap = np.ones_like(masked_heatmap)

    heatmap_no_nan = np.nan_to_num(masked_heatmap, nan=0.0)
    heatmap_no_nan_wtd = heatmap_no_nan * pxbypx_weightmap

    sum = np.sum(heatmap_no_nan_wtd)

    count = np.count_nonzero(~np.isnan(masked_heatmap))
    mean_value = sum / count if count > 0 else 0

    if debug:
        print(f"roi_mean: sum={sum}, count={count}, mean_value={mean_value}")  # Debug output

    return {"mean": mean_value}

def compare_dif(stats1, stats2):
    """
    Compare two statistics dictionaries and return the differences.
    Args:
        stats1 (dict): First statistics dictionary.
        stats2 (dict): Second statistics dictionary.
    Returns:
        dict: A dictionary containing the differences.
    """
    
    if len(stats1) != len(stats2):
        raise ValueError(f"Statistics dictionaries must have the same length for comparison, instead found {len(stats1)} and {len(stats2)}: first->{stats1.keys()} second->{stats2.keys()}")
    if not stats1 or not stats2 or len(stats1) == 0 or len(stats2) == 0:
        raise ValueError("Statistics dictionaries cannot be empty for comparison.")
    if stats1.keys() != stats2.keys():
        raise ValueError(f"Statistics dictionaries must have the same keys for comparison, instead found: first->{stats1.keys()} and second->{stats2.keys()}.")

    differences = {}
    for au in stats1.keys():
        # let error be raised if au not in stats2
        s1 = stats1[au]
        s2 = stats2[au] 

        if len(s1) != 1:
            raise ValueError(f"With 'compare_dif()', statistics must be a single value, but got {len(s1)} values in AU {au}.")
        if len(s1) != len(s2):
            raise ValueError(f"Statistics for AU {au} must have the same length in both dictionaries.")
        
        s1 = next(iter(s1.values()))
        s2 = next(iter(s2.values()))
        differences[au] = abs(s1 - s2)

    return differences

def compare_meandif(stats1, stats2):
    """
    Compare the mean of the differences of means.
    Args:
        stats1 (dict): First statistics dictionary.
        stats2 (dict): Second statistics dictionary.
    Returns:
        dict: A dictionary containing the differences in mean values.
    """
    differences = compare_dif(stats1, stats2)
    mean_diff = sum(differences.values()) / len(differences) if differences else 0
    
    return mean_diff

def compare_difmean(stats1, stats2):
    """
    Compare the difference of means of means. I.e. stats1 contains a dictionary of means, so you take the mean of that.
    Args:
        stats1 (dict): First statistics dictionary.
        stats2 (dict): Second statistics dictionary.
    Returns:
        dict: A dictionary containing the differences.
    """
    subject1 = next(iter(stats1.keys()))
    subject2 = next(iter(stats2.keys()))

    if len(stats1) != len(stats2):
        raise ValueError(f"Statistics dictionaries must have the same length for comparison, instead found {len(stats1)} and {len(stats2)}. First->{stats1.keys()} Second->{stats2.keys()}")
    if not stats1 or not stats2 or len(stats1) == 0 or len(stats2) == 0:
        raise ValueError("Statistics dictionaries cannot be empty for comparison.")
    if stats1.keys() != stats2.keys():
        raise ValueError(f"Statistics dictionaries must have the same keys for comparison, instead found: first->{stats1.keys()} and second->{stats2.keys()}.")

    if len(next(iter(stats1.values()))) != 1:
        raise ValueError("With 'compare_difmean()', statistics must be a single value, but got multiple values in the statistics dictionaries.")
    if len(next(iter(stats2.values()))) != 1:
        raise ValueError("With 'compare_difmean()', statistics must be a single value, but got multiple values in the statistics dictionaries.")

    mean1 = np.mean([next(iter(v.values())) for v in stats1.values()])
    mean2 = np.mean([next(iter(v.values())) for v in stats2.values()])

    return abs(mean1 - mean2)

def convert_faceparts_roi_means_from_dict_to_vector(stats_dict):
    """
    Convert a stats dictionary like {'Left Eyebrow': {'mean': 0.3555507771417289}, 'Right Eyebrow': {'mean': 0.10694237923453793}, ...}
    to a vector like [0.3555507771417289, 0.10694237923453793, ...] in a fixed order of ROIs.
    """
    
    vector = []
    for roi in ROI_ORDER_FACEPARTS:
        if roi in stats_dict:
            vector.append(stats_dict[roi]['mean'])
        else:
            raise ValueError(f"ROI '{roi}' not found in stats dictionary keys: {list(stats_dict.keys())}")
    return np.array(vector)