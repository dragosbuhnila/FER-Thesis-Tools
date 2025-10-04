# NOTE: these paths are to be joined with the (saliency maps) base directory  
import os


saliencies_folders_rel_paths = {
    # 1) Single person (first two are best, last is worst)
    "fedmar":           r"human_Results\FEDMAR\heatmaps",
    "marfro":           r"human_Results\MARFRO\heatmaps",
    "matvin":           r"human_Results\MATVIN\heatmaps",

    # 2) Aggregated persons
    "men":              r"human_Mappe numpy\mappe aggregate\heatmaps_maschi",
    "women":            r"human_Mappe numpy\mappe aggregate\heatmaps_femmine",
    "best":             r"human_Mappe numpy\mappe aggregate\heatmaps_codasup",
    "worst":            r"human_Mappe numpy\mappe aggregate\heatmaps_codainf",

    # 3) Neural Network (ConvNext)
    "convnext_bub":     r"canonical\bubbles\bubbles\ConvNeXt\bubbles_adele",
    "convnext_ext":     r"HEATMAPS\EXTERNAL\ConvNeXt",
    "convnext_grad":    r"HEATMAPS\GRADCAM\ConvNeXt", # Come è possibile che non ci siano i layer ma tutto insieme?????? wtf
}

TESTERS_NAME_SETS_PATH = os.path.join("saliency_maps", "human_Results", "testers_name_sets.pkl")
testers_name_sets = {}
with open(TESTERS_NAME_SETS_PATH, "rb") as f:
    import pickle
    testers_name_sets = pickle.load(f)
