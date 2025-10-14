# NOTE: these paths are to be joined with the (saliency maps) base directory  
import os

# 0) Important folders
BASE_DIR = os.path.join(".", "saliency_maps")

# 1) Saliency folders (mostly aggregated ones)
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

# 2) Tester groups (used for saliencies)
TESTERS_NAME_SETS_PATH = os.path.join(BASE_DIR, "human_Results", "testers_name_sets.pkl")
testers_name_sets = {}
with open(TESTERS_NAME_SETS_PATH, "rb") as f:
    import pickle
    testers_name_sets = pickle.load(f)
    # Looks something like:
    # {'best': ['FEDMAR', 'MARFRO', 'MATVIN'],
    #  'worst': ['CODAINF', 'CODASUP', 'LUCAMO'],
    #  'men': ['FEDMAR', 'MARFRO', 'MATVIN'],
    #  'women': ['FEDMAR', 'MARFRO', 'MATVIN'],
    # }


# 3) Test set paths
OCCLUDED_TEST_SET_PATH = os.path.join(BASE_DIR, "zzz_other_and_zips", "output_occlusions", "output_images_testset", "bosphorus_test_HQ")
OCCLUDED_TEST_SET_H5_PATH = os.path.join(BASE_DIR, "zzz_other_and_zips", "h5_files", "occluded_test_set.h5")
OCCLUDED_TEST_SET_RESIZED_PATH = os.path.join(BASE_DIR, "zzz_other_and_zips", "output_occlusions", "output_images_testset_resized")

# 4) Model paths
FINETUNING_MODELS_FOLDER = os.path.join(BASE_DIR, "drive_federica", "model", "finetuning")
ALL_MODELS_PATHS = {
    "resnet_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_ResNet_finetuning"),
    "pattlite_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_PattLite_finetuning"),
    "vgg19_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_VGG19_finetuning"),
    "inceptionv3_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_InceptionV3_finetuning"),
    "convnext_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_ConvNeXt_finetuning"),
    # "efficientnet_finetuning": os.path.join(FINETUNING_WEIGHTS_FOLDER, "pretrained_EfficientNetB1_finetuning"),
}