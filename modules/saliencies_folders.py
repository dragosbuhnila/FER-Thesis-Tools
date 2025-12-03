# NOTE: these paths are to be joined with the (saliency maps) base directory  
import os

# 0) Important folders
BASE_DIR = os.path.join(".", "saliency_maps")

# 1) Saliency folders (mostly aggregated ones)
saliencies_folders_rel_paths = {
    # 1a) Single person 1 (first two are best, last is worst)
    "fedmar":           r"human_Results\FEDMAR\heatmaps",
    "marfro":           r"human_Results\MARFRO\heatmaps",
    "matvin":           r"human_Results\MATVIN\heatmaps",

    # 1b) Single person 2 (3 at 76%, 6 at 75%, 1 at 74%)
    # 76%
    "fedama":           r"human_Results\FEDAMA\heatmaps",
    "dalgon":           r"human_Results\DALGON\heatmaps",
    "paorus":           r"human_Results\PAORUS\heatmaps",
    # 75%
    "fraghi":          r"human_Results\FRAGHI\heatmaps",
    "giugui":          r"human_Results\GIUGUI\heatmaps",
    "mirtek":          r"human_Results\MIRTEK\heatmaps",
    "silfer":          r"human_Results\SILFER\heatmaps",
    "valcol":          r"human_Results\VALCOL\heatmaps",
    "beafra":          r"human_Results\BEAFRA\heatmaps",
    # 74%
    "edodon":          r"human_Results\EDODON\heatmaps",

    # 2) Aggregated persons
    "men":              r"human_Mappe numpy\mappe aggregate\heatmaps_maschi",
    "women":            r"human_Mappe numpy\mappe aggregate\heatmaps_femmine",
    "best":             r"human_Mappe numpy\mappe aggregate\heatmaps_codasup",
    "worst":            r"human_Mappe numpy\mappe aggregate\heatmaps_codainf",

    # 3) Bubbles
    "convnext_bub":     r"canonical\bubbles\bubbles\ConvNeXt\bubbles_adele",
    "efficientnet_bub": r"canonical\bubbles\bubbles\EfficientNetB1\bubbles_adele",
    "inceptionv3_bub":  r"canonical\bubbles\bubbles\InceptionV3\bubbles_adele",
    "pattlite_bub":     r"canonical\bubbles\bubbles\PattLite\bubbles_adele",
    "resnet_bub":       r"canonical\bubbles\bubbles\ResNet\bubbles_adele",
    "vgg19_bub":        r"canonical\bubbles\bubbles\VGG19\bubbles_adele",
    "yolo_bub":         r"canonical\bubbles\bubbles\YOLO\bubbles_adele",

    # 4) External
    "convnext_ext":     r"HEATMAPS\EXTERNAL\ConvNeXt",
    "efficientnet_ext": r"HEATMAPS\EXTERNAL\EfficientNetB1",
    "inceptionv3_ext":  r"HEATMAPS\EXTERNAL\InceptionV3",
    "pattlite_ext":     r"HEATMAPS\EXTERNAL\PattLite",
    "resnet_ext":       r"HEATMAPS\EXTERNAL\ResNet",
    "vgg19_ext":        r"HEATMAPS\EXTERNAL\VGG19",
    "yolo_ext":         r"HEATMAPS\EXTERNAL\YOLO",

    # 5) Grad
    "convnext_grad":    r"HEATMAPS\GRADCAM\ConvNeXt", # Come è possibile che non ci siano i layer ma tutto insieme?????? wtf
    "efficientnet_grad":r"HEATMAPS\GRADCAM\EfficientNetB1",
    "inceptionv3_grad": r"HEATMAPS\GRADCAM\InceptionV3",
    "pattlite_grad":    r"HEATMAPS\GRADCAM\PattLite",
    "resnet_grad":      r"HEATMAPS\GRADCAM\ResNet",
    "vgg19_grad":       r"HEATMAPS\GRADCAM\VGG19",
    "yolo_grad":        r"HEATMAPS\GRADCAM\YOLO",
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