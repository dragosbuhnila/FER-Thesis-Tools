# NOTE: these paths are to be joined with the (saliency maps) base directory  
import os

# confronto tra il soggetto con la più alta performance (FEDMAR e MARFRO, a turno) versus soggetto con la performance più bassa;
# confronto maschi vs femmine;
# confronto upper tail vs lower tail;
# confronto Bubbles ConvNext versus soggetto con medesima performance di 78% (a turno FEDMAR e MARFRO)
# confronto External Perturbation ConvNext versus soggetto con medesima performance di 78% (a turno FEDMAR e MARFRO)
# confronto GradCam LAYER30 ConvNext versus soggetto con medesima performance di 78% (a turno FEDMAR e MARFRO)

# tutti i soggetti che hanno ottenuto accuratezze 76% (3 persone), 75% (6 persone) e 74% (1 persona), a confronto con 
# tutte le reti che hanno ottenuto le medesime accuratezze (rispettivamente 4, 1, 1), 
# in tutte le combinazioni possibili tra uomo e rete. Questo per Bubbles, Gradcam e External. 
# Dovrebbero essere in totale 19x3 confronti se ho fatto bene i conti.


# 0) Important folders
BASE_DIR = os.path.join(".", "saliency_maps")
HEATMAPS_FEDE_DIR_BASENAME = "HEATMAPS_fede"
HEATMAPS_OCCFT_DIR_BASENAME = "HEATMAPS_occft"
HEATMAPS_ALE_DIR_BASENAME = "HEATMAPS_ale"

HEATMAPS_ALE_SINGLE_HUMANS_DIR_PATH = os.path.join(BASE_DIR, HEATMAPS_ALE_DIR_BASENAME, "Results")

RANKING_PHASE2_FILE_PATH = os.path.join(BASE_DIR, HEATMAPS_ALE_DIR_BASENAME, "testers_ranking.pkl")
GENDER_PHASE2_FILE_PATH =  os.path.join(BASE_DIR, HEATMAPS_ALE_DIR_BASENAME, "testers_gender.txt")
SETS_PHASE2_FILE_PATH =    os.path.join(BASE_DIR, HEATMAPS_ALE_DIR_BASENAME, "testers_name_sets.pkl")

# 1) Saliency folders (mostly aggregated ones)
saliencies_folders_rel_paths_phase2 = {
    # 1a) Single person 1 (for best human vs worst human)
    "rebleo":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\REBLEO\heatmaps", # 68.94%
    "krijak":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\KRIJAK\heatmaps", # 46.72%
    "lucrug":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\LUCRUG\heatmaps", # 23.36%


    # 1b) Single person 2 with ranges (should look for accs at 0.7000 0.6886 0.6800 0.6771 0.6771 0.6657 0.6343 => 70%, 69%, 68%, 67%, 63%)

    # 70% (VGG19)
    # "rebleo":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\REBLEO\heatmaps", # 68.94% --- A LITTLE BIT OUTSIDE OF RANGE

    # 69% (YOLO)        
    # "rebleo":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\REBLEO\heatmaps", # 68.94%  # keep commented for knowing it's here but don't break the dict with dupes
    
    # 68% (MobileNet, Convnext, Inception)   
    "amaesc":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\AMAESC\heatmaps", # 68.37%
    "silven":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\SILVEN\heatmaps", # 68.09%
    "fedmar":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\FEDMAR\heatmaps", # 68.09%
    "mardib":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\MARDIB\heatmaps", # 67.80%

    # 67% (ResNet) 
    "alitom":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\ALITOM\heatmaps", # 67.24%
    "marcec":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\MARCEC\heatmaps", # 67.24%

    # 63% (EfficientNet)
    "roclab":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\ROCLAB\heatmaps", # 63.25%



    # # 1b) Single person 2 with ranges (should look for accs at 0.7000 0.6886 0.6800 0.6771 0.6771 0.6657 0.6343 => 70%, 69%, 68%, 67%, 63%)

    # # 70.00% (VGG19)                                                            [ range: 69.50% - 70.50% ]
    # # "rebleo":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\REBLEO\heatmaps", # 68.94% --- A LITTLE BIT OUTSIDE OF RANGE

    # # 68.86% (YOLO)                                                             [ range: 68.36% - 69.36% ]
    # # "rebleo":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\REBLEO\heatmaps", # 68.94%  # keep commented for knowing it's here but don't break the dict with dupes
    
    # # 68.00% (MobileNet)                                                        [ range: 67.50% - 68.50% ]
    # "amaesc":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\AMAESC\heatmaps", # 68.37%
    # "silven":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\SILVEN\heatmaps", # 68.09%
    # "fedmar":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\FEDMAR\heatmaps", # 68.09%
    # "mardib":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\MARDIB\heatmaps", # 67.80%

    # # 67.71% (Convnext, Inception)                                              [ range: 67.21% - 68.21% ]
    # # "silven":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\SILVEN\heatmaps", # 68.09%
    # # "fedmar":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\FEDMAR\heatmaps", # 68.09%
    # # "mardib":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\MARDIB\heatmaps", # 67.80%
    # "alitom":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\ALITOM\heatmaps", # 67.24%
    # "marcec":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\MARCEC\heatmaps", # 67.24%

    # # 66.57% (ResNet)                                                           [ range: 66.07% - 67.07% ]
    # "filgua":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\FILGUA\heatmaps", # 66.10%

    # # 63.43% (EfficientNet)                                                     [ range: 62.93% - 63.93% ]
    # "fabiac":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\FABIAC\heatmaps", # 63.53%
    # "roclab":                   rf"{HEATMAPS_ALE_DIR_BASENAME}\Results\ROCLAB\heatmaps", # 63.25%



    # 2) Aggregated persons
    "men":                      rf"{HEATMAPS_ALE_DIR_BASENAME}\AGGREGATE\maschi\heatmaps",
    "women":                    rf"{HEATMAPS_ALE_DIR_BASENAME}\AGGREGATE\femmine\heatmaps",
    "best":                     rf"{HEATMAPS_ALE_DIR_BASENAME}\AGGREGATE\upper\heatmaps",
    "worst":                    rf"{HEATMAPS_ALE_DIR_BASENAME}\AGGREGATE\lower\heatmaps",


    # 3) Bubbles
    "occft_convnext_bub":       rf"{HEATMAPS_OCCFT_DIR_BASENAME}\Bubbles\occft_convnext",
    "occft_efficientnet_bub":   rf"{HEATMAPS_OCCFT_DIR_BASENAME}\Bubbles\occft_efficientnetb1",
    "occft_inceptionv3_bub":    rf"{HEATMAPS_OCCFT_DIR_BASENAME}\Bubbles\occft_inceptionv3",
    "occft_pattlite_bub":       rf"{HEATMAPS_OCCFT_DIR_BASENAME}\Bubbles\occft_pattlite",
    "occft_resnet_bub":         rf"{HEATMAPS_OCCFT_DIR_BASENAME}\Bubbles\occft_resnet",
    "occft_vgg19_bub":          rf"{HEATMAPS_OCCFT_DIR_BASENAME}\Bubbles\occft_vgg19",
    "occft_yolo_bub":           rf"{HEATMAPS_OCCFT_DIR_BASENAME}\Bubbles\occft_yolo",

    # 4) External   
    "occft_convnext_ext":       rf"{HEATMAPS_OCCFT_DIR_BASENAME}\EXTERNAL\occft_convneXt",
    "occft_efficientnet_ext":   rf"{HEATMAPS_OCCFT_DIR_BASENAME}\EXTERNAL\occft_efficientNetB1",
    "occft_inceptionv3_ext":    rf"{HEATMAPS_OCCFT_DIR_BASENAME}\EXTERNAL\occft_inceptionV3",
    "occft_pattlite_ext":       rf"{HEATMAPS_OCCFT_DIR_BASENAME}\EXTERNAL\occft_pattlite",
    "occft_resnet_ext":         rf"{HEATMAPS_OCCFT_DIR_BASENAME}\EXTERNAL\occft_resnet",
    "occft_vgg19_ext":          rf"{HEATMAPS_OCCFT_DIR_BASENAME}\EXTERNAL\occft_vgg19",
    "occft_yolo_ext":           rf"{HEATMAPS_OCCFT_DIR_BASENAME}\EXTERNAL\occft_yolo",

    # 5) Grad   (last layer)
    "occft_convnext_grad":      rf"{HEATMAPS_OCCFT_DIR_BASENAME}\GRADCAM\occft_convneXt",
    "occft_efficientnet_grad":  rf"{HEATMAPS_OCCFT_DIR_BASENAME}\GRADCAM\occft_efficientNetB1",
    "occft_inceptionv3_grad":   rf"{HEATMAPS_OCCFT_DIR_BASENAME}\GRADCAM\occft_inceptionV3",
    "occft_pattlite_grad":      rf"{HEATMAPS_OCCFT_DIR_BASENAME}\GRADCAM\occft_pattlite",
    "occft_resnet_grad":        rf"{HEATMAPS_OCCFT_DIR_BASENAME}\GRADCAM\occft_resnet",
    "occft_vgg19_grad":         rf"{HEATMAPS_OCCFT_DIR_BASENAME}\GRADCAM\occft_vgg19",
    "occft_yolo_grad":          rf"{HEATMAPS_OCCFT_DIR_BASENAME}\GRADCAM\occft_yolo",
}    


saliencies_folders_rel_paths_phase1 = {
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
    "convnext_ext":     rf"{HEATMAPS_FEDE_DIR_BASENAME}\EXTERNAL\ConvNeXt",
    "efficientnet_ext": rf"{HEATMAPS_FEDE_DIR_BASENAME}\EXTERNAL\EfficientNetB1",
    "inceptionv3_ext":  rf"{HEATMAPS_FEDE_DIR_BASENAME}\EXTERNAL\InceptionV3",
    "pattlite_ext":     rf"{HEATMAPS_FEDE_DIR_BASENAME}\EXTERNAL\PattLite",
    "resnet_ext":       rf"{HEATMAPS_FEDE_DIR_BASENAME}\EXTERNAL\ResNet",
    "vgg19_ext":        rf"{HEATMAPS_FEDE_DIR_BASENAME}\EXTERNAL\VGG19",
    "yolo_ext":         rf"{HEATMAPS_FEDE_DIR_BASENAME}\EXTERNAL\YOLO",

    # 5) Grad (last layer)
    "convnext_grad":    rf"{HEATMAPS_FEDE_DIR_BASENAME}\GRADCAM\ConvNeXt", 
    "efficientnet_grad":rf"{HEATMAPS_FEDE_DIR_BASENAME}\GRADCAM\EfficientNetB1",
    "inceptionv3_grad": rf"{HEATMAPS_FEDE_DIR_BASENAME}\GRADCAM\InceptionV3",
    "pattlite_grad":    rf"{HEATMAPS_FEDE_DIR_BASENAME}\GRADCAM\PattLite",
    "resnet_grad":      rf"{HEATMAPS_FEDE_DIR_BASENAME}\GRADCAM\ResNet",
    "vgg19_grad":       rf"{HEATMAPS_FEDE_DIR_BASENAME}\GRADCAM\VGG19",
    "yolo_grad":        rf"{HEATMAPS_FEDE_DIR_BASENAME}\GRADCAM\YOLO",
}

# 2) Tester groups (used for saliencies)
TESTERS_NAME_SETS_PHASE1_PATH = os.path.join(BASE_DIR, "human_Results", "testers_name_sets.pkl")
testers_name_sets_phase1 = {}
with open(TESTERS_NAME_SETS_PHASE1_PATH, "rb") as f:
    import pickle
    testers_name_sets_phase1 = pickle.load(f)
    # Looks something like:
    # {'best': ['FEDMAR', 'MARFRO', 'MATVIN'],
    #  'worst': ['CODAINF', 'CODASUP', 'LUCAMO'],
    #  'men': ['FEDMAR', 'MARFRO', 'MATVIN'],
    #  'women': ['FEDMAR', 'MARFRO', 'MATVIN'],
    # }

testers_name_sets_phase2 = {}
with open(SETS_PHASE2_FILE_PATH, "rb") as f:
    import pickle
    testers_name_sets_phase2 = pickle.load(f)


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