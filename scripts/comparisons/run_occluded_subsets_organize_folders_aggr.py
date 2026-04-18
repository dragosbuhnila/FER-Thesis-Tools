import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import subprocess

from modules.filenames_utils import CANONICAL_SUBSETS

# Define paths
PYTHON_EXE = os.path.abspath("./.venv/Scripts/python.exe")
SCRIPT = os.path.abspath("./compare_canonicals.py")
MERGING_SCRIPT = os.path.abspath("./scripts/make_images_grid.py")
METHOD = 4 # 4 - organize_folders_aggregated
PHASE = 2

# Define comparisons 55 total (4 + 3 + 3 + 36 + 6 + 3 = 55)
comparisons = [ 
    # # 0) Base comparisons (best (69%) vs worst (47%, 23%) individuals, gender aggregated, tails aggregated)
    # ("rebleo", "krijak"),
    # ("rebleo", "lucrug"),
    # ("men", "women"),
    # ("best", "worst"),

    # 1) 70% (1 network vs 1 person)  (NOTE that REBLEO is 69% but I want to compare at least one person with VGG19)
    # _________________________________________________________________
    # occft_VGG19       vs      rebleo
    ("occft_vgg19_grad", "rebleo"),
    ("occft_vgg19_bub", "rebleo"),
    ("occft_vgg19_ext", "rebleo"),

    # 2) 69% (1 network vs 1 person)
    # _________________________________________________________________
    # occft_YOLO        vs      rebleo
    ("occft_yolo_grad", "rebleo"),
    ("occft_yolo_bub", "rebleo"),
    ("occft_yolo_ext", "rebleo"),

    # 3) 68% (3 network vs 4 persons)
    # _________________________________________________________________
    # occft_MobileNet    vs      amaesc
    ("occft_pattlite_grad", "amaesc"),
    ("occft_pattlite_bub", "amaesc"),
    ("occft_pattlite_ext", "amaesc"),
    # _________________________________________________________________
    # occft_MobileNet    vs      silven
    ("occft_pattlite_grad", "silven"),
    ("occft_pattlite_bub", "silven"),
    ("occft_pattlite_ext", "silven"),
    # _________________________________________________________________
    # occft_MobileNet    vs      fedmar
    ("occft_pattlite_grad", "fedmar"),
    ("occft_pattlite_bub", "fedmar"),
    ("occft_pattlite_ext", "fedmar"),
    # _________________________________________________________________
    # occft_MobileNet    vs      mardib
    ("occft_pattlite_grad", "mardib"),
    ("occft_pattlite_bub", "mardib"),
    ("occft_pattlite_ext", "mardib"),
    # _________________________________________________________________
    # occft_ConvNeXt    vs      amaesc
    ("occft_convnext_grad", "amaesc"),
    ("occft_convnext_bub", "amaesc"),
    ("occft_convnext_ext", "amaesc"),
    # _________________________________________________________________
    # occft_ConvNeXt    vs      silven
    ("occft_convnext_grad", "silven"),
    ("occft_convnext_bub", "silven"),
    ("occft_convnext_ext", "silven"),
    # _________________________________________________________________
    # occft_ConvNeXt    vs      fedmar
    ("occft_convnext_grad", "fedmar"),
    ("occft_convnext_bub", "fedmar"),
    ("occft_convnext_ext", "fedmar"),
    # _________________________________________________________________
    # occft_ConvNeXt    vs      mardib
    ("occft_convnext_grad", "mardib"),
    ("occft_convnext_bub", "mardib"),
    ("occft_convnext_ext", "mardib"),
    # _________________________________________________________________
    # occft_Inception   vs      amaesc
    ("occft_inceptionv3_grad", "amaesc"),
    ("occft_inceptionv3_bub", "amaesc"),
    ("occft_inceptionv3_ext", "amaesc"),
    # _________________________________________________________________
    # occft_Inception   vs      silven
    ("occft_inceptionv3_grad", "silven"),
    ("occft_inceptionv3_bub", "silven"),
    ("occft_inceptionv3_ext", "silven"),
    # _________________________________________________________________
    # occft_Inception   vs      fedmar
    ("occft_inceptionv3_grad", "fedmar"),
    ("occft_inceptionv3_bub", "fedmar"),
    ("occft_inceptionv3_ext", "fedmar"),
    # _________________________________________________________________
    # occft_Inception   vs      mardib
    ("occft_inceptionv3_grad", "mardib"),
    ("occft_inceptionv3_bub", "mardib"),
    ("occft_inceptionv3_ext", "mardib"),

    # 4) 67% (1 network vs 2 persons)
    # _________________________________________________________________
    # occft_ResNet      vs      alitom
    ("occft_resnet_grad", "alitom"),
    ("occft_resnet_bub", "alitom"),
    ("occft_resnet_ext", "alitom"),
    # _________________________________________________________________
    # occft_ResNet      vs      marcec
    ("occft_resnet_grad", "marcec"),
    ("occft_resnet_bub", "marcec"),
    ("occft_resnet_ext", "marcec"),

    # 5) 63% (1 network vs 1 person)
    # occft_EfficientNet vs     roclab
    ("occft_efficientnet_grad", "roclab"),
    ("occft_efficientnet_bub", "roclab"),
    ("occft_efficientnet_ext", "roclab")
]

print(f"Total comparisons to run: {len(comparisons)}")

for subset in CANONICAL_SUBSETS:
    #     "negative_ANGRY",
    # "negative_DISGUST",
    # "negative_FEAR",
    if "match" in subset or "negative_ANGRY" in subset or "negative_DISGUST" in subset or "negative_FEAR" in subset or "negative_HAPPY" in subset:
        # skip cause it crashed and this was already computed
        continue
    for subject1, subject2 in comparisons:
        print(f"Running comparison: {subject1} vs {subject2} (subset: {subset})...")
        subprocess.run([PYTHON_EXE, SCRIPT, str(METHOD), str(PHASE), subject1, subject2, subset], check=True)

    print(f"All comparisons completed for subset {subset}.")