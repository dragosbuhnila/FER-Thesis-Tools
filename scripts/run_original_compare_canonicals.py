import subprocess
import os

# Define paths
PYTHON_EXE = os.path.abspath("./.venv/Scripts/python.exe")
SCRIPT = os.path.abspath("./compare_canonicals.py")
MERGING_SCRIPT = os.path.abspath("./scripts/merge_images_in_grid.py")
METHOD = 3 # 3 - compare_canonicals
PHASE = 1

# network names:
# "convnext_grad":    
# "efficientnet_grad":
# "inceptionv3_grad": 
# "pattlite_grad":    
# "resnet_grad":      
# "vgg19_grad":       
# "yolo_grad"

# Define comparisons: total = 11 + 36 + 18 + 3 = 68
comparisons = [
    # 0) First comparisons (decided in August): 
    # best model Convnext vs FEDMAR and MARFRO [78%] | 
    ("convnext_grad", "fedmar"),
    ("convnext_grad", "marfro"),
    ("convnext_bub", "fedmar"),
    ("convnext_bub", "marfro"),
    ("convnext_ext", "fedmar"),
    ("convnext_ext", "marfro"),
    # best vs worst individuals: FEDMAR [78%] vs MARFRO [78%] vs MATVIN [55%]
    ("fedmar", "matvin"),
    ("marfro", "matvin"),
    ("fedmar", "marfro"),
    # gender and tails (tails calculated with mean and stdev: upper 17 and lower 17)
    ("men", "women"),
    ("best", "worst"),

    # 1) 76% (4 networks vs 3 persons)
    # _________________________________________________________________
    # efficientnet      vs      fedama
    ("efficientnet_grad", "fedama"),
    ("efficientnet_bub", "fedama"),
    ("efficientnet_ext", "fedama"),
    # _________________________________________________________________
    # pattlite          vs      fedama
    ("pattlite_grad", "fedama"),
    ("pattlite_bub", "fedama"),
    ("pattlite_ext", "fedama"),
    # _________________________________________________________________
    # vgg19             vs      fedama
    ("vgg19_grad", "fedama"),
    ("vgg19_bub", "fedama"),
    ("vgg19_ext", "fedama"),
    # _________________________________________________________________
    # yolo              vs      fedama
    ("yolo_grad", "fedama"),
    ("yolo_bub", "fedama"),
    ("yolo_ext", "fedama"),
    # _________________________________________________________________
    # efficientnet      vs      dalgon
    ("efficientnet_grad", "dalgon"),
    ("efficientnet_bub", "dalgon"),
    ("efficientnet_ext", "dalgon"),
    # _________________________________________________________________
    # pattlite          vs      dalgon
    ("pattlite_grad", "dalgon"),
    ("pattlite_bub", "dalgon"),
    ("pattlite_ext", "dalgon"),
    # _________________________________________________________________
    # vgg19             vs      dalgon
    ("vgg19_grad", "dalgon"),
    ("vgg19_bub", "dalgon"),
    ("vgg19_ext", "dalgon"),
    # _________________________________________________________________
    # yolo              vs      dalgon
    ("yolo_grad", "dalgon"),
    ("yolo_bub", "dalgon"),
    ("yolo_ext", "dalgon"),
    # _________________________________________________________________
    # efficientnet      vs      paorus
    ("efficientnet_grad", "paorus"),
    ("efficientnet_bub", "paorus"),
    ("efficientnet_ext", "paorus"),
    # _________________________________________________________________
    # pattlite          vs      paorus
    ("pattlite_grad", "paorus"),
    ("pattlite_bub", "paorus"),
    ("pattlite_ext", "paorus"),
    # _________________________________________________________________
    # vgg19             vs      paorus
    ("vgg19_grad", "paorus"),
    ("vgg19_bub", "paorus"),
    ("vgg19_ext", "paorus"),
    # _________________________________________________________________
    # yolo              vs      paorus
    ("yolo_grad", "paorus"),
    ("yolo_bub", "paorus"),
    ("yolo_ext", "paorus"),

    # 2) 75% (1 network vs 6 persons)
    # _________________________________________________________________
    # inceptionv3       vs      fraghi
    ("inceptionv3_grad", "fraghi"),
    ("inceptionv3_bub", "fraghi"),
    ("inceptionv3_ext", "fraghi"),
    # _________________________________________________________________
    # inceptionv3       vs      giugui
    ("inceptionv3_grad", "giugui"),
    ("inceptionv3_bub", "giugui"),
    ("inceptionv3_ext", "giugui"),
    # _________________________________________________________________
    # inceptionv3       vs      mirtek
    ("inceptionv3_grad", "mirtek"),
    ("inceptionv3_bub", "mirtek"),
    ("inceptionv3_ext", "mirtek"),
    # _________________________________________________________________
    # inceptionv3       vs      silfer
    ("inceptionv3_grad", "silfer"),
    ("inceptionv3_bub", "silfer"),
    ("inceptionv3_ext", "silfer"),
    # _________________________________________________________________
    # inceptionv3       vs      valcol
    ("inceptionv3_grad", "valcol"),
    ("inceptionv3_bub", "valcol"),
    ("inceptionv3_ext", "valcol"),
    # _________________________________________________________________
    # inceptionv3       vs      beafra
    ("inceptionv3_grad", "beafra"),
    ("inceptionv3_bub", "beafra"),
    ("inceptionv3_ext", "beafra"),

    # 3) 74% (1 network vs 1 person)
    # _________________________________________________________________
    # resnet            vs      edodon
    ("resnet_grad", "edodon"),
    ("resnet_bub", "edodon"),
    ("resnet_ext", "edodon"),

]

print(f"Total comparisons to run: {len(comparisons)}")
exit(0)

# Run comparisons
for subject1, subject2 in comparisons:
    print(f"Running comparison: {subject1} vs {subject2}")
    subprocess.run([PYTHON_EXE, SCRIPT, str(METHOD), str(PHASE), subject1, subject2], check=True)
print("All comparisons completed.")

# Merge resulting images into grids
print("Merging resulting images into grids...")
subprocess.run([PYTHON_EXE, MERGING_SCRIPT], check=True)
