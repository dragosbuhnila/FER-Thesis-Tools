import subprocess
import os

# Define paths
PYTHON_EXE = os.path.abspath("./.venv/Scripts/python.exe")
SCRIPT = os.path.abspath("./compare_canonicals.py")
MERGING_SCRIPT = os.path.abspath("./scripts/merge_images_in_grid.py")
METHOD = 3

# Define comparisons
comparisons = [
    ("convnext_grad", "fedmar"),
    ("convnext_grad", "marfro"),
    ("convnext_bub", "fedmar"),
    ("convnext_bub", "marfro"),
    ("convnext_ext", "fedmar"),
    ("convnext_ext", "marfro"),
    ("fedmar", "matvin"),
    ("marfro", "matvin"),
    ("fedmar", "marfro"),
    ("men", "women"),
    ("best", "worst"),
]

# Run comparisons
# for subject1, subject2 in comparisons:
#     print(f"Running comparison: {subject1} vs {subject2}")
#     subprocess.run([PYTHON_EXE, SCRIPT, str(METHOD), subject1, subject2], check=True)
# print("All comparisons completed.")

# Merge resulting images into grids
print("Merging resulting images into grids...")
subprocess.run([PYTHON_EXE, MERGING_SCRIPT], check=True)
