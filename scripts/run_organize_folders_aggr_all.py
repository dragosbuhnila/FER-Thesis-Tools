import subprocess
import os

# Define paths
PYTHON_EXE = os.path.abspath("./.venv/Scripts/python.exe")
SCRIPT = os.path.abspath("./compare_canonicals.py")
MERGING_SCRIPT = os.path.abspath("./scripts/make_images_grid.py")
METHOD = 4

# Define comparisons (uncommented ones from the batch file)
comparisons = [
    ("fedmar", "marfro"),
    # Uncomment the following lines if needed:
    # ("fedmar", "matvin"),
    # ("marfro", "matvin"),
    # ("men", "women"),
    # ("best", "worst"),
    # ("convnext_bub", "fedmar"),
    # ("convnext_bub", "marfro"),
    # ("convnext_ext", "fedmar"),
    # ("convnext_ext", "marfro"),
    # ("convnext_grad", "fedmar"),
    # ("convnext_grad", "marfro"),
]

# Run comparisons
for subject1, subject2 in comparisons:
    print(f"Running comparison: {subject1} vs {subject2}")
    subprocess.run([PYTHON_EXE, SCRIPT, str(METHOD), subject1, subject2], check=True)

print("All comparisons completed.")