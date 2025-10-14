import subprocess
import os

# Define paths
PYTHON_EXE = os.path.abspath("./.venv/Scripts/python.exe")
SCRIPT = os.path.abspath("./compare_canonicals.py")
MERGING_SCRIPT = os.path.abspath("./scripts/make_images_grid.py")
METHOD = 5  # 5 - organize_folders_granularly

# Define comparisons
comparisons = [
    # ("men", "women"),  # maschi vs femmine
    ("best", "worst"),  # upper tail vs lower tail
]

# Run comparisons
for subject1, subject2 in comparisons:
    print(f"Running comparison: {subject1} vs {subject2}")
    subprocess.run([PYTHON_EXE, SCRIPT, str(METHOD), subject1, subject2], check=True)

print("All comparisons completed.")