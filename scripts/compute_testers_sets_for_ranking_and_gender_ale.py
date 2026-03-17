import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle

from modules.saliencies_folders import BASE_DIR, HEATMAPS_ALE_DIR_BASENAME, RANKING_PHASE2_FILE_PATH, GENDER_PHASE2_FILE_PATH



HUMAN_RESULTS_BASE_FOLDER_PATH = os.path.join(BASE_DIR, HEATMAPS_ALE_DIR_BASENAME)
NAME_SETS_OUTPUT_FILENAME = os.path.join(HUMAN_RESULTS_BASE_FOLDER_PATH, "testers_name_sets.pkl")
if not os.path.exists(NAME_SETS_OUTPUT_FILENAME):
    os.makedirs(os.path.dirname(NAME_SETS_OUTPUT_FILENAME), exist_ok=True)

RANKING_FILE = RANKING_PHASE2_FILE_PATH
GENDER_FILE = GENDER_PHASE2_FILE_PATH

UPPER_TAIL = {
    "ALITOM": 0.672365,
    "MARCEC": 0.672365,
    "MARDIB": 0.678063,
    "SILVEN": 0.680912,
    "FEDMAR": 0.680912,
    "AMAESC": 0.683761,
    "REBLEO": 0.689459,
}

LOWER_TAIL = {
    "LUCRUG": 0.233618,
    "KRIJAK": 0.467236,
    "FILCOM": 0.495726,
}


DEBUG = True

if __name__ == "__main__":
    # > Open the ranking and gender files
    ranking = pickle.load(open(RANKING_FILE, "rb"))
    length_of_ranking = len(ranking)
    print(f"length of ranking: {length_of_ranking}")

    genders = []
    with open(GENDER_FILE, "r") as gender_file:
        for line in gender_file:
            line = line.strip()
            if line:
                key, value = line.split("\t")
                if value not in ("Maschio", "Femmina"):
                    raise ValueError(f"Unexpected gender value: {value}")
                genders.append((key.strip(), value.strip()))
        
    length_of_gender = len(genders)
    print(f"length of gender: {length_of_gender}")

    if length_of_ranking != length_of_gender:
        raise ValueError(f"Ranking and gender lists have different lengths: {length_of_ranking} != {length_of_gender}")

    if DEBUG:
        print("\n=== RANKING ===")
        for entry in ranking:
            print(entry)
        print("\n=== GENDER ===")
        for entry in genders:
            print(f"{entry}")
    # < 

    # > Create sets
    ## upper and lower tail are now hardcoded bc the stdev thing yielded different percentages
    upper_tail_names = {name for name in UPPER_TAIL}
    lower_tail_names = {name for name in LOWER_TAIL}

    if DEBUG:
        print(f"\n=== UPPER TAIL === ({len(upper_tail_names)} entries)")
        for entry in upper_tail_names:
            print(entry)
        print(f"\n=== LOWER TAIL === ({len(lower_tail_names)} entries)")
        for entry in lower_tail_names:
            print(entry)

    male_names =    {name for name, gender in genders if gender == "Maschio"}
    female_names =  {name for name, gender in genders if gender == "Femmina"}
    if DEBUG:
        print(f"\n=== MALE === ({len(male_names)} entries)")
        for entry in male_names:
            print(entry)
        print(f"\n=== FEMALE === ({len(female_names)} entries)")
        for entry in female_names:
            print(entry)

    name_sets = {
        "best": upper_tail_names,
        "worst": lower_tail_names,
        "men": male_names,
        "women": female_names
    }

    with open(NAME_SETS_OUTPUT_FILENAME, "wb") as output_file:
        pickle.dump(name_sets, output_file)
    # < 

    # > Try and open it, and show it
    with open(NAME_SETS_OUTPUT_FILENAME, "rb") as input_file:
        loaded_name_sets = pickle.load(input_file)
    print("Loaded name sets:")
    for key, names in loaded_name_sets.items():
        print(f"  {key}: {len(names)} entries")
    print("contents:")
    print(f"best: {loaded_name_sets['best']}")
    print(f"worst: {loaded_name_sets['worst']}")
    print(f"men: {loaded_name_sets['men']}")
    print(f"women: {loaded_name_sets['women']}")
    # <