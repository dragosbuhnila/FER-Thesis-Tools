import os
import pickle

RANKING_FILE = os.path.join("saliency_maps", "human_Results", "testers_ranking.pkl")
GENDER_FILE = os.path.join("saliency_maps", "human_Results", "testers_gender.txt")

NAME_SETS_OUTPUT_FILENAME = os.path.join("saliency_maps", "human_Results", "testers_name_sets.pkl")
if not os.path.exists(NAME_SETS_OUTPUT_FILENAME):
    os.makedirs(os.path.dirname(NAME_SETS_OUTPUT_FILENAME), exist_ok=True)

DEBUG = False

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
    

    length = length_of_ranking  # or length_of_gender

    if DEBUG:
        print("\n=== RANKING ===")
        for entry in ranking:
            print(entry)
        print("\n=== GENDER ===")
        for entry in genders:
            print(f"{entry}")
    # < 

    # > Create sets
    upper_tail = ranking[:length // 2]
    upper_tail_names = {name for name, _ in upper_tail}
    lower_tail = ranking[length // 2:]
    lower_tail_names = {name for name, _ in lower_tail}

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