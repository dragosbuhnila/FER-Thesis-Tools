# Dragos Buhnila 2025 ========================================================================================

import sys
from modules.compare_saliency_maps import compare_single_person, compare_top_and_emotions, compare_two_subjects, compare_two_subjects_cmd, compute_left_right_wrapper, create_organized_folders_aggr, create_organized_folders_aggr_cmd, create_organized_folders_gran, create_organized_folders_gran_cmd, print_compare_saliency_maps_macros, recalculate_all_stats, test_comparison

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")


# >>> run LEFT-RIGHT for both FEDE and OCCFT
# & "C:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/.venv/Scripts/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/compare_canonicals.py" 6
# >>> run FOLDERS-GRAN for OCCFT best worst
# & "C:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/.venv/Scripts/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/compare_canonicals.py" 5 2 best worst
# >>> run FOLDERS-GRAN for OCCFT men women
# & "C:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/.venv/Scripts/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/compare_canonicals.py" 5 2 men women
# >>> run SUBJECTS for pattlite_bub/mobilenet_bub vs AMAESC
# & "C:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/.venv/Scripts/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/compare_canonicals.py" 3 2 occft_pattlite_bub amaesc
if __name__ == "__main__":
    # Args: <choice=3> <subject_1> <subject_2>

    print_compare_saliency_maps_macros()

    while True:
        print("\n=== Main Menu ===")
        print("0) Test metric between a full-blue and a full-red heatmap")
        print("1) Combinations Single Person")
        print("2) Combinations Top%, and Emotion")
        print("3) Versus")
        print("4) Create organized folders with aggregated heatmaps")
        print("5) Create organized folders with granular heatmaps")
        print("6) Compute left-right heatmaps")
        print("8) Recalculate statistics for all heatmaps")
        print("9) Quit")
        if len(sys.argv) > 1:
            choice = sys.argv[1]
        else:
            choice = input("Enter choice: ").strip()

        if choice == "0":
            test_comparison()

        elif choice == "1":
            compare_single_person()

        elif choice == "2":
            compare_top_and_emotions()

        elif choice == "3":
            if len(sys.argv) == 5:
                compare_two_subjects_cmd(sys.argv[2], sys.argv[3], sys.argv[4])
                exit(0)
            elif len(sys.argv) > 1:
                print("  >> Please provide either: \n> <choice> + <phase> + <subject_name_1> + <subject_name_2> \n> nothing")
            else:
                compare_two_subjects()
            continue

        elif choice == "4": 
            if len(sys.argv) == 5:
                create_organized_folders_aggr_cmd(sys.argv[2], sys.argv[3], sys.argv[4])
                exit(0)
            elif len(sys.argv) > 1:
                print("  >> Please provide either: \n> <choice> + <phase> + <subject_name_1> + <subject_name_2> \n> nothing")
            else:
                create_organized_folders_aggr()
            continue

        elif choice == "5":
            if len(sys.argv) == 5:
                create_organized_folders_gran_cmd(sys.argv[2], sys.argv[3], sys.argv[4])
                exit(0)
            elif len(sys.argv) > 1:
                print("  >> Please provide either: \n> <choice> + <phase> + <subject_name_1> + <subject_name_2> \n> nothing")
            else:
                create_organized_folders_gran()
        
        elif choice == '6':
            if len(sys.argv) == 2:
                print("  >> Computing left-right heatmaps for folder: HEATMAPS_machines_phase2")
                compute_left_right_wrapper("HEATMAPS_machines_phase2")
                print("  >> Computing left-right heatmaps for folder: HEATMAPS_machines_phase1")
                compute_left_right_wrapper("HEATMAPS_machines_phase1")
                exit(0)
            else:
                print("  >> Please only provide function choice (6 in this case).")

        elif choice == "8":
            recalculate_all_stats()

        elif choice == "9":
            print("Goodbye!")
            break

        else:
            print("  >> Invalid choice.")