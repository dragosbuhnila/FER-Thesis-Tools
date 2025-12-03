# Dragos Buhnila 2025 ========================================================================================

import sys
from modules.compare_saliency_maps import compare_single_person, compare_top_and_emotions, compare_two_subjects, compare_two_subjects_cmd, create_organized_folders_aggr, create_organized_folders_aggr_cmd, create_organized_folders_gran, create_organized_folders_gran_cmd, print_compare_saliency_maps_macros, recalculate_all_stats, test_comparison

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

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
        print("8) Recalculate statistics for all heatmaps")
        print("9) Quit")
        if len(sys.argv) == 4:
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
            if len(sys.argv) == 4:
                compare_two_subjects_cmd(sys.argv[2], sys.argv[3])
                exit(0)
            elif len(sys.argv) > 1:
                print("  >> Please provide either: \n> two subject identifiers + <choice> \n> nothing")
            else:
                compare_two_subjects()
            continue

        elif choice == "4": 
            if len(sys.argv) == 4:
                create_organized_folders_aggr_cmd(sys.argv[2], sys.argv[3])
                exit(0)
            elif len(sys.argv) > 1:
                print("  >> Please provide either: \n> two subject identifiers + <choice> \n> nothing")
            else:
                create_organized_folders_aggr()
            continue

        elif choice == "5":
            if len(sys.argv) == 4:
                create_organized_folders_gran_cmd(sys.argv[2], sys.argv[3])
                exit(0)
            elif len(sys.argv) > 1:
                print("  >> Please provide either: \n> two subject identifiers + <choice> \n> nothing")
            else:
                create_organized_folders_gran()

        elif choice == "8":
            recalculate_all_stats()

        elif choice == "9":
            print("Goodbye!")
            break

        else:
            print("  >> Invalid choice.")