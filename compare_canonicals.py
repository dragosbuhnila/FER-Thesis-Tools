# Dragos Buhnila 2025 ========================================================================================

from modules.compare_saliency_maps import compare_single_person, compare_top_and_emotions, print_compare_saliency_maps_macros, recalculate_all_stats, test_comparison

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

if __name__ == "__main__":
    print_compare_saliency_maps_macros()

    while True:
        print("\n=== Main Menu ===")
        print("0) Test metric between a full-blue and a full-red heatmap")
        print("1) Single Person")
        print("2) Top%, and Emotion")
        print("3) Top% + Gender, and Emotion (not implemented)")
        print("8) Recalculate statistics for all heatmaps")
        print("9) Quit")
        choice = input("Enter choice: ").strip()

        if choice == "0":
            test_comparison()

        elif choice == "1":
            compare_single_person()

        elif choice == "2":
            compare_top_and_emotions()

        elif choice == "3":
            print("  >> Option not implemented yet.")
            continue

        elif choice == "8":
            recalculate_all_stats()

        elif choice == "9":
            print("Goodbye!")
            break

        else:
            print("  >> Invalid choice.")