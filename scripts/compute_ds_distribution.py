import os
from collections import Counter

def compute_stats(folder_path):
    stats = {
        "positive": 0,
        "negative": 0,
        "match": 0,
        "mismatch": 0,
        "overlapped_emotions": Counter()
    }

    # Iterate through all subfolders and files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                # Check if the filename contains "positive" or "negative"
                if "masked-positive" in file:
                    stats["positive"] += 1
                elif "masked-negative" in file:
                    stats["negative"] += 1

                # Check if the filename contains "match" or "mismatch"
                if "mismatch" in file:
                    stats["mismatch"] += 1
                elif "match" in file:
                    stats["match"] += 1

                # Extract the overlapped emotion (last emotion in the filename)
                parts = file.split("__masked-")
                if len(parts) > 1:
                    overlapped_emotion = parts[1].split("-")[1].split("_")[0]
                    stats["overlapped_emotions"][overlapped_emotion] += 1

    return stats

if __name__ == "__main__":
    folder_path = "../output_images/bosphorus_test_HQ"  # Path to the output testset folder
    stats = compute_stats(folder_path)

    # Print the statistics
    print(f"Number of positive masks: {stats['positive']}")
    print(f"Number of negative masks: {stats['negative']}")
    print(f"Number of matches: {stats['match']}")
    print(f"Number of mismatches: {stats['mismatch']}")
    print("Overlapped emotions:")
    for emotion, count in stats["overlapped_emotions"].items():
        print(f"  {emotion}: {count}")