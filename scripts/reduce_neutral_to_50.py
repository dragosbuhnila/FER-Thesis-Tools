import os

if __name__ == "__main__":
    neutral_folder = "../bosphorus_test_HQ/NEUTRAL"

    # Get all files in the NEUTRAL folder
    files = [f for f in os.listdir(neutral_folder) if f.endswith("_pinken.png")]

    # Dictionary to track the first file for each subject
    subjects_seen = {}

    for file in sorted(files):  # Sort files to ensure consistent order
        # Extract the subject ID (e.g., "bs012" from "bosphorus_bs012_NEUTRAL_483_pinken.png")
        subject_id = file.split("_")[1]

        # If the subject is already seen, delete the file
        if subject_id in subjects_seen:
            file_path = os.path.join(neutral_folder, file)
            os.remove(file_path)
            print(f"Removed duplicate: {file}")
        else:
            # Mark the subject as seen
            subjects_seen[subject_id] = file

    print("Duplicates removed. Only the first file for each subject is kept.")