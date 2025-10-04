import os
import shutil

INPUT_FOLDER = "zzz_input_saliency_maps"
OUTPUT_FOLDER = "zzz_output_means_vectors"

CANONICAL_ONLY = True

DEBUG = False

if __name__ == "__main__":
    # > normalize subjects folders
    for subject_folder in os.listdir(INPUT_FOLDER):
        subject_folder_path = os.path.join(INPUT_FOLDER, subject_folder)
        if os.path.isdir(subject_folder_path):
            # > Delete everything except heatmaps folder
            for fname in os.listdir(subject_folder_path):
                if fname.lower() != "heatmaps" and not fname.lower().endswith('canonical.npy'):
                    fpath = os.path.join(subject_folder_path, fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                    elif os.path.isdir(fpath):
                        shutil.rmtree(fpath) 

            # > Move the heatmaps that are canonical to the subject dir
            for fname in os.listdir(subject_folder_path):
                if fname.lower() != "heatmaps" and not fname.lower().endswith('canonical.npy'):
                    raise ValueError(f"Subject folder {subject_folder_path} should contain only one folder named 'heatmaps' or already extracted npy files.")
                
                if fname.lower() != "heatmaps":
                    continue

                fname_path = os.path.join(subject_folder_path, fname)
                canonical_npys = [f for f in os.listdir(fname_path) if f.endswith('canonical.npy')]
                for canonical_npy in canonical_npys:
                    os.rename(os.path.join(fname_path, canonical_npy), os.path.join(subject_folder_path, canonical_npy))
            # <

            # > Finally remove the heatmaps folder
            if os.path.exists(os.path.join(subject_folder_path, "heatmaps")):
                shutil.rmtree(os.path.join(subject_folder_path, "heatmaps"))
    print("Normalized subject folders.")
    # <