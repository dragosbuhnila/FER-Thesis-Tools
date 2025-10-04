
import os
import shutil
import numpy as np

from modules.landmark_utils import ROI_ORDER_FACEPARTS as ROI_ORDER
from modules.roi_statistics import convert_faceparts_roi_means_from_dict_to_vector as convert_from_dict_to_vector

INPUT_FOLDER = "zzz_output_means_vectors"
OUTPUT_FOLDER = "zzz_output_means_vectors_no-dict"

CANONICAL_ONLY = True

DEBUG = False


if __name__ == "__main__":
    # > Show the first output file you find as example
    # >> List contents of input folder
    things_in_input_folder = os.listdir(INPUT_FOLDER)
    files = [thing for thing in things_in_input_folder if os.path.isfile(os.path.join(INPUT_FOLDER, thing))]
    dirs = [thing for thing in things_in_input_folder if os.path.isdir(os.path.join(INPUT_FOLDER, thing))]

    for file in files:
        if not file.endswith('.npy'):
            raise ValueError(f"Non-npy file found in input folder: {file}")

    # >> Try files first
    found = False
    example_stats = None
    example_fname = None
    for heatmap_fname in files:
        if heatmap_fname.endswith('mean-vector.npy'):
            example_stats = np.load(os.path.join(INPUT_FOLDER, heatmap_fname), allow_pickle=True).item()
            example_fname = heatmap_fname
            found = True
            break
    # >> Then try directories
    if not found:
        for d in dirs:
            heatmap_fnames = os.listdir(os.path.join(INPUT_FOLDER, d))
            for heatmap_fname in heatmap_fnames:
                if heatmap_fname.endswith('mean-vector.npy'):
                    example_stats = np.load(os.path.join(OUTPUT_FOLDER, heatmap_fname), allow_pickle=True).item()
                    example_fname = heatmap_fname
                    found = True
                    break
            if found:
                break
    # >> Finally show  
    if not found and example_stats is None:
        raise ValueError(f"No output files found in input folder ({INPUT_FOLDER}) or its subfolders({len(dirs)}): found={found}, example_stats={example_stats}")
    if list(example_stats.keys()) != ROI_ORDER:
        raise ValueError(f"ROIs in example stats do not match expected order. Found: {list(example_stats.keys())}, Expected: {ROI_ORDER}")
    print(f"Example stats ROIs, in order, are: {list(example_stats.keys())}")
    print(f"Example stats for {example_fname} look like this:")
    for roi, vals in example_stats.items():
        print(f"  {roi}: {vals}")
    # <


    # > Empty output folder
    if os.path.exists(OUTPUT_FOLDER):
        print(f"Emptying output folder: {OUTPUT_FOLDER}")
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    # <


    # > Do the conversion           # dict is like {'Left Eyebrow': {'mean': 0.3555507771417289}, 'Right Eyebrow': {'mean': 0.10694237923453793}, ...}
                                    # I want to convert it to a vector like [0.3555507771417289, 0.10694237923453793, ...] (in a fixed order of ROIs)
    # >> First process files in the input folder
    for dict_filename in files:
        if not dict_filename.endswith('.npy'):
            raise ValueError(f"Non-npy file found in input folder: {dict_filename}")
        
        dict_relpath = os.path.join(INPUT_FOLDER, dict_filename)
        stats_dict = np.load(dict_relpath, allow_pickle=True).item() 
        stats_vector = convert_from_dict_to_vector(stats_dict)

        # Save the vector to the output folder
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        output_path = os.path.join(OUTPUT_FOLDER, dict_filename)
        
        np.save(output_path, stats_vector, allow_pickle=True)
    # <<

    # >> Then process directories
    for d in dirs:
        dirpath = os.path.join(INPUT_FOLDER, d)
        dict_fnames = os.listdir(dirpath)
        dict_fnames = [f for f in dict_fnames if f.endswith('.npy')]

        for dict_fname in dict_fnames:
            dict_relpath = os.path.join(dirpath, dict_fname)
            stats_dict = np.load(dict_relpath, allow_pickle=True).item() 
            stats_vector = convert_from_dict_to_vector(stats_dict)

            # Save the vector to the output folder inside a folder named like d
            input_subfolder = os.path.join(OUTPUT_FOLDER, d)
            if not os.path.exists(input_subfolder):
                os.makedirs(input_subfolder)
            output_path = os.path.join(input_subfolder, dict_fname)
            np.save(output_path, stats_vector, allow_pickle=True)
    # <<
    # <


    # > Verify output
    things_in_output_folder = os.listdir(OUTPUT_FOLDER)
    files = [thing for thing in things_in_output_folder if os.path.isfile(os.path.join(OUTPUT_FOLDER, thing))]
    dirs = [thing for thing in things_in_output_folder if os.path.isdir(os.path.join(OUTPUT_FOLDER, thing))]

    found = False
    example_stats = None
    example_fname = None
    for file in files:
        if not file.endswith('.npy'):
            raise ValueError(f"Non-npy file found in output folder: {file}")
        
        output_stats = np.load(os.path.join(OUTPUT_FOLDER, file), allow_pickle=True)

        found = True
        example_stats = output_stats
        example_fname = file
        break

    if not found:
         for d in dirs:
            dirpath = os.path.join(OUTPUT_FOLDER, d)
            output_fnames = os.listdir(dirpath)
            output_fnames = [f for f in output_fnames if f.endswith('.npy')]

            for output_fname in output_fnames:
                output_stats = np.load(os.path.join(dirpath, output_fname), allow_pickle=True)

                found = True
                example_stats = output_stats
                example_fname = output_fname
                break
            if found:
                break
    
    print(f"Example output file: {example_fname}")
    print(example_stats)
    