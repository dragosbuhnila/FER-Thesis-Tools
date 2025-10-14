import os
import re
from PIL import Image
import numpy as np
from collections import defaultdict

# from modules.compare_saliency_maps import COMPARISON_GRID_FOLDER

DEBUG = True

INPUT_FOLDER = os.path.join(".", "saliency_maps", "zzz_other_and_zips", "output_comparisons_meandif")
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "results_of_merging")
ALL_FILES_UNSORTED_FOLDER = os.path.join(OUTPUT_FOLDER, "all_files_unsorted")
NOF_IMAGES = 4 # Number of images to merge in a row. If more or less images are found for a pattern, a warning is issued and the group is skipped.

def extract_pattern(filename, pattern_regex=r"Comparison_(.*?)_VS_.*?_with_.*?\.png", debug=False):
    """Extract the common pattern from a filename using regex."""
    match = re.search(pattern_regex, filename)
    if match:
        if debug:
            print(f"Extracted pattern '{match.group(1)}' from filename '{filename}'")
        return match.group(1)
    if debug:
        print(f"Could not extract pattern from filename '{filename}'")
    return None

def merge_images_horizontally(image_paths, output_path, spacing=10):
    """Merge multiple images horizontally with spacing between them, upscaling all to the max height."""
    images = [Image.open(path) for path in image_paths]
    heights = [img.size[1] for img in images]
    max_height = max(heights)

    # Resize all images to max_height, keeping aspect ratio
    resized_images = []
    for img in images:
        w, h = img.size
        if h != max_height:
            new_w = int(w * (max_height / h))
            img = img.resize((new_w, max_height), Image.LANCZOS)
        resized_images.append(img)

    widths = [img.size[0] for img in resized_images]
    total_width = sum(widths) + spacing * (len(resized_images) - 1)

    merged_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for img in resized_images:
        merged_img.paste(img, (x_offset, 0))
        x_offset += img.size[0] + spacing

    if os.path.exists(output_path):
        os.remove(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    merged_img.save(output_path)
    
    print(f"Created: {output_path}")

def find_and_merge_image_groups(folder, pattern_regex=None):
    """Find groups of images with common patterns and merge them."""
    print(f"Scanning folder: {folder}")
    print(f"Found {len(os.listdir(folder))} total files.")
    filenames = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(filenames)} image files.")
    filenames = [f for f in filenames if "merged" not in f]  # Exclude already merged images
    print(f"Found {len(filenames)} unmerged files.")
    
    # Group files by pattern
    groups = defaultdict(list)
    for filename in filenames:
        if pattern_regex:
            pattern = extract_pattern(filename, pattern_regex, debug=DEBUG)
        else:
            raise NotImplementedError("A pattern_regex must be provided.")
            
        if pattern:
            groups[pattern].append(os.path.join(folder, filename))

    # Process each group that has exactly NOF_IMAGES images
    for pattern, image_paths in groups.items():
        if len(image_paths) == NOF_IMAGES:
            # Sort to ensure consistent order
            image_paths.sort()
            output_filename = f"{pattern}_merged.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            merge_images_horizontally(image_paths, output_path)
        else:
            print(f"Warning: Group '{pattern}' has {len(image_paths)} images, expected 3. Skipping.")

if __name__ == "__main__":
    # Pattern for "Comparison_X_with_Y()" style filenames
    pattern_regex = r"(Comparison_(.*?)_VS_(.*?))_(.*?).png"
    
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    find_and_merge_image_groups(INPUT_FOLDER, pattern_regex)

    all_pngs = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "merged" not in f]
    for f in all_pngs:
        src_path = os.path.join(INPUT_FOLDER, f)
        dst_path = os.path.join(ALL_FILES_UNSORTED_FOLDER, f)
        if not os.path.exists(ALL_FILES_UNSORTED_FOLDER):
            os.makedirs(ALL_FILES_UNSORTED_FOLDER)
        if not os.path.exists(dst_path):
            os.rename(src_path, dst_path)
            print(f"Moved {src_path} to {dst_path}")
        else:
            print(f"File {dst_path} already exists. Skipping move.")
    
    print("All image groups processed.")