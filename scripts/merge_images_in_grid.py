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
            # Ensure the order is "subj1" vs "subj2" based on the filenames
            image_paths = sorted(image_paths, key=lambda x: re.search(r"_VS_(.*?)_", os.path.basename(x)).group(1))
            output_filename = f"{pattern}_merged.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            merge_images_horizontally(image_paths, output_path)
        else:
            print(f"Warning: Group '{pattern}' has {len(image_paths)} images, expected 3. Skipping.")

def merge_images_vertically(image_paths, output_path, spacing=10):
    """Merge multiple images vertically with spacing, upscaling all to the max width."""
    images = [Image.open(path) for path in image_paths]
    widths = [img.size[0] for img in images]
    max_width = max(widths)

    # Resize all images to max_width, keeping aspect ratio
    resized_images = []
    for img in images:
        w, h = img.size
        if w != max_width:
            new_h = int(h * (max_width / w))
            img = img.resize((max_width, new_h), Image.LANCZOS)
        resized_images.append(img)

    heights = [img.size[1] for img in resized_images]
    total_height = sum(heights) + spacing * (len(resized_images) - 1)

    merged_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in resized_images:
        merged_img.paste(img, (0, y_offset))
        y_offset += img.size[1] + spacing

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    merged_img.save(output_path)
    print(f"Created: {output_path}")


def find_and_post_merge_models(folder, output_subfolder="post_merged_models"):
    """
    Look for files like:
      Comparison_OCCFT_CONVNEXT_BUB_VS_AMAESC_merged.png
    Group by (model, subject) for models starting with OCCFT_ and require methods BUB, EXT, GRAD.
    Create a vertical merged image in the order [BUB, EXT, GRAD].
    """
    post_out = os.path.join(folder, output_subfolder)
    os.makedirs(post_out, exist_ok=True)

    filenames = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.endswith("_merged.png")]
    pattern = re.compile(r"^Comparison_(?P<model>OCCFT_[^_]+)_(?P<method>BUB|EXT|GRAD)_VS_(?P<subject>[^_]+).*?_merged\.png$", re.IGNORECASE)

    groups = defaultdict(dict)  # (model, subject) -> {method: path}
    for fname in filenames:
        m = pattern.match(fname)
        if not m:
            continue
        model = m.group("model")
        method = m.group("method").upper()
        subject = m.group("subject")
        key = (model, subject)
        groups[key][method] = os.path.join(folder, fname)

    required = ["BUB", "EXT", "GRAD"]
    for (model, subject), method_map in groups.items():
        if all(k in method_map for k in required):
            ordered_paths = [method_map[k] for k in required]
            out_name = f"Comparison_{model}_VS_{subject}_methods_vertical.png"
            out_path = os.path.join(post_out, out_name)
            merge_images_vertically(ordered_paths, out_path, spacing=10)
        else:
            missing = [k for k in required if k not in method_map]
            print(f"Skipping ({model}, {subject}) — missing methods: {missing}")


RUN_ONLY_FINAL_MERGE = False


if __name__ == "__main__":
    if not RUN_ONLY_FINAL_MERGE:
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


    # After existing processing/moves
    find_and_post_merge_models(OUTPUT_FOLDER)
    print("All image groups processed.")