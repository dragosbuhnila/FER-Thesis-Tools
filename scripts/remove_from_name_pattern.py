import os

# ====== Remove or Preserve ======
REMOVE = False

# ====== Pattern to remove/preserve ======
# PATTERN = "flipped.png"
PATTERN = "pinken.png"

if __name__ == "__main__":
    ds_folder = "../bosphorus_test_HQ"
    folders = [f for f in os.listdir(ds_folder) if os.path.isdir(os.path.join(ds_folder, f))]

    for folder in folders:
        folder_path = os.path.join(ds_folder, folder)
        if REMOVE:
            # print(f"Removing images with the pattern '{PATTERN}' from folder: {folder}")
            images_names = [f for f in os.listdir(folder_path) if f.endswith(PATTERN)] 
        else:
            # print(f"Preserving images with the pattern '{PATTERN}' in folder: {folder}")
            images_names = [f for f in os.listdir(folder_path) if not f.endswith(PATTERN)]

        # Check if there are exactly 50 images in each folder
        if len(images_names) == 50:
            print(f"Folder {folder} has exactly 50 images with the pattern '{PATTERN}'.")
        elif len(images_names) == 0:
            print(f"Folder {folder} has no images with the pattern '{PATTERN}'.")
        else:
            print(f"Folder {folder} has {len(images_names)} images with the pattern '{PATTERN}', which is not 50.")

        for image_name in images_names:
            # rename by removing final pattern
            image_path = os.path.join(folder_path, image_name)
            os.remove(image_path)
            # print(f"Removed {image_path}")

        

       
