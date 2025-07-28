import os
import cv2
import numpy as np
import mediapipe as mp

# For the time being always use the same beige/pink
BG_COLOR = (203,151,107)
OVERWRITE_FILE = False

if __name__ == "__main__":
    ds_folder = "../bosphorus_test_HQ"
    folders = [f for f in os.listdir(ds_folder) if os.path.isdir(os.path.join(ds_folder, f))]

    for folder in folders:
        folder_path = os.path.join(ds_folder, folder)
        images_names = [f for f in os.listdir(folder_path) if f.endswith("png")]

        # Replace each image with a new one in which the background color is modified
        for image_name in images_names:
            image_path = os.path.join(folder_path, image_name)

            # Read the image
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error reading {image_name} in folder {folder}. Skipping...")
                continue
            
            bg_color = BG_COLOR  # Default background color
            # bg_color = get_background_color(img)  # Get background color from the image

            # Ensure the image has an alpha channel (transparency)
            if img.shape[-1] == 4:  # RGBA
                # Separate the alpha channel
                b, g, r, a = cv2.split(img)

                # Create a mask for black pixels (where RGB is [0, 0, 0])
                black_mask = (b == 0) & (g == 0) & (r == 0)

                # Replace black pixels with the new background color
                b[black_mask] = bg_color[2]  # Blue channel
                g[black_mask] = bg_color[1]  # Green channel
                r[black_mask] = bg_color[0]  # Red channel

                # Merge the channels back
                img = cv2.merge((b, g, r, a))
            else:  # No alpha channel (standard RGB)
                # Separate the alpha channel
                b, g, r= cv2.split(img)

                # Create a mask for black pixels (where RGB is [0, 0, 0])
                black_mask = (b == 0) & (g == 0) & (r == 0)

                # Replace black pixels with the new background color
                b[black_mask] = bg_color[2]  # Blue channel
                g[black_mask] = bg_color[1]  # Green channel
                r[black_mask] = bg_color[0]  # Red channel

                # Merge the channels back
                img = cv2.merge((b, g, r))

            if OVERWRITE_FILE:
                # Overwrite the original image
                cv2.imwrite(image_path, img)
                print(f"Overwritten {image_name} in folder {folder}")
            else:
                # Save the modified image with a new name
                new_image_path = os.path.join(folder_path, f"{image_name[:-4]}_pinken.png")
                cv2.imwrite(new_image_path, img)
                print(f"Processed {image_name} in folder {folder}")

    print("Background color replacement completed!")