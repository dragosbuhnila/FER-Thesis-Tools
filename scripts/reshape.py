import cv2
import os

CUR_FOLDER = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    basefaces = [baseface for baseface in os.listdir(CUR_FOLDER) if baseface.startswith('baseface') and baseface.endswith('.png')]
    print(f"found {len(basefaces)} basefaces: {basefaces}")
    for baseface in basefaces:
        # Read the image
        img = cv2.imread(os.path.join(CUR_FOLDER, baseface))
        if img is None:
            print("Image not found!")
        else:
            # Resize to (height=85, width=75)
            resized = cv2.resize(img, (75*10, 85*10), interpolation=cv2.INTER_AREA)
            final_name = f"{baseface.split('/')[-1].split('.')[0]}_reshaped.png"
            cv2.imwrite(os.path.join(CUR_FOLDER, final_name), resized)
            print(f"Image reshaped and saved as {final_name}")