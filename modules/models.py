import os
import cv2

from modules.landmark_utils import detect_facial_landmarks, load_landmark_coordinates, save_landmark_coordinates


BASEFACES_FOLDER = "./saliency_maps/basefaces"

class BaseFace:
    def __init__(self, fname, force_remap=False):
        # a filename
        self.fname = fname

        # b image shape
        image = cv2.imread(os.path.join(BASEFACES_FOLDER, fname))
        if image is None:
            raise FileNotFoundError(f"Base face image {fname} could not be loaded.")
        self.shape = image.shape

        # c landmarks
        img_path = os.path.join(BASEFACES_FOLDER, fname)
        self.landmarks = load_landmark_coordinates(img_path, coords_folder=BASEFACES_FOLDER)
        if self.landmarks is None or force_remap:
            print(f"Detecting landmarks from {img_path}")
            self.landmarks = detect_facial_landmarks(img_path)
            save_landmark_coordinates(img_path, self.landmarks, coords_folder=BASEFACES_FOLDER)
        else:
            print(f"Loading cached landmarks from {img_path}")
    
    def __str__(self):
        return f"BaseFace(fname={self.fname}, shape={self.shape}, there are {len(self.landmarks)} landmarks)"
