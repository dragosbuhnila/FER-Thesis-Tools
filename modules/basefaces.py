import os

import cv2
from modules.landmark_utils import detect_facial_landmarks, load_landmark_coordinates, save_landmark_coordinates


BASEFACES_FOLDER = "./saliency_maps/basefaces"

# =============================================================================
# =============================================================================

class BaseFace:
    def __init__(self, fname, force_remap=False):
        # a filename
        self.fname = fname

        # b image 
        image = cv2.imread(os.path.join(BASEFACES_FOLDER, fname))
        if image is None:
            raise FileNotFoundError(f"Base face image {fname} could not be loaded.")
        self.image = image
        
        # c image shape
        self.shape = image.shape

        # d landmarks
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

def get_base_face(emotion):
    if emotion.upper() not in basefaces.keys():
        print(f"Base face for emotion {emotion} not found, returning neutral base face.")
        emotion = "NEUTRAL"
    return basefaces[emotion.upper()]

# =============================================================================
# =============================================================================

# > Load basefaces
baseface_fnames = [base_face_fname for base_face_fname in os.listdir(BASEFACES_FOLDER) if base_face_fname.endswith('reshaped.png')]
if len(baseface_fnames) != 7:
    raise ValueError(f"Expected 7 basefaces, found {len(baseface_fnames)}. Please check the basefaces directory.")

basefaces = {}
for baseface_fname in baseface_fnames:
    selected_emotion = baseface_fname.split('_')[1].upper()  # Assuming the filename format is like "baseface_emotion_reshaped.png"
    basefaces[selected_emotion] = BaseFace(baseface_fname)
    # print(f"Loaded base face for emotion {emotion}. {basefaces[emotion]}")
