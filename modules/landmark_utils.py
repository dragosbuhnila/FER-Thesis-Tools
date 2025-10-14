import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python import BaseOptions

COORDSFOLDER = "landmark_coordinates"

EMOTION_AUS = {
    "ANGRY":    ["AU4", "AU23", "AU24"],
    "DISGUST":  ["AU9", "AU10", "AU16"],
    "FEAR":     ["AU1+2", "AU27"],
    "HAPPY":    ["AU6", "AU12"],
    "SAD":      ["AU1", "AU15"],
    "SURPRISE": ["AU1+2+5", "AU26"]
}

AU_SUBSET = { # 14 AUs
    "AU1" :  "AU1",
    "AU2" :  "AU2",
    "AU4" :  "AU4",
    "AU5" :  "AU5",
    "AU6" :  "AU6",
    "AU9" :  "AU9",
    "AU10": "AU10",
    "AU12": "AU12",
    "AU15": "AU15",  
    "AU16": "AU16",
    "AU23": "AU23", 
    "AU24": "AU24", 
    "AU26": "AU26", 
    "AU27": "AU27",
}

class LandmarkDict(dict):
    def __getitem__(self, key):
        # Custom logic before returning the value
        key = AU_SUBSET[key] if key in AU_SUBSET else key
        value = super().__getitem__(key)
        return value

AU_LANDMARKS = LandmarkDict({
    # ========================================================================
    "AU1+2+5": [[46, 105, 66, 107,                              # Eyebrows Upper L
                190, 56, 28, 27, 29, 247,                       # Eye Upper L
                46],            
                [336, 296, 334, 283,                            # Eyebrows Upper R
                467, 259, 257, 258, 286, 414,                   # Eye Upper R
                336]],

    "AU1+2":   [[46, 105, 66, 107,                              # Eyebrows Upper L
                46],            
                [336, 296, 334, 283,                            # Eyebrows Upper R
                336]],

    # AU1 === inner brow raiser === frontalis muscle (the medial portion) === 
    "AU1":      [[66, 107], [336, 296]],                        # Eyebrows Upper Internal

    # "AU1":      [66, 107, 336, 296],                          # Eyebrows Upper Internal

    # "AU1full":  [52, 65, 55, 285, 295, 282,                   # Eyebrow Lower     
    #             46, 105, 66, 107, 9, 336, 296, 334, 283,      # Eyebrows Upper (contains some lower)
    #             69, 108, 337, 299],                           # some medial frontalis

    # AU2 === outer brow raiser === frontalis muscle (the lateral portion) === 
    "AU2":      [[46, 105], [334, 283]],                        # Eyebrows Upper Outer (contains some lower)

    # "AU2":      [46, 105, 334, 283],                          # Eyebrows Upper Outer (contains some lower)

    # "AU2full":  [52, 65, 55, 285, 295, 282,                   # Eyebrow Lower     
    #             46, 105, 66, 107, 9, 336, 296, 334, 283,      # Eyebrows Upper (contains some lower)
    #             104, 333],                                    # some lateral frontalis

    # AU4 === brow lowerer ==== corrugator supercilii, depressor supercilii, and/or procerus muscles === 
    "AU4":      [[46, 105, 66, 107, 9, 336, 296, 334, 283]],    # Eyebrows Upper (contains some lower)

    # "AU4":      [46, 105, 66, 107, 9, 336, 296, 334, 283],    # Eyebrows Upper (contains some lower)

    # "AU4full":  [52, 65, 55, 8, 285, 295, 282,                # Eyebrow Lower 
    #             46, 105, 66, 107, 9, 336, 296, 334, 283,      # Eyebrows Upper (contains some lower)
    #             8, 9],                                        # Center

    # AU5 === upper lid raiser === levator palpebrae superioris ===
    "AU5":      [[414, 286, 258, 257, 259, 467],                # Eye Upper R
                [247, 29, 27, 28, 56, 190]],                    # Eye Upper L

    # "AU5":      [414, 286, 258, 257, 259, 467,                # Eye Upper R
                #  247, 29, 27, 28, 56, 190],                   # Eye Upper L

    # "AU5full":  [414, 286, 258, 257, 259, 467,                # Eye Upper R
    #             247, 29, 27, 28, 56, 190,                     # Eye Upper L
    #             256, 253, 339, 446,                           # Eye Lower R
    #             226, 110, 23, 26, 243],                       # Eye Lower L  

    # AU6 === cheek raiser === orbicularis oculi muscle (the orbital portion) ===
    "AU6":      [[256, 253, 339, 446,                           # Eye Lower R
                346, 329, 277,                                  # Cheek Upper R
                256],
                [226, 110, 23, 26, 243,                         # Eye Lower L  
                 47, 100, 117,                                  # Cheek Upper L
                 226]],

    # "AU6":      [256, 253, 339, 446,                          # Eye Lower R
    #             226, 110, 23, 26, 243,                        # Eye Lower L  
    #             277, 329, 348, 347, 346,                      # Cheek Upper R
    #             117, 118, 119, 100, 47],                      # Cheek Upper L

    # "AU6full":  [414, 286, 258, 257, 259, 467,                # Eye Upper R
    #             247, 29, 27, 28, 56, 190,                     # Eye Upper L
    #             256, 253, 339, 446,                           # Eye Lower R
    #             226, 110, 23, 26, 243,                        # Eye Lower L  
    #             277, 329, 348, 347, 346,                      # Cheek Upper R
    #             117, 118, 119, 100, 47],                      # Cheek Upper L

    # AU9 === nose wrinkler === levator labii superioris alaeque nasi === 
    "AU9":      [[195,
                 355, 279, 358, 392,                 # Nose Right (added 358 for gap in inverse mask)
                 166, 129, 49, 126,                   # Nose Left  (added 129 for gap in inverse mask)
                 195]],

    # "AU9":      [417, 351, 399, 420, 279, 392,                # Nose Right
    #             193, 122, 174, 198, 49, 166],                 # Nose Left  

    # "AU9full": [417, 351, 399, 420, 279, 392,                 # Nose Right
    #             193, 122, 174, 198, 49, 166,                  # Nose Left
    #             1,],                                          # Nose Center

    # AU10 === upper lip raiser === levator labii superioris === 
    "AU10":     [[216, 206, 203, 423, 426, 436,                 # NasoLabial Fold
                  291, 269, 0, 39,61,                           # Upper Lip Outer
                 216]],

    # "AU10":     [216, 206, 203, 423, 426, 436,                # NasoLabial Fold
    #             61, 39, 0, 269, 291,                          # Upper Lip Outer
    #             81, 13, 311,],                                # Upper Lip Inner

    # "AU10full": [329, 429,                                    # levator labii superioris Right 
    #             100, 209,                                     # levator labii superioris Left  
    #             216, 206, 423, 203, 426, 436,                 # NasoLabial Fold
    #             61, 39, 0, 269, 291,                          # Upper Lip Outer
    #             81, 13, 311,],                                # Upper Lip Inner

    # AU12 === lip corner puller === zygomaticus major ===
    "AU12":     [[436, 422,                                     # Mouth Side R
                405, 15, 181,                                   # Lower Lip Outer Alt w/o Chelions
                202, 216,                                       # Mouth Side L 
                436]],

    # "AU12":     [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             81, 13, 311,                                  # Upper Lip Inner
    #             88, 14, 318,                                  # Lower Lip Inner
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             202, 216, 436, 422],                          # Mouth Side
    
    # "AU12full": [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             81, 13, 311,                                  # Upper Lip Inner
    #             88, 14, 318,                                  # Lower Lip Inner
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             202, 216, 203, 423, 436, 422],                # Mouth Side

    # AU15 === lip corner depressor === depressor anguli oris ===
    "AU15":     [[39, 0, 269,                                   # Upper Lip Outer w/o Chelions
                287,                                            # Lip corner extension R
                405, 17, 181,                                   # Lower Lip Outer w/o Chelions
                57,                                             # Lip corner extension L
                39]],

    # "AU15":     [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             88, 14, 318,                                  # Lower Lip Inner
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             57, 287],                                     # Lip corner extension

    # "AU15full": [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             88, 14, 318,                                  # Lower Lip Inner
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             57, 287],                                     # Lip corner extension

    # AU16 === lower lip depressor === depressor labii inferioris === 
    "AU16":     [[291, 404, 16, 180, 61]],                      # Lower Lip Mid

    # "AU16":     [88, 14, 318,                                 # Lower Lip Inner
    #             61, 181, 17, 405, 291],                       # Lower Lip Outer

    # "AU16full": [88, 14, 318,                                 # Lower Lip Inner
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             57, 182, 18, 406, 287],                       # Below Lips

    # AU23 === lip tightener === orbicularis oris muscle (marginal portion) === 
    "AU23":     [[39, 0, 269,                                   # Upper Lip Outer w/o Chelions
                405, 17, 181,                                   # Lower Lip Outer w/o Chelions
                39]],

    # "AU23":     [39, 0, 269,                                  # Upper Lip Outer w/o Chelions
    #              88, 14, 318,                                 # Lower Lip Inner w/o Chelions
    #              181, 17, 405],                               # Lower Lip Outer w/o Chelions

    # "AU23full": [57, 165, 164, 391, 287,                      # Above Lips
    #             39, 0, 269,                                   # Upper Lip Outer w/o Chelions
    #             81, 13, 311,                                  # Upper Lip Inner w/o Chelions
    #             88, 14, 318,                                  # Lower Lip Inner w/o Chelions
    #             181, 17, 405,                                 # Lower Lip Outer w/o Chelions
    #             57, 182, 18, 406, 287],                       # Below Lips

    # AU24 === lip presser === orbicularis oris muscle (marginal portion) === 
    "AU24":     [[61, 39, 0, 269, 291,                          # Upper Lip Outer
                405, 17, 181,                                   # Lower Lip Outer w/o Chelions
                61]],

    # "AU24":     [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #              88, 14, 318,                                 # Lower Lip Inner w/o Chelions
    #              61, 181, 17, 405, 291,],                     # Lower Lip Outer

    # "AU24full": [57, 165, 164, 391, 287,                      # Above Lips
    #             61, 39, 0, 269, 291,                          # Upper Lip Outer
    #             81, 13, 311,                                  # Upper Lip Inner w/o Chelions
    #             88, 14, 318,                                  # Lower Lip Inner w/o Chelions
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             57, 182, 18, 406, 287],                       # Below Lips

    # AU26 === jaw drop === relaxation of masseter, temporalis, and medial pterygoid
    "AU26":     [[61, 180, 16, 404, 291,                        # Lower Lip Mid
                262, 428, 199, 208, 32,                         # Chin
                61]],

    # "AU26":     [88, 14, 318,                                 # Lower Lip Inner w/o Chelions
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             32, 176, 148, 152, 377, 400, 262],            # Chin

    # "AU26full": [88, 14, 318,                                 # Lower Lip Inner w/o Chelions
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             32, 176, 148, 152, 377, 400, 262],            # Chin

    # AU27 === mouth stretch === lateral pterygoid and the suprahyoid (anterior digastric, geniohyoid, and mylohyoid) ===
    "AU27":     [[61, 39, 0, 269, 291,                          # Upper Lip Outer
                262, 428, 199, 208, 32,                         # Chin
                61]],

    # "AU27":     [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             88, 14, 318,                                  # Lower Lip Inner w/o Chelions
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             32, 176, 148, 152, 377, 400, 262],            # Chin

    # "AU27full": [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             88, 14, 318,                                  # Lower Lip Inner w/o Chelions
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             32, 176, 148, 152, 377, 400, 262],            # Chin
})

FACE_PARTS_LANDMARKS = LandmarkDict({
    "Left Eyebrow": [[70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 
                      70]],
    "Right Eyebrow": [[336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 
                       336]],
    "Left Eye": [[35, 247, 470, 190, 244, 233, 23, 110, 35,
                  35]],
    "Right Eye": [[265, 467, 475, 414, 464, 453, 253, 339,
                   265]],
    # "Left Eye": [[33, 160, 158, 157, 173, 154, 145, 163
    #               , 33]],
    # "Right Eye": [[463, 384, 386, 388, 263, 390, 374, 381, 
    #               463]],
    "Left Cheek": [[117, 119, 114, 100, 142, 203, 206, 205, 50,
                    117]],
    "Right Cheek": [[346, 348, 343, 329, 371, 423, 426, 425, 280, 
                     346]],
    "Nose": [[168, 456, 360, 289, 19, 59, 131, 236,
              168]],  
    "Mouth": [[57, 186, 165, 164, 391, 410, 287, 
               422, 424, 418, 421, 200, 201, 194, 204, 202,
               57]],
    "Corrugator": [[8, 337, 108,
                    8]],
})

ROI_ORDER_FACEPARTS = ['Left Eyebrow', 'Right Eyebrow', 'Left Eye', 'Right Eye', 'Left Cheek', 'Right Cheek', 'Nose', 'Mouth', 'Corrugator']

FACE_PARTS_LANDMARKS_LRMERGED = LandmarkDict({
    "Eyebrows":     [FACE_PARTS_LANDMARKS["Left Eyebrow"][0], 
                     FACE_PARTS_LANDMARKS["Right Eyebrow"][0]],
    "Eyes":         [FACE_PARTS_LANDMARKS["Left Eye"][0], 
                     FACE_PARTS_LANDMARKS["Right Eye"][0]],
    "Cheeks":       [FACE_PARTS_LANDMARKS["Left Cheek"][0], 
                     FACE_PARTS_LANDMARKS["Right Cheek"][0]],
    "Nose":         [FACE_PARTS_LANDMARKS["Nose"][0]],
    "Mouth":        [FACE_PARTS_LANDMARKS["Mouth"][0]],
    "Corrugator":   [FACE_PARTS_LANDMARKS["Corrugator"][0]],
})

def detect_facial_landmarks(image_path):
    """
    Returns the coordinates for landmarks on an image.
    """
    # TODO: check if coordinates aren't already cached to save time (so you also need to add a feature to clear the cache in case you change something in the related MACROS)

    # Load the image
    image = mp.Image.create_from_file(image_path)

    # Set up FaceLandmarker
    model_path = os.path.join(".", "saliency_maps", "mediapipe", "face_landmarker.task")
    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)

        if result.face_landmarks:
            if len(result.face_landmarks) > 1:
                raise ValueError(f"Multiple faces detected in {image_path}. Only one face is expected.")
            face_landmarks = result.face_landmarks[0]
            if len(face_landmarks) < 4:
                raise ValueError(f"Insufficient landmarks detected in {image_path}, detected {len(face_landmarks)}.")
            
            desired_landmarks = []

            annotated_image = cv2.imread(image_path)
            h, w, _ = annotated_image.shape
            for face_landmarks in result.face_landmarks:
                for idx in range(len(face_landmarks)):
                    landmark = face_landmarks[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    desired_landmarks.append((x, y))
            
            return desired_landmarks
        else:
            raise ValueError(f"No face detected in {image_path}")
        

def save_landmark_coordinates(path, landmark_coordinates, coords_folder=COORDSFOLDER):
    """
        Saves the landmark coordinates to a .npy file in the COORDSFOLDER.
        The path/filename includes emotion as folder and same name as the image
    """
    filename = f"{path[:-4]}.npy"  # Save as .npy
    filepath = os.path.join(coords_folder, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure all subfolders exist
    np.save(filepath, np.array(landmark_coordinates, dtype=np.int32))


def load_landmark_coordinates(path, coords_folder=COORDSFOLDER):
    """
        The path/filename includes emotion as folder and same name as the image
    """
    filename = f"{path[:-4]}.npy"
    filepath = os.path.join(coords_folder, filename)
    if not os.path.exists(filepath):
        return None
    return np.load(filepath)

def get_all_AUs():
    """
    Returns a list of all AUs available in the LANDMARKS dictionary.
    """
    basic_aus = [key for key in AU_LANDMARKS.keys() if not "+" in key]
    return basic_aus

def get_all_face_parts():
    """
    Returns a list of all face parts available in the FACE_PARTS_LANDMARKS dictionary.
    """
    return list(FACE_PARTS_LANDMARKS.keys())

def get_all_face_parts_lrmerged():
    """
    Returns a list of all face parts available in the FACE_PARTS_LANDMARKS dictionary.
    """
    return list(FACE_PARTS_LANDMARKS_LRMERGED.keys())