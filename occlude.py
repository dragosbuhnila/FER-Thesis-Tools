# Dragos Buhnila 2025 ========================================================================================

import os
import random
import matplotlib.pyplot as plt
import cv2
from modules.landmark_utils import detect_facial_landmarks, save_landmark_coordinates, load_landmark_coordinates
from modules.mask_n_heatmap_utils import mask_face_circles, mask_face_dots, mask_face_lines, apply_inverse_masks 

# ========================== SETTINGS ===========================================================================

# === show dots, full-masks, or inverse-masks ===
# MASK_TYPE = "dots" 
MASK_TYPE = "circles"
# MASK_TYPE = "full"  
# MASK_TYPE = "inverse"  

MASK_COLOR = (54, 61, 52) # Good ones: graphite_gray: (54, 61, 52); 

# === show the 14 AUs, the 6 emotions, or the non-matching emotions ===
# SHOW_EMOTIONS_OR_AUS = "aus"  
SHOW_EMOTIONS_OR_AUS = "emos"  
# SHOW_EMOTIONS_OR_AUS = "makeds"

if SHOW_EMOTIONS_OR_AUS == "makeds" and MASK_TYPE != "full":
    raise ValueError("For making the occluded test dataset, please set MASK_TYPE to 'full'.")

SAVE_IMAGES = True  # saves single images
SAVE_PLOTS = True  # saves the plots with many images and masks
FORCE_REMAP = False  # forces remapping of the landmarks

# ==== Dataset ====
DSFOLDER = "bosphorus_test_HQ"
OUTPUTFOLDER = os.path.join("saliency_maps", "zzz_other_and_zips", "output_occlusions")
if SHOW_EMOTIONS_OR_AUS == "makeds":
    OUTPUTFOLDER = "output_images_testset"
SUBJECTS = ["bs031", "bs003", "bs006", "bs030"]
# SUBJECTS = ["bs031"]

# ==== Action Units and Emotions dictioaries ====
# EMOTION_AUS = {
#     "ANGRY":    ["AU4", "AU23"],
#     "DISGUST":  ["AU9", "AU10", "AU16"],
#     "FEAR":     ["AU1", "AU2", "AU27"],
#     "HAPPY":    ["AU6", "AU12"],
#     "SAD":      ["AU1", "AU15"],
#     "SURPRISE": ["AU1", "AU2", "AU5", "AU26"]
# }

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

LANDMARKS = LandmarkDict({
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

# ============================================================================================================

def find_images_paths(subject):
    """
        Example output:
        {
            "HAPPY": "HAPPY/bs031_HAPPY_0001.png",
            "ANGRY": "ANGRY/bs031_ANGRY_0002.png",
            "SURPRISE": "SURPRISE/bs031_SURPRISE_0003.png"
        }
    """
    images_paths = dict()
    for emotion in os.listdir(DSFOLDER):
        if emotion == "NEUTRAL": continue # Skip neutral emotion as it does not have AUs
        sub_filename = f"{subject}_{emotion}"
        for filename in os.listdir(os.path.join(DSFOLDER, emotion)):
            if sub_filename in filename:
                images_paths[emotion] = f"{emotion}/{filename}"
                break
    return images_paths

def get_emotion_by_au(au):
    """
    Returns the emotion associated with the given AU.
    """
    for emotion, aus in EMOTION_AUS.items():
        # print(f"Checking emotion: {emotion} for AU: {au}")
        if au in aus:
            return emotion
        elif au == "AU1" or au == "AU2":
            return "FEAR"
        elif au == "AU5":
            return "SURPRISE"  # AU5 is associated with SURPRISE in AU1+2+5
    return None

def spin_emotion(spin: int, emotion_to_exclude: str = None) -> str:
    emotions = list(EMOTION_AUS.keys())
    emotions.sort()
    if emotion_to_exclude and emotion_to_exclude in emotions:
        emotions.remove(emotion_to_exclude)
    return emotions[spin]

# ============================================================================================================


if __name__ == "__main__":
    # 1) ========== Make a dict for the image paths. Based on subject, then emotion ========== 
    # NOTE: filename already contains the emotion as folder
    filenames_dict = dict()
    for subject in SUBJECTS:
        images_paths = find_images_paths(subject)
        filenames_dict[subject] = images_paths
    # print(f"Filenames dictionary: {filenames_dict}")

    # 2) ========== Make a similar dict but for the coordinates. Based on subject, then emotion ========== 
    landmark_coordinates_dict = dict()
    for i, subject in enumerate(SUBJECTS):
        # print(f"Processing subject: {subject}")
        subject_filenames = filenames_dict[subject]

        # Plot all images with facial landmarks for the current subject
        for j, (emotion, path) in enumerate(subject_filenames.items()):
            landmark_coordinates = load_landmark_coordinates(path)
            
            if landmark_coordinates is None or FORCE_REMAP:
                print(f"Detecting landmarks for {subject} - {emotion} from {path}")
                img_path = os.path.join(DSFOLDER, path)
                landmark_coordinates = detect_facial_landmarks(img_path)
                save_landmark_coordinates(path, landmark_coordinates)
            else:
                print(f"Loading cached landmarks for {subject} - {emotion} from {path}")

            subject_emotion = landmark_coordinates_dict.get(subject, {})
            subject_emotion[emotion] = landmark_coordinates
            landmark_coordinates_dict[subject] = subject_emotion       

    # 3.1) ========== Paste the landmarks on each image. Based on MATCHING emotion (e.g. happiness will have landmarks on the AUs for happiness) ========== 
    if SHOW_EMOTIONS_OR_AUS == "emos":
        nof_subjects = len(SUBJECTS)
        nof_emotions = len(filenames_dict[SUBJECTS[0]])
        
        fig, axes = plt.subplots(nof_subjects, nof_emotions, figsize=(15, 2.2 * nof_subjects))
        fig.suptitle(f"Facial Landmarks for Subjects: {", ".join(SUBJECTS)}", fontsize=16)

        for i, subject in enumerate(SUBJECTS):
            print(f"Processing subject: {subject}")
            subject_filenames = filenames_dict[subject]

            # Plot all images with facial landmarks for the current subject
            for j, (emotion, path) in enumerate(subject_filenames.items()):
                img_path = os.path.join(DSFOLDER, path)
                
                AUs = EMOTION_AUS[emotion]
                latest_img = cv2.imread(img_path)
                au_configs = []
                # Plot the masks AU by AU
                for au in AUs:
                    # Get the coordinates
                    for landmark_set in LANDMARKS[au]:
                        landmark_coordinates = []
                        for landmark_idx in landmark_set:
                            landmark_coordinates.append(landmark_coordinates_dict[subject][emotion][landmark_idx])

                        if landmark_set[0] == landmark_set[-1]:
                            closed_curve = True
                        else:
                            closed_curve = False

                        # Plot the landmarks. Note that different AUs may want use different drawing settings, which is why we're doing this in a loop AU by AU
                        if MASK_TYPE == "dots":
                            latest_img = mask_face_dots(latest_img, landmark_coordinates)
                        if MASK_TYPE == "circles":
                            latest_img = mask_face_circles(latest_img, landmark_coordinates, mask_color=MASK_COLOR)
                        elif MASK_TYPE == "full":
                            latest_img = mask_face_lines(latest_img, landmark_coordinates, fill=closed_curve, mask_color=MASK_COLOR) # green: 0, 255, 110
                        elif MASK_TYPE == "inverse":
                            au_configs.append({"landmark_coordinates": landmark_coordinates, 
                                                "fill": closed_curve})
                        else:
                            raise ValueError(f"Unknown mask type: {MASK_TYPE}")
                        
                if MASK_TYPE == "inverse":
                    latest_img = apply_inverse_masks(latest_img, au_configs, mask_color=MASK_COLOR)

                annotated_image = latest_img
                if SAVE_IMAGES == True:
                    save_path = os.path.join(OUTPUTFOLDER, f"{path[:-4]}_masked-{MASK_TYPE}-{emotion}.png")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure all subfolders exist
                    cv2.imwrite(save_path, annotated_image)
                    print(f"Saved image with landmarks to {save_path}")

                # axes[i].imshow(cv2.cvtColor(landmarks, cv2.COLOR_BGR2RGB))
                # axes[i].set_title(emotion)
                # axes[i].axis("off")
                if nof_subjects == 1:
                    axes[j].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                    axes[j].set_title(f"{subject} - {emotion}")
                    axes[j].axis("off")
                else:
                    axes[i, j].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                    axes[i, j].set_title(f"{subject} - {emotion}")
                    axes[i, j].axis("off")
                
        if SAVE_PLOTS:
            subjects_str = "-".join(SUBJECTS)
            output_path = os.path.join(OUTPUTFOLDER, f"emotions_for_subjects_{subjects_str}_mask-{MASK_TYPE}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Saved plot with emotions to {output_path}")
        plt.show()
    # 3.2) ========== Paste the landmarks on first subject's images. Paste AUs one by one on some CHOSEN emotion ========== 
    elif SHOW_EMOTIONS_OR_AUS == "aus":
        subject = SUBJECTS[0] 
        emotion_base = "SURPRISE"  
        print(f"Displaying AUs for subject {subject}")

        nof_rows = 3
        AUs_per_row = 5  # total max 15 AUs (we have 14)
        fig, axes = plt.subplots(nof_rows, AUs_per_row, figsize=(15, 2.2 * AUs_per_row))
        fig.suptitle(f"Facial Landmarks for all AUs on subject {subject}", fontsize=16)

        k = 0
        for au in AU_SUBSET.keys():
            emotion = get_emotion_by_au(au)  # Get the emotion associated with the AU
            path = filenames_dict[subject][emotion]  # Get the path for the emotion
            img_path = os.path.join(DSFOLDER, path)
            latest_img = cv2.imread(img_path)   

            # Get the coordinates
            au_configs = []
            for landmark_set in LANDMARKS[au]:
                landmark_coordinates = []
                for landmark_idx in landmark_set:
                    landmark_coordinates.append(landmark_coordinates_dict[subject][emotion][landmark_idx])

                if landmark_set[0] == landmark_set[-1]:
                    closed_curve = True
                else:
                    closed_curve = False

                # Plot the landmarks. Note that different AUs may want use different drawing settings, which is why we're doing this in a loop AU by AU
                if MASK_TYPE == "dots":
                    latest_img = mask_face_dots(latest_img, landmark_coordinates)
                if MASK_TYPE == "circles":
                    latest_img = mask_face_circles(latest_img, landmark_coordinates, mask_color=MASK_COLOR)
                elif MASK_TYPE == "full":
                    latest_img = mask_face_lines(latest_img, landmark_coordinates, fill=closed_curve, mask_color=MASK_COLOR)
                elif MASK_TYPE == "inverse":
                    au_configs.append({"landmark_coordinates": landmark_coordinates, 
                                        "fill": closed_curve})
                else:
                    raise ValueError(f"Unknown mask type: {MASK_TYPE}")
            
            if MASK_TYPE == "inverse":
                latest_img = apply_inverse_masks(latest_img, au_configs, mask_color=MASK_COLOR)

            annotated_image = latest_img
            if SAVE_IMAGES == True:
                filename_only = os.path.basename(path)  # Get only the filename, not the full path
                save_path = os.path.join(OUTPUTFOLDER, f"{au}/{filename_only[:-4]}_masked-{MASK_TYPE}-{au}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure all subfolders exist
                cv2.imwrite(save_path, annotated_image)
                print(f"Saved image with landmarks to {save_path}")

            i = k // AUs_per_row
            j = k % AUs_per_row
            axes[i, j].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            axes[i, j].set_title(f"{subject} - {au} ({emotion})")
            axes[i, j].axis("off")
                
            k += 1

        if SAVE_PLOTS:
            output_path = os.path.join(OUTPUTFOLDER, f"aus_for_subject_{subject}_mask-{MASK_TYPE}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Saved plot with AUs to {output_path}")
        plt.show()
    # elif SHOW_EMOTIONS_OR_AUS == "nomatch": # TODO: implement new things here too
        # pass
        # for k in range(6):
        #     nof_subjects = len(SUBJECTS)
        #     nof_emotions = len(filenames_dict[SUBJECTS[0]])
            
        #     row_heights = [0.2] + [1] * nof_subjects  # First row is smaller, others are equal
        #     fig, axes = plt.subplots(
        #         nrows=nof_subjects + 1, 
        #         ncols=nof_emotions, 
        #         figsize=(15, 2.2 * (nof_subjects + 1)),
        #         gridspec_kw={'height_ratios': row_heights}  # Adjust row heights
        #     )
        #     fig.suptitle(f"Facial Landmarks for Subjects: {', '.join(SUBJECTS)}", fontsize=16)
        #     plt.subplots_adjust(top=0.9)  # Adjust the top margin (closer to 1 means less space)

        #     # Add emotion labels in the first row
        #     for j, emotion in enumerate(filenames_dict[SUBJECTS[0]].keys()):
        #         axes[0, j].text(0.5, 0.5, emotion, fontsize=10, ha='center', va='center')  # Smaller font size
        #         axes[0, j].axis('off')  # Hide the axes for the label row

        #     for i, subject in enumerate(SUBJECTS):
        #         print(f"Processing subject: {subject}")
        #         subject_filenames = filenames_dict[subject]

        #         # Plot all images with facial landmarks for the current subject
        #         for j, (emotion, path) in enumerate(subject_filenames.items()):
        #             img_path = os.path.join(DSFOLDER, path)
        #             emotion = spin_emotion(k)  # Spin the emotion to get a different one each time  
                    
        #             AUs = EMOTION_AUS[emotion]
        #             latest_img = cv2.imread(img_path)
        #             # Plot the masks AU by AU
        #             for au in AUs:
        #                 # Get the coordinates
        #                 landmark_coordinates = []
        #                 for landmark_idx in LANDMARKS[au]:
        #                     landmark_coordinates.append(landmark_coordinates_dict[subject][emotion][landmark_idx])

        #                 # Plot the landmarks. Note that different AUs may want use different drawing settings, which is why we're doing this in a loop AU by AU
        #                 if MASK_TYPE == "dots":
        #                     latest_img = mask_face_dots(latest_img, landmark_coordinates)
        #                 elif MASK_TYPE == "full":
        #                     latest_img = mask_face_lines(latest_img, landmark_coordinates)
                            
        #             annotated_image = latest_img
        #             if SAVE_IMAGES == True:
        #                 save_path = os.path.join(OUTPUTFOLDER, f"{path}_masked-{MASK_TYPE}-{emotion}.png")
        #                 os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure all subfolders exist
        #                 cv2.imwrite(save_path, annotated_image)
        #                 print(f"Saved image with landmarks to {save_path}")


        #             axes[i + 1, j].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        #             axes[i + 1, j].set_title(f"{subject} - {emotion}")
        #             axes[i + 1, j].axis("off")
                    
        #     plt.show()
    elif SHOW_EMOTIONS_OR_AUS == "makeds":
        # We want to make the dataset as described in docs/occluded_testset_composition.md
        # 1) Start with the non neutral emotions
        emotions = list(EMOTION_AUS.keys()) # neutral is already not included

        # 1.1) Setup the amounts
        nof_images_tot = 50
        nof_match_occlusions = 10
        nof_mismatch_occlusions = nof_images_tot - nof_match_occlusions
        nof_occlusion_types = 5 * 2  # 5 occlusion types (each unmatched emotion), each with +/- occlusion
        nof_images_per_occlusion_type = nof_mismatch_occlusions // nof_occlusion_types
        if nof_images_tot % nof_occlusion_types != 0:
            raise ValueError(f"Number of images {nof_images_tot} is not divisible by number of occlusion types {nof_occlusion_types}")

        for emotion in emotions:
            # 1.2) Gather the faces in an array and shuffle
            imagenames = os.listdir(os.path.join(DSFOLDER, emotion))
            imagenames = [name for name in imagenames if name.endswith('.png')]
            if len(imagenames) != 50:
                raise ValueError(f"Expected 50 images for emotion {emotion}, but found {len(imagenames)}")
            
            random.shuffle(imagenames)

            # 1.3) Do the matched occlusions first then the mismatched combinations
            for i in range(0, nof_images_tot):
                img_path = os.path.join(DSFOLDER, emotion, imagenames[i])
                landmark_coordinates_all = load_landmark_coordinates(img_path)
                
                if landmark_coordinates_all is None or FORCE_REMAP:
                    landmark_coordinates_all = detect_facial_landmarks(img_path)
                    save_landmark_coordinates(img_path, landmark_coordinates_all)

                if i < nof_match_occlusions:
                    chosen_emotion = emotion
                    AUs = EMOTION_AUS[chosen_emotion]
                    mask_type = "positive"
                    print(f"Using matched emotion for image {i}. mask_type: {mask_type}")
                else:
                    mism_emotion_index = (i - nof_match_occlusions) // (nof_images_per_occlusion_type * 2)
                    chosen_emotion = spin_emotion(mism_emotion_index, emotion_to_exclude=emotion)
                    AUs = EMOTION_AUS[chosen_emotion]
                    if i % 2 == 0:
                        mask_type = "positive"
                    else:
                        mask_type = "negative"
                    print(f"Using mismatched emotion index: {mism_emotion_index} for image {i}. mask_type: {mask_type}")
                
                latest_img = cv2.imread(img_path)
                au_configs = []
                # Plot the masks AU by AU
                for au in AUs:
                    # Get the coordinates
                    for landmark_set in LANDMARKS[au]:
                        landmark_coordinates = []
                        for landmark_idx in landmark_set:
                            landmark_coordinates.append(landmark_coordinates_all[landmark_idx])

                        if landmark_set[0] == landmark_set[-1]:
                            closed_curve = True
                        else:
                            closed_curve = False

                        if mask_type == "positive":
                            latest_img = mask_face_lines(latest_img, landmark_coordinates, fill=closed_curve, mask_color=MASK_COLOR) # green: 0, 255, 110
                        elif mask_type == "negative":
                            au_configs.append({"landmark_coordinates": landmark_coordinates, 
                                                "fill": closed_curve})
                        else:
                            raise ValueError(f"Unknown mask type: {mask_type}")

                if mask_type == "negative":
                    latest_img = apply_inverse_masks(latest_img, au_configs, mask_color=MASK_COLOR)

                annotated_image = latest_img

                if i < nof_match_occlusions:
                    save_path = os.path.join(OUTPUTFOLDER, f"{img_path[:-11]}__masked-{mask_type}-{chosen_emotion}_match.png")
                else:
                    save_path = os.path.join(OUTPUTFOLDER, f"{img_path[:-11]}__masked-{mask_type}-{chosen_emotion}_mismatch.png")
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure all subfolders exist
                cv2.imwrite(save_path, annotated_image)
                print(f"Saved image with landmarks to {save_path}")
        
        # 2) Now we want to add the neutral emotion images
        emotion = "NEUTRAL"
        nof_occlusion_types = 6 * 2  # 5 occlusion types (each unmatched emotion), each with +/- occlusion
        nof_images_per_occlusion_type = nof_images_tot // nof_occlusion_types
        nof_extras = nof_images_tot - (nof_occlusion_types * nof_images_per_occlusion_type)
        print(f"Number of extras: {nof_extras}")

        # 2.1) Gather the faces in an array and shuffle
        imagenames = os.listdir(os.path.join(DSFOLDER, emotion))
        imagenames = [name for name in imagenames if name.endswith('.png')]
        if len(imagenames) != 50:
            raise ValueError(f"Expected 50 images for emotion {emotion}, but found {len(imagenames)}")
        
        random.shuffle(imagenames)

        # 2.2) Create the occlusions
        for i in range(0, nof_images_tot):
            img_path = os.path.join(DSFOLDER, emotion, imagenames[i])
            landmark_coordinates_all = load_landmark_coordinates(img_path)
            
            if landmark_coordinates_all is None or FORCE_REMAP:
                print(f"Detecting landmarks for {emotion} from {img_path}")
                landmark_coordinates_all = detect_facial_landmarks(img_path)
                save_landmark_coordinates(img_path, landmark_coordinates_all)
            else:
                print(f"Loading cached landmarks for {emotion} from {img_path}")

            if i < (nof_images_tot - nof_extras):
                mism_emotion_index = i // (nof_images_per_occlusion_type * 2)
                chosen_emotion = spin_emotion(mism_emotion_index)
                AUs = EMOTION_AUS[chosen_emotion]
                if i % 2 == 0:
                    mask_type = "positive"
                else:
                    mask_type = "negative"
                print(f"[{emotion}] Using mismatched emotion index: {mism_emotion_index} for image {i}. mask_type: {mask_type}")
            else:
                chosen_emotion = spin_emotion(random.randint(0, len(EMOTION_AUS) - 1))
                AUs = EMOTION_AUS[chosen_emotion]
                if i % 2 == 0:
                    mask_type = "positive"
                else:
                    mask_type = "negative"
                print(f"[{emotion}] Using mismatched emotion index: {mism_emotion_index} for image {i}. mask_type: {mask_type}")
            
            latest_img = cv2.imread(img_path)
            au_configs = []
            # Plot the masks AU by AU
            for au in AUs:
                # Get the coordinates
                for landmark_set in LANDMARKS[au]:
                    landmark_coordinates = []
                    for landmark_idx in landmark_set:
                        landmark_coordinates.append(landmark_coordinates_all[landmark_idx])

                    if landmark_set[0] == landmark_set[-1]:
                        closed_curve = True
                    else:
                        closed_curve = False

                    if mask_type == "positive":
                        latest_img = mask_face_lines(latest_img, landmark_coordinates, fill=closed_curve, mask_color=MASK_COLOR) # green: 0, 255, 110
                    elif mask_type == "negative":
                        au_configs.append({"landmark_coordinates": landmark_coordinates, 
                                            "fill": closed_curve})
                    else:
                        raise ValueError(f"Unknown mask type: {mask_type}")

            if mask_type == "negative":
                latest_img = apply_inverse_masks(latest_img, au_configs, mask_color=MASK_COLOR)

            annotated_image = latest_img

            save_path = os.path.join(OUTPUTFOLDER, f"{img_path[:-11]}__masked-{mask_type}-{chosen_emotion}_mismatch.png")
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure all subfolders exist
            cv2.imwrite(save_path, annotated_image)
            print(f"Saved image with landmarks to {save_path}")

