# CSM_per_persona.py (robusto + diagonale corretta + counts_matrix.csv)
# OUTPUT PULITO + LOOK IDENTICO ALL'ORIGINALE:
# - base face mostrata come nel tuo originale (cv2 BGR -> imshow => "patina azzurra")
# - overlay "jet" SOLO sul volto (fuori volto trasparente tramite face_mask)
# - overlay NON viene disegnata se la mappa è vuota/quasi-vuota (EPS) -> niente "quadrato blu"
# - contorni puliti: maschera = CONVEX HULL dei landmark canonical
# - salva heatmaps .npy e counts_matrix.csv

import os
import math
import warnings
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

warnings.filterwarnings("ignore")


# -------------------------
# Utils
# -------------------------
def _normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int
) -> Union[Tuple[int, int], Tuple[None, None]]:
    """Converts normalized value pair to pixel coordinates."""

    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        return None, None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def _mask_type_from_name(img_name: str) -> str:
    if "masked-positive" in img_name:
        return "positive"
    if "masked-negative" in img_name:
        return "negative"
    return "other"


def _norm01(m: np.ndarray) -> np.ndarray:
    """Normalizza in [0,1] robusto (no crash su costanti / nan)."""
    m = np.array(m, dtype=float)
    m[~np.isfinite(m)] = 0.0
    mmin = float(np.min(m))
    mmax = float(np.max(m))
    if mmax > mmin:
        return (m - mmin) / (mmax - mmin)
    return np.zeros_like(m)


def _face_mask_from_canonical_landmarks(face_landmarks, Fwidth: int, Fheight: int) -> np.ndarray:
    """
    Maschera pulita del volto: convex hull dei landmark canonical.
    True=volto, False=fuori volto.
    """
    pts = []
    for lm in face_landmarks:
        x, y = _normalized_to_pixel_coordinates(lm.x, lm.y, Fwidth, Fheight)
        if x is not None:
            pts.append([x, y])

    if len(pts) < 3:
        return np.ones((Fheight, Fwidth), dtype=bool)

    pts = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts)

    mask = np.zeros((Fheight, Fwidth), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


# -------------------------
# Heatmap
# -------------------------
def heatmap_gauss(gaze: pd.DataFrame, min_duration: float, max_duration: float) -> np.ndarray:
    """
    Heatmap 1161x1161 basata su fissazioni (gaze_x/gaze_y) e durata.
    NOTA: assume coordinate gaze_x/gaze_y in [0,1160].
    """
    gaze = gaze[
        (gaze["gaze_x"] >= 0) & (gaze["gaze_x"] <= 1160) &
        (gaze["gaze_y"] >= 0) & (gaze["gaze_y"] <= 1160)
    ]

    if gaze.empty:
        return np.zeros((1161, 1161), dtype=float)

    gaze_x = gaze["gaze_x"].values
    gaze_y = gaze["gaze_y"].values
    duration = gaze["Duration"].values.astype(float)

    new_min, new_max = 50.0, 100.0

    # evita divisione per zero
    if max_duration is None or min_duration is None or max_duration == min_duration:
        duration = np.full_like(duration, new_min, dtype=float)
    else:
        duration = (duration - min_duration) / (max_duration - min_duration) * (new_max - new_min) + new_min

    x = np.arange(0, 1161)
    y = np.arange(0, 1161)
    X, Y = np.meshgrid(x, y)

    heatmap = np.zeros((1161, 1161), dtype=float)

    for i in range(len(duration)):
        sigma = float(duration[i])
        scale = sigma / new_min
        gaussian = np.exp(-((X - gaze_x[i]) ** 2 + (Y - gaze_y[i]) ** 2) / (2 * sigma ** 2)) * scale
        heatmap += gaussian

    hmin = float(np.min(heatmap))
    hmax = float(np.max(heatmap))
    if hmax == hmin:
        return np.zeros_like(heatmap)

    return (heatmap - hmin) / (hmax - hmin)


# -------------------------
# Main
# -------------------------
def main(output_path: str, limite: float, name: str):
    # Paths
    imeges_path_norm = os.path.join("..", "bosphorus_test_HQ")
    canonical_faces_path = os.path.join("..", "saliency_maps", "basefaces", "Canonical Faces", "Canonical faces original")
    model_path = os.path.join("..", "saliency_maps", "mediapipe", "face_landmarker.task")


    # Plot params
    al = 0.6
    lim = limite

    # Anti-rumore overlay (EVITA "aloni blu" su celle che devono restare "base only")
    EPS = 1e-6

    # Files
    df_path = os.path.join(output_path, f"{name}.csv")
    fix_path = os.path.join("..", "saliency_maps/HEATMAPS_ale/Results", name, "output_csv", "output.csv")

    # Sanity checks
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Missing subject CSV: {df_path}")
    if not os.path.exists(fix_path):
        raise FileNotFoundError(f"Missing fixations CSV: {fix_path}")

    # MediaPipe detector
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Load CSVs
    df = pd.read_csv(df_path)
    print(df.head())

    needed_cols = {"Real_Emotion", "Image_Name", "Answer"}
    if not needed_cols.issubset(df.columns):
        raise ValueError(f"{df_path} must contain columns: {sorted(list(needed_cols))}")

    fix_df = pd.read_csv(fix_path)
    for col in ["Image_Name", "gaze_x", "gaze_y", "Duration"]:
        if col not in fix_df.columns:
            raise ValueError(f"{fix_path} is missing column: {col}")

    max_duration = float(fix_df["Duration"].max())
    min_duration = float(fix_df["Duration"].min())

    # Fixed emotions order + column mapping
    emotions = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]
    col_map = {e: i for i, e in enumerate(emotions)}

    # Counters matrix (True x Predicted) for USED images only
    counts_matrix = np.zeros((7, 7), dtype=int)
    original_counts = {e: 0 for e in emotions}   # total rows in df per emotion
    processed_counts = {e: 0 for e in emotions}  # images actually used (post-skip)

    # Figure 7x7
    fig, axs = plt.subplots(7, 7, figsize=(20, 20))
    fig.suptitle("Canonical Model Saliency Maps", fontsize=28)
    fig.supylabel("True", fontsize=28)

    for i in range(7):
        for j in range(7):
            axs[i, j].axis("off")

    # Row labels + Column titles
    row_names = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    for r, label in enumerate(row_names):
        axs[r, 0].text(-0.1, 0.5, label, va="center", ha="right",
                       transform=axs[r, 0].transAxes, fontsize=20)
    for c, title in enumerate(row_names):
        axs[0, c].set_title(title, fontsize=20)

    # Output dirs
    heatmap_dir = os.path.join(output_path, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # Failure log (saved at end)
    fail_log = []

    # Overlay colormap: masked -> transparent
    cmap_overlay = plt.cm.jet.copy()
    cmap_overlay.set_bad(alpha=0)

    def _plot_base_and_overlay(ax, base_bgr: np.ndarray, face_mask_local: np.ndarray, overlay: np.ndarray | None):
        """
        Base = come originale: cv2 BGR passato diretto a imshow (-> patina azzurra).
        Overlay = jet SOLO nel volto (mask), fuori trasparente.
        """
        ax.imshow(base_bgr)  # INTENZIONALE: look identico all'originale (BGR->imshow)
        if overlay is None:
            return

        ov_raw = np.array(overlay, dtype=float)
        ov_raw[~np.isfinite(ov_raw)] = 0.0

        # se è praticamente vuoto -> non disegnare overlay
        if float(np.max(ov_raw)) <= EPS:
            return

        ov = _norm01(ov_raw)
        ov_masked = np.ma.masked_where(~face_mask_local, ov)
        ax.imshow(ov_masked, alpha=al, cmap=cmap_overlay, vmin=0, vmax=lim)

    # Loop emotions (rows)
    for riga, emotion in enumerate(emotions):
        images = df[df["Real_Emotion"] == emotion]["Image_Name"].values
        original_counts[emotion] = len(images)

        print(emotion)

        # Canonical face + landmarks
        canonical_png = os.path.join(canonical_faces_path, f"{emotion}.png")
        if not os.path.exists(canonical_png):
            fail_log.append(
                {"name": name, "true_emotion": emotion, "image_name": canonical_png,
                 "mask_type": "n/a", "fail_reason": "missing_canonical_png"}
            )
            print(f"[SKIP] Missing canonical face png: {canonical_png}")
            continue

        canonical_face_mp = mp.Image.create_from_file(canonical_png)
        Fwidth, Fheight = canonical_face_mp.width, canonical_face_mp.height
        Fdetection_result = detector.detect(canonical_face_mp)
        face_landmarks_list = Fdetection_result.face_landmarks

        if not face_landmarks_list:
            fail_log.append(
                {"name": name, "true_emotion": emotion, "image_name": canonical_png,
                 "mask_type": "n/a", "fail_reason": "no_face_landmarks_on_canonical"}
            )
            print(f"[SKIP] No face landmarks on canonical {emotion}.png")
            continue

        num_landmarks = len(face_landmarks_list[0])

        # base face come originale (BGR)
        face_bgr = cv2.imread(canonical_png)
        if face_bgr is None:
            fail_log.append(
                {"name": name, "true_emotion": emotion, "image_name": canonical_png,
                 "mask_type": "n/a", "fail_reason": "cannot_read_canonical_png"}
            )
            print(f"[SKIP] Cannot read canonical: {canonical_png}")
            continue

        # assicurati dimensioni coerenti
        if face_bgr.shape[1] != Fwidth or face_bgr.shape[0] != Fheight:
            face_bgr = cv2.resize(face_bgr, (Fwidth, Fheight), interpolation=cv2.INTER_AREA)

        # maschera volto pulita dai landmark canonical
        face_mask = _face_mask_from_canonical_landmarks(face_landmarks_list[0], Fwidth, Fheight)

        # Accumulators (maps)
        canonical_heatmap = np.zeros((Fheight, Fwidth), dtype=float)
        tot_correct = 0  # diagonale

        tot_Anger = tot_Disgust = tot_Fear = tot_Happiness = tot_Neutral = tot_Sadness = tot_Surprise = 0
        CSM_Anger = np.zeros((Fheight, Fwidth), dtype=float)
        CSM_Disgust = np.zeros((Fheight, Fwidth), dtype=float)
        CSM_Fear = np.zeros((Fheight, Fwidth), dtype=float)
        CSM_Happiness = np.zeros((Fheight, Fwidth), dtype=float)
        CSM_Neutral = np.zeros((Fheight, Fwidth), dtype=float)
        CSM_Sadness = np.zeros((Fheight, Fwidth), dtype=float)
        CSM_Surprise = np.zeros((Fheight, Fwidth), dtype=float)

        # Stats pos/neg (debug)
        stats = {
            "positive": {"tot": 0, "ok": 0, "no_face": 0, "nan_grid": 0, "missing_img": 0},
            "negative": {"tot": 0, "ok": 0, "no_face": 0, "nan_grid": 0, "missing_img": 0},
            "other": {"tot": 0, "ok": 0, "no_face": 0, "nan_grid": 0, "missing_img": 0},
        }

        # Loop images for this true emotion
        for img in images:
            img = str(img)
            mask_type = _mask_type_from_name(img)
            stats[mask_type]["tot"] += 1

            # predicted
            try:
                predicted_class = df[df["Image_Name"] == img]["Answer"].values[0]
                predicted_class = str(predicted_class).strip().upper()
            except Exception:
                fail_log.append({"name": name, "true_emotion": emotion, "image_name": img,
                                 "mask_type": mask_type, "fail_reason": "missing_answer_in_df"})
                print(f"[SKIP missing-answer] {emotion} | {img}")
                continue

            # path image
            image_name = img.split(".")[0]
            img_path = os.path.join(imeges_path_norm, emotion, f"{image_name}.png")

            if not os.path.exists(img_path):
                stats[mask_type]["missing_img"] += 1
                fail_log.append({"name": name, "true_emotion": emotion, "image_name": img_path,
                                 "mask_type": mask_type, "fail_reason": "missing_image_file"})
                print(f"[SKIP missing-img] {emotion} | {img_path}")
                continue

            # detect landmarks on the image
            image_mp = mp.Image.create_from_file(img_path)
            Iwidth, Iheight = image_mp.width, image_mp.height
            Idetection_result = detector.detect(image_mp)
            image_landmarks_list = Idetection_result.face_landmarks

            if not image_landmarks_list:
                stats[mask_type]["no_face"] += 1
                fail_log.append({"name": name, "true_emotion": emotion, "image_name": img,
                                 "mask_type": mask_type, "fail_reason": "no_face_landmarks"})
                print(f"[SKIP no-face] {emotion} | {img}")
                continue

            # build heatmap
            gaze = fix_df[fix_df["Image_Name"] == img]
            heatmap = heatmap_gauss(gaze, min_duration, max_duration)

            # heatmap values at image landmarks
            heatmap_values = []
            n_img_landmarks = len(image_landmarks_list[0])
            n_use = min(num_landmarks, n_img_landmarks)

            for idx in range(n_use):
                lm = image_landmarks_list[0][idx]
                x, y = _normalized_to_pixel_coordinates(lm.x, lm.y, Iwidth, Iheight)
                if x is None:
                    heatmap_values.append(0.0)
                else:
                    heatmap_values.append(float(heatmap[y, x]))

            if len(heatmap_values) < num_landmarks:
                heatmap_values.extend([0.0] * (num_landmarks - len(heatmap_values)))

            # map to canonical landmark coordinates
            coordinates_face = np.zeros((num_landmarks, 3), dtype=float)
            for idx in range(num_landmarks):
                flm = face_landmarks_list[0][idx]
                x, y = _normalized_to_pixel_coordinates(flm.x, flm.y, Fwidth, Fheight)
                if x is None:
                    x, y = 0, 0
                z = heatmap_values[idx]
                coordinates_face[idx, :] = np.array([x, y, z], dtype=float)

            points = coordinates_face[:, 0:2]
            grid_x, grid_y = np.mgrid[0:Fwidth:75j, 0:Fheight:85j]

            try:
                grid_z = griddata(points, coordinates_face[:, 2], (grid_x, grid_y), method="cubic")
            except Exception:
                stats[mask_type]["nan_grid"] += 1
                fail_log.append({"name": name, "true_emotion": emotion, "image_name": img,
                                 "mask_type": mask_type, "fail_reason": "griddata_exception"})
                print(f"[SKIP grid-exc] {emotion} | {img}")
                continue

            if grid_z is None:
                stats[mask_type]["nan_grid"] += 1
                fail_log.append({"name": name, "true_emotion": emotion, "image_name": img,
                                 "mask_type": mask_type, "fail_reason": "griddata_none"})
                print(f"[SKIP grid-none] {emotion} | {img}")
                continue

            grid_z = np.array(grid_z, dtype=float)
            grid_z[~np.isfinite(grid_z)] = 0.0
            grid_z[grid_z < 0] = 0.0
            grid_z = grid_z.T

            if np.isnan(grid_z).all():
                stats[mask_type]["nan_grid"] += 1
                fail_log.append({"name": name, "true_emotion": emotion, "image_name": img,
                                 "mask_type": mask_type, "fail_reason": "grid_all_nan"})
                print(f"[SKIP nan-grid] {emotion} | {img}")
                continue

            # IMAGE IS USED (post-skip)
            stats[mask_type]["ok"] += 1
            processed_counts[emotion] += 1

            # update counts matrix (True x Pred) for used images only
            if predicted_class in col_map:
                counts_matrix[col_map[emotion], col_map[predicted_class]] += 1

            # accumulate maps for plotting/saving
            if predicted_class == emotion:
                tot_correct += 1
                canonical_heatmap += grid_z
            else:
                if predicted_class == "ANGRY":
                    CSM_Anger += grid_z
                    tot_Anger += 1
                elif predicted_class == "DISGUST":
                    CSM_Disgust += grid_z
                    tot_Disgust += 1
                elif predicted_class == "FEAR":
                    CSM_Fear += grid_z
                    tot_Fear += 1
                elif predicted_class == "HAPPY":
                    CSM_Happiness += grid_z
                    tot_Happiness += 1
                elif predicted_class == "NEUTRAL":
                    CSM_Neutral += grid_z
                    tot_Neutral += 1
                elif predicted_class == "SAD":
                    CSM_Sadness += grid_z
                    tot_Sadness += 1
                elif predicted_class == "SURPRISE":
                    CSM_Surprise += grid_z
                    tot_Surprise += 1

        # debug prints
        print(
            f"[{name}] {emotion} | correct_used={tot_correct} | "
            f"POS tot={stats['positive']['tot']} ok={stats['positive']['ok']} no_face={stats['positive']['no_face']} nan={stats['positive']['nan_grid']} missing_img={stats['positive']['missing_img']} || "
            f"NEG tot={stats['negative']['tot']} ok={stats['negative']['ok']} no_face={stats['negative']['no_face']} nan={stats['negative']['nan_grid']} missing_img={stats['negative']['missing_img']} || "
            f"OTH tot={stats['other']['tot']} ok={stats['other']['ok']} no_face={stats['other']['no_face']} nan={stats['other']['nan_grid']} missing_img={stats['other']['missing_img']}"
        )

        # -------------------------
        # plotting (base sempre, overlay solo se c'è davvero)
        # -------------------------
        col_diag = col_map[emotion]
        _plot_base_and_overlay(
            axs[riga, col_diag],
            face_bgr,
            face_mask,
            canonical_heatmap if tot_correct > 0 else None
        )

        np.save(
            os.path.join(heatmap_dir, f"{emotion}_canonical.npy"),
            _norm01(canonical_heatmap) if (tot_correct > 0 and float(np.max(canonical_heatmap)) > EPS)
            else np.zeros((Fheight, Fwidth), dtype=float)
        )

        def _plot_and_save(mat: np.ndarray, count: int, pred_emotion: str, suffix: str):
            col_idx = col_map[pred_emotion]
            _plot_base_and_overlay(
                axs[riga, col_idx],
                face_bgr,
                face_mask,
                mat if count > 0 else None
            )
            np.save(
                os.path.join(heatmap_dir, f"{emotion}_{suffix}.npy"),
                _norm01(mat) if (count > 0 and float(np.max(mat)) > EPS)
                else np.zeros((Fheight, Fwidth), dtype=float)
            )

        if emotion != "ANGRY":
            _plot_and_save(CSM_Anger, tot_Anger, "ANGRY", "Angry")
        if emotion != "DISGUST":
            _plot_and_save(CSM_Disgust, tot_Disgust, "DISGUST", "Disgust")
        if emotion != "FEAR":
            _plot_and_save(CSM_Fear, tot_Fear, "FEAR", "Fear")
        if emotion != "HAPPY":
            _plot_and_save(CSM_Happiness, tot_Happiness, "HAPPY", "Happiness")
        if emotion != "NEUTRAL":
            _plot_and_save(CSM_Neutral, tot_Neutral, "NEUTRAL", "Neutral")
        if emotion != "SAD":
            _plot_and_save(CSM_Sadness, tot_Sadness, "SAD", "Sad")
        if emotion != "SURPRISE":
            _plot_and_save(CSM_Surprise, tot_Surprise, "SURPRISE", "Surprise")

    # Save figure
    plt.savefig(os.path.join(output_path, "CSM_CM.png"))

    # Save failure log
    if fail_log:
        pd.DataFrame(fail_log).to_csv(os.path.join(output_path, "failed_images_log.csv"), index=False)

    # Save counts matrix + totals
    counts_df = pd.DataFrame(counts_matrix, index=emotions, columns=emotions)
    counts_df["N_original"] = [original_counts[e] for e in emotions]
    counts_df["N_processed"] = [processed_counts[e] for e in emotions]
    counts_df.to_csv(os.path.join(output_path, "counts_matrix.csv"))


# -------------------------
# Run
# -------------------------
names = [
    "LUCRUG",
    "KRIJAK",
    "FILCOM",
    "LORLAF",
    "SAMPIG",
    "FILVEL",
    "ALELUC",
    "LORSCA",
    "STEALT",
    "STEPAL",
    "CARPAR",
    "CHISPA",
    "FRAMUC",
    "VIRTOM",
    "SILCAF",
    "ANDESP",
    "EMAFOS",
    "MATOCC",
    "NICRIB",
    "DOMTEL",
    "GIATRI",
    "MARRIC",
    "VALTAR",
    "SARGAR",
    "VITMER",
    "AMIEFT",
    "DRABUH",
    "EMAGRE",
    "ARMLAR",
    "GIOBRU",
    "ROCOLI",
    "ELETOR",
    "MARIAM",
    "SIMMAF",
    "VALRUO",
    "ROCLAB",
    "FABIAC",
    "ALEGRA",
    "ELEOLI",
    "MARFRO",
    "REBDEL",
    "GREBAZ",
    "FILGUA",
    "ALITOM",
    "MARCEC",
    "MARDIB",
    "SILVEN",
    "FEDMAR",
    "AMAESC",
    "REBLEO"
]

for name in names:
    output_path = f"../saliency_maps/HEATMAPS_ale/Results/{name}"
    limite = 1
    try:
        main(output_path, limite, name)
        print(f"{name} completed")
    except Exception as e:
        print(f"[FATAL] Error in {name}: {e}")