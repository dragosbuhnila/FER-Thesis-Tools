import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


# -----------------------------
# MASCHERA SFONDO (fuori faccia)
# -----------------------------
def background_mask_floodfill(face_bgr: np.ndarray, tol: int = 8) -> np.ndarray:
    """
    Ritorna bg_mask True dove è sfondo, usando flood-fill dagli angoli.
    tol controlla quanto “simile” deve essere il colore per essere considerato sfondo.
    """
    h, w = face_bgr.shape[:2]
    bg = np.zeros((h, w), dtype=bool)

    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

    for sx, sy in seeds:
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        tmp = face_bgr.copy()
        cv2.floodFill(
            tmp, ff_mask,
            seedPoint=(sx, sy),
            newVal=(0, 0, 0),
            loDiff=(tol, tol, tol),
            upDiff=(tol, tol, tol),
            flags=flags
        )
        bg |= (ff_mask[1:-1, 1:-1] == 255)

    # piccola pulizia
    k = np.ones((5, 5), np.uint8)
    bg_u8 = (bg.astype(np.uint8) * 255)
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_CLOSE, k, iterations=1)
    return bg_u8 > 0


def safe_minmax_norm(hm: np.ndarray) -> np.ndarray:
    """Min-max normalization robusta (gestisce NaN e caso costante)."""
    mn = np.nanmin(hm)
    mx = np.nanmax(hm)
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or (mx <= mn):
        return np.full_like(hm, np.nan, dtype=np.float32)
    return (hm - mn) / (mx - mn)


def apply_bg_nan(hm: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    out = hm.astype(np.float32, copy=True)
    out[bg_mask] = np.nan
    return out


# -----------------------------
# MAIN (layout IDENTICO al tuo)
# -----------------------------
def main(path, names):

    canonical_faces_path = 'Canonical faces cutted'

    # creare un subplot 7x7 (IDENTICO)
    fig, axs = plt.subplots(7, 7, figsize=(20, 20))
    fig.suptitle('Canonical Model Saliency Maps', fontsize=28)
    fig.suptitle('Predicted', fontsize=28)
    fig.supylabel('True', fontsize=28)

    # nascondere gli assi
    for i in range(7):
        for j in range(7):
            axs[i, j].axis('off')

    axs[0, 0].text(-0.1, 0.5, 'Anger', va='center', ha='right', rotation=0, transform=axs[0, 0].transAxes, fontsize=20)
    axs[1, 0].text(-0.1, 0.5, 'Disgust', va='center', ha='right', rotation=0,transform=axs[1, 0].transAxes, fontsize=20)
    axs[2, 0].text(-0.1, 0.5, 'Fear', va='center', ha='right', rotation=0,transform=axs[2, 0].transAxes, fontsize=20)
    axs[3, 0].text(-0.1, 0.5, 'Happiness', va='center', ha='right', rotation=0,transform=axs[3, 0].transAxes, fontsize=20)
    axs[4, 0].text(-0.1, 0.5, 'Neutral', va='center', ha='right', rotation=0,transform=axs[4, 0].transAxes, fontsize=20)
    axs[5, 0].text(-0.1, 0.5, 'Sadness', va='center', ha='right', rotation=0,transform=axs[5, 0].transAxes, fontsize=20)
    axs[6, 0].text(-0.1, 0.5, 'Surprise', va='center', ha='right', rotation=0,transform=axs[6, 0].transAxes, fontsize=20)

    axs[0, 0].set_title('Anger', fontsize=20)
    axs[0, 1].set_title('Disgust', fontsize=20)
    axs[0, 2].set_title('Fear', fontsize=20)
    axs[0, 3].set_title('Happiness', fontsize=20)
    axs[0, 4].set_title('Neutral', fontsize=20)
    axs[0, 5].set_title('Sadness', fontsize=20)
    axs[0, 6].set_title('Surprise', fontsize=20)

    riga = 0
    colonna = 0

    emotions = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']

    # Colormap: NaN trasparenti (fondamentale!)
    cmap = plt.cm.jet.copy()
    cmap.set_bad(alpha=0)

    for emotion in emotions:
        print(f"Processing {emotion}")

        face = cv2.imread(os.path.join(canonical_faces_path, f"{emotion}.png"))
        if face is None:
            raise FileNotFoundError(f"Non trovo {emotion}.png in {canonical_faces_path}")

        Fheight, Fwidth, _ = face.shape

        # bg_mask (sfondo) dalla faccia canonica
        bg_mask = background_mask_floodfill(face, tol=8)

        tot = 0
        tot_Anger = 0
        tot_Disgust = 0
        tot_Fear = 0
        tot_Happiness = 0
        tot_Neutral = 0
        tot_Sadness = 0
        tot_Surprise = 0

        canonical_heatmap = np.zeros((Fheight, Fwidth), dtype=np.float32)
        CSM_Anger = np.zeros((Fheight, Fwidth), dtype=np.float32)
        CSM_Disgust = np.zeros((Fheight, Fwidth), dtype=np.float32)
        CSM_Fear = np.zeros((Fheight, Fwidth), dtype=np.float32)
        CSM_Happiness = np.zeros((Fheight, Fwidth), dtype=np.float32)
        CSM_Neutral = np.zeros((Fheight, Fwidth), dtype=np.float32)
        CSM_Sadness = np.zeros((Fheight, Fwidth), dtype=np.float32)
        CSM_Surprise = np.zeros((Fheight, Fwidth), dtype=np.float32)

        for name in names:
            hm_dir = f"{path}/{name}/heatmaps"
            if not os.path.isdir(hm_dir):
                continue

            files = os.listdir(hm_dir)
            for file_name in files:
                emo = file_name.split('_')[0]
                if emo != emotion:
                    continue

                heatmap = np.load(f"{hm_dir}/{file_name}").astype(np.float32)

                # IMPORTANTISSIMO: metti NaN fuori faccia PRIMA di sommare
                heatmap = apply_bg_nan(heatmap, bg_mask)

                emoz = file_name.split('_')[1].split('.')[0]
                if emoz == 'canonical':
                    canonical_heatmap += np.nan_to_num(heatmap, nan=0.0)
                    tot += 1
                elif emoz == 'Angry':
                    CSM_Anger += np.nan_to_num(heatmap, nan=0.0)
                    tot_Anger += 1
                elif emoz == 'Disgust':
                    CSM_Disgust += np.nan_to_num(heatmap, nan=0.0)
                    tot_Disgust += 1
                elif emoz == 'Fear':
                    CSM_Fear += np.nan_to_num(heatmap, nan=0.0)
                    tot_Fear += 1
                elif emoz == 'Happiness':
                    CSM_Happiness += np.nan_to_num(heatmap, nan=0.0)
                    tot_Happiness += 1
                elif emoz == 'Neutral':
                    CSM_Neutral += np.nan_to_num(heatmap, nan=0.0)
                    tot_Neutral += 1
                elif emoz == 'Sad':
                    CSM_Sadness += np.nan_to_num(heatmap, nan=0.0)
                    tot_Sadness += 1
                elif emoz == 'Surprise':
                    CSM_Surprise += np.nan_to_num(heatmap, nan=0.0)
                    tot_Surprise += 1

        al = 0.6
        lim = 1

        folder_heatmaps = 'OUTPUTS/heatmaps'
        os.makedirs(folder_heatmaps, exist_ok=True)

        # --- canonical sulla diagonale ---
        if tot == 0:
            axs[riga, colonna].imshow(face)
        else:
            canonical_heatmap = canonical_heatmap / float(tot)
            canonical_heatmap = safe_minmax_norm(canonical_heatmap)
            canonical_heatmap = apply_bg_nan(canonical_heatmap, bg_mask)

            axs[riga, colonna].imshow(face)
            axs[riga, colonna].imshow(canonical_heatmap, alpha=al, cmap=cmap, vmin=0, vmax=lim)
            np.save(os.path.join(f'{folder_heatmaps}/{emotion}_canonical.npy'), canonical_heatmap)

        # --- colonne predette (off-diagonal) ---
        if tot_Anger == 0 and emotion != 'ANGRY':
            axs[riga, 0].imshow(face)
        elif emotion != 'ANGRY':
            CSM_Anger = CSM_Anger / float(tot_Anger)
            CSM_Anger = safe_minmax_norm(CSM_Anger)
            CSM_Anger = apply_bg_nan(CSM_Anger, bg_mask)
            axs[riga, 0].imshow(face)
            axs[riga, 0].imshow(CSM_Anger, alpha=al, cmap=cmap, vmin=0, vmax=lim)
            np.save(os.path.join(f'{folder_heatmaps}/{emotion}_Anger.npy'), CSM_Anger)

        if tot_Disgust == 0 and emotion != 'DISGUST':
            axs[riga, 1].imshow(face)
        elif emotion != 'DISGUST':
            CSM_Disgust = CSM_Disgust / float(tot_Disgust)
            CSM_Disgust = safe_minmax_norm(CSM_Disgust)
            CSM_Disgust = apply_bg_nan(CSM_Disgust, bg_mask)
            axs[riga, 1].imshow(face)
            axs[riga, 1].imshow(CSM_Disgust, alpha=al, cmap=cmap, vmin=0, vmax=lim)
            np.save(os.path.join(f'{folder_heatmaps}/{emotion}_Disgust.npy'), CSM_Disgust)

        if tot_Fear == 0 and emotion != 'FEAR':
            axs[riga, 2].imshow(face)
        elif emotion != 'FEAR':
            CSM_Fear = CSM_Fear / float(tot_Fear)
            CSM_Fear = safe_minmax_norm(CSM_Fear)
            CSM_Fear = apply_bg_nan(CSM_Fear, bg_mask)
            axs[riga, 2].imshow(face)
            axs[riga, 2].imshow(CSM_Fear, alpha=al, cmap=cmap, vmin=0, vmax=lim)
            np.save(os.path.join(f'{folder_heatmaps}/{emotion}_Fear.npy'), CSM_Fear)

        if tot_Happiness == 0 and emotion != 'HAPPY':
            axs[riga, 3].imshow(face)
        elif emotion != 'HAPPY':
            CSM_Happiness = CSM_Happiness / float(tot_Happiness)
            CSM_Happiness = safe_minmax_norm(CSM_Happiness)
            CSM_Happiness = apply_bg_nan(CSM_Happiness, bg_mask)
            axs[riga, 3].imshow(face)
            axs[riga, 3].imshow(CSM_Happiness, alpha=al, cmap=cmap, vmin=0, vmax=lim)
            np.save(os.path.join(f'{folder_heatmaps}/{emotion}_Happiness.npy'), CSM_Happiness)

        if tot_Neutral == 0 and emotion != 'NEUTRAL':
            axs[riga, 4].imshow(face)
        elif emotion != 'NEUTRAL':
            CSM_Neutral = CSM_Neutral / float(tot_Neutral)
            CSM_Neutral = safe_minmax_norm(CSM_Neutral)
            CSM_Neutral = apply_bg_nan(CSM_Neutral, bg_mask)
            axs[riga, 4].imshow(face)
            axs[riga, 4].imshow(CSM_Neutral, alpha=al, cmap=cmap, vmin=0, vmax=lim)
            np.save(os.path.join(f'{folder_heatmaps}/{emotion}_Neutral.npy'), CSM_Neutral)

        if tot_Sadness == 0 and emotion != 'SAD':
            axs[riga, 5].imshow(face)
        elif emotion != 'SAD':
            CSM_Sadness = CSM_Sadness / float(tot_Sadness)
            CSM_Sadness = safe_minmax_norm(CSM_Sadness)
            CSM_Sadness = apply_bg_nan(CSM_Sadness, bg_mask)
            axs[riga, 5].imshow(face)
            axs[riga, 5].imshow(CSM_Sadness, alpha=al, cmap=cmap, vmin=0, vmax=lim)
            np.save(os.path.join(f'{folder_heatmaps}/{emotion}_Sadness.npy'), CSM_Sadness)

        if tot_Surprise == 0 and emotion != 'SURPRISE':
            axs[riga, 6].imshow(face)
        elif emotion != 'SURPRISE':
            CSM_Surprise = CSM_Surprise / float(tot_Surprise)
            CSM_Surprise = safe_minmax_norm(CSM_Surprise)
            CSM_Surprise = apply_bg_nan(CSM_Surprise, bg_mask)
            axs[riga, 6].imshow(face)
            axs[riga, 6].imshow(CSM_Surprise, alpha=al, cmap=cmap, vmin=0, vmax=lim)
            np.save(os.path.join(f'{folder_heatmaps}/{emotion}_Surprise.npy'), CSM_Surprise)

        riga += 1
        colonna += 1

    plt.savefig(os.path.join('OUTPUTS', 'CSM_ALL.png'))


# =========================
# RUN
# =========================
output_path = "Results"

names_all = [
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

main(output_path, names_all)
print("completed")