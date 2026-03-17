import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

from modules.saliencies_folders import BASE_DIR, HEATMAPS_ALE_DIR_BASENAME

HUMAN_RESULTS_BASE_FOLDER_PATH = os.path.join(BASE_DIR, HEATMAPS_ALE_DIR_BASENAME)
OUTPUT_FILE_PATH = os.path.join(HUMAN_RESULTS_BASE_FOLDER_PATH, "testers_gender.txt")

TESTER_TO_GENDER_DICT = {
    "ALEGRA": "Maschio",
    "ALELUC": "Maschio",
    "AMIEFT": "Maschio",
    "ANDESP": "Maschio",
    "ARMLAR": "Maschio",
    "CARPAR": "Maschio",
    "DOMTEL": "Maschio",
    "DRABUH": "Maschio",
    "EMAFOS": "Maschio",
    "FABIAC": "Maschio",
    "FILCOM": "Maschio",
    "FILGUA": "Maschio",
    "FILVEL": "Maschio",
    "KRIJAK": "Maschio",
    "LORSCA": "Maschio",
    "LUCRUG": "Maschio",
    "MARDIB": "Maschio",
    "MATOCC": "Maschio",
    "NICRIB": "Maschio",
    "ROCLAB": "Maschio",
    "ROCOLI": "Maschio",
    "SAMPIG": "Maschio",
    "SIMMAF": "Maschio",
    "STEALT": "Maschio",
    "STEPAL": "Maschio",
    "VALTAR": "Maschio",

    "ALITOM": "Femmina",
    "AMAESC": "Femmina",
    "CHISPA": "Femmina",
    "ELEOLI": "Femmina",
    "ELETOR": "Femmina",
    "EMAGRE": "Femmina",
    "FEDMAR": "Femmina",
    "FRAMUC": "Femmina",
    "GIATRI": "Femmina",
    "GIOBRU": "Femmina",
    "GREBAZ": "Femmina",
    "LORLAF": "Femmina",
    "MARCEC": "Femmina",
    "MARFRO": "Femmina",
    "MARIAM": "Femmina",
    "MARRIC": "Femmina",
    "REBDEL": "Femmina",
    "REBLEO": "Femmina",
    "SARGAR": "Femmina",
    "SILCAF": "Femmina",
    "SILVEN": "Femmina",
    "VALRUO": "Femmina",
    "VIRTOM": "Femmina",
    "VITMER": "Femmina",
}


if __name__ == "__main__":
    with open(OUTPUT_FILE_PATH, "w") as f:
        for tester, gender in TESTER_TO_GENDER_DICT.items():
            f.write(f"{tester}\t{gender}\n")

# RESULT WILL BE LIKE:
# AGAPIG	Femmina
# ALBMAD	Maschio
# ALECEL	Femmina
# ANDCUG	Maschio
# ANDDAL	Maschio
# ANDING	Maschio
# ANDMAM	Maschio
# ANTALT	Maschio
# ANTBIL	Femmina
# ANTCAN	Maschio
# ARAMAG	Femmina
# AURFEA	Femmina
# AZRCUR	Femmina
# BEAFRA	Femmina
# BENPER	Femmina
# CARGUL	Maschio
# CHIDER	Femmina
# CHIVAN	Femmina
# COSGAG	Femmina
# DALGON	Femmina
# DEMVEN	Femmina
# EDODON	Maschio
# EDOPEL	Maschio
# ELEMAR	Femmina
# ELEOLI	Femmina