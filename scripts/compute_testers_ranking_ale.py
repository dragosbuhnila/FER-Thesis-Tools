import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

from modules.saliencies_folders import BASE_DIR, HEATMAPS_ALE_DIR_BASENAME

HUMAN_RESULTS_BASE_FOLDER_PATH = os.path.join(BASE_DIR, HEATMAPS_ALE_DIR_BASENAME)

def extract_precision(report_path):
    with open(report_path, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("Precisione:"):
                try:
                    return float(line.strip().split(":")[1])
                except Exception:
                    raise ValueError(f"Could not parse precision in {report_path}")
    raise ValueError(f"Precisione field not found in {report_path}")

TESTER_TO_ACCURACY_DICT = {
    "LUCRUG": 0.233618,
    "KRIJAK": 0.467236,
    "FILCOM": 0.495726,
    "LORLAF": 0.524217,
    "SAMPIG": 0.524217,
    "FILVEL": 0.527066,
    "ALELUC": 0.527066,
    "LORSCA": 0.532764,
    "STEALT": 0.532764,
    "STEPAL": 0.544160,
    "CARPAR": 0.555556,
    "CHISPA": 0.558405,
    "FRAMUC": 0.558405,
    "VIRTOM": 0.564103,
    "SILCAF": 0.569801,
    "ANDESP": 0.572650,
    "EMAFOS": 0.572650,
    "MATOCC": 0.581197,
    "NICRIB": 0.584046,
    "DOMTEL": 0.586895,
    "GIATRI": 0.586895,
    "MARRIC": 0.589744,
    "VALTAR": 0.592593,
    "SARGAR": 0.595442,
    "VITMER": 0.598291,
    "AMIEFT": 0.601140,
    "DRABUH": 0.603989,
    "EMAGRE": 0.603989,
    "ARMLAR": 0.609687,
    "GIOBRU": 0.609687,
    "ROCOLI": 0.618234,
    "ELETOR": 0.618234,
    "MARIAM": 0.621083,
    "SIMMAF": 0.621083,
    "VALRUO": 0.626781,
    "ROCLAB": 0.632479,
    "FABIAC": 0.635328,
    "ALEGRA": 0.641026,
    "ELEOLI": 0.652422,
    "MARFRO": 0.655271,
    "REBDEL": 0.655271,
    "GREBAZ": 0.658120,
    "FILGUA": 0.660969,
    "ALITOM": 0.672365,
    "MARCEC": 0.672365,
    "MARDIB": 0.678063,
    "SILVEN": 0.680912,
    "FEDMAR": 0.680912,
    "AMAESC": 0.683761,
    "REBLEO": 0.689459,
}


if __name__ == "__main__":
    accuracies = list(TESTER_TO_ACCURACY_DICT.items())
    accuracies.sort(key=lambda x: x[1], reverse=True)

    
    print("Tester ranking by accuracy:")
    for rank, entry in enumerate(accuracies, 1):
        print(f"{rank:2d}. {entry[0]:10s}  Accuracy: {entry[1]:.4f}")

    # Save the ranking to a file
    ranking_file = os.path.join(HUMAN_RESULTS_BASE_FOLDER_PATH, "testers_ranking.pkl")
    with open(ranking_file, "wb") as f:
        pickle.dump(accuracies, f)


# RESULT WILL BE LIKE:
# Item 1/1:
# list (len=54) {
#     [0]: tuple (len=2) {
#         [0]: (len=6)'FEDMAR'
#         [1]: 0.7806267806267806
#     }
#     [1]: tuple (len=2) {
#         [0]: (len=6)'MARFRO'
#         [1]: 0.7777777777777778
#     }
#     [2]: tuple (len=2) {
#         [0]: (len=6)'ANDMAM'
#         [1]: 0.7720797720797721
#     }
#     [3]: tuple (len=2) {
#         [0]: (len=6)'AURFEA'
#         [1]: 0.7720797720797721
#     }
#     [4]: tuple (len=2) {
#         [0]: (len=6)'FRAANT'
#         [1]: 0.7692307692307693
#     }
#     [5]: tuple (len=2) {
#         [0]: (len=6)'ALECEL'
#         [1]: 0.7663817663817664
#     }
#     [6]: tuple (len=2) {
#         [0]: (len=6)'ANDDAL'
#         [1]: 0.7635327635327636
#     }
#     [7]: tuple (len=2) {
#         [0]: (len=6)'FEDAMA'
#         [1]: 0.7606837606837606
#     }
#     [8]: tuple (len=2) {
#         [0]: (len=6)'PAORUS'
#         [1]: 0.7606837606837606
#     }
#     [9]: tuple (len=2) {
#         [0]: (len=6)'DALGON'
#         [1]: 0.7578347578347578
#     }
#     [10]: tuple (len=2) {
#         [0]: (len=6)'FRAGHI'
#         [1]: 0.7521367521367521
#     }
#     [11]: tuple (len=2) {
#         [0]: (len=6)'BEAFRA'
#         [1]: 0.7464387464387464
#     }
#     [12]: tuple (len=2) {
#         [0]: (len=6)'SILFER'
#         [1]: 0.7464387464387464
#     }
#     [13]: tuple (len=2) {
#         [0]: (len=6)'GIUGUI'
#         [1]: 0.7435897435897436
#     }
#     [14]: tuple (len=2) {
#         [0]: (len=6)'MIRTEK'
#         [1]: 0.7435897435897436
#     }
#     ... (24 more items) ...
#     [39]: tuple (len=2) {
#         [0]: (len=6)'SARCLA'
#         [1]: 0.6866096866096866
#     }
#     [40]: tuple (len=2) {
#         [0]: (len=6)'CARGUL'
#         [1]: 0.6837606837606838
#     }
#     [41]: tuple (len=2) {
#         [0]: (len=6)'FABCAM'
#         [1]: 0.6837606837606838
#     }
#     [42]: tuple (len=2) {
#         [0]: (len=6)'FATGJI'
#         [1]: 0.6837606837606838
#     }
#     [43]: tuple (len=2) {
#         [0]: (len=6)'FILROG'
#         [1]: 0.6809116809116809
#     }
#     [44]: tuple (len=2) {
#         [0]: (len=6)'MADFES'
#         [1]: 0.6752136752136753
#     }
#     [45]: tuple (len=2) {
#         [0]: (len=6)'MAUMOR'
#         [1]: 0.6752136752136753
#     }
#     [46]: tuple (len=2) {
#         [0]: (len=6)'ANTALT'
#         [1]: 0.6723646723646723
#     }
#     [47]: tuple (len=2) {
#         [0]: (len=6)'AGAPIG'
#         [1]: 0.6666666666666666
#     }
#     [48]: tuple (len=2) {
#         [0]: (len=6)'MARCAM'
#         [1]: 0.6666666666666666
#     }
#     [49]: tuple (len=2) {
#         [0]: (len=6)'RICROG'
#         [1]: 0.6666666666666666
#     }
#     [50]: tuple (len=2) {
#         [0]: (len=6)'ROBGOT'
#         [1]: 0.6552706552706553
#     }
#     [51]: tuple (len=2) {
#         [0]: (len=6)'MATDIG'
#         [1]: 0.6182336182336182
#     }
#     [52]: tuple (len=2) {
#         [0]: (len=6)'ANTCAN'
#         [1]: 0.6125356125356125
#     }
#     [53]: tuple (len=2) {
#         [0]: (len=6)'MATVIN'
#         [1]: 0.5470085470085471
#     }
# }