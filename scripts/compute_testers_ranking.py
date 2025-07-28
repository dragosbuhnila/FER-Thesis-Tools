import os
import pickle

base_folder = "./saliency_maps/human_Results"

def extract_precision(report_path):
    with open(report_path, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("Precisione:"):
                try:
                    return float(line.strip().split(":")[1])
                except Exception:
                    raise ValueError(f"Could not parse precision in {report_path}")
    raise ValueError(f"Precisione field not found in {report_path}")

if __name__ == "__main__":
    testers = [folder_name for folder_name in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder_name))]
    precisions = []
    for tester in testers:
        report_path = os.path.join(base_folder, tester, "report.txt")
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"report.txt not found for tester {tester}")
        precision = extract_precision(report_path)
        precisions.append((tester, precision))

    # Sort by precision descending
    precisions.sort(key=lambda x: x[1], reverse=True)

    print("Tester ranking by precision:")
    for rank, entry in enumerate(precisions, 1):
        print(f"{rank:2d}. {entry[0]:10s}  Precisione: {entry[1]:.4f}")

    # Save the ranking to a file
    ranking_file = os.path.join(base_folder, "testers_ranking.pkl")
    with open(ranking_file, "wb") as f:
        pickle.dump(precisions, f)