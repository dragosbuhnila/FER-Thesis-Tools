import os
import sys
import pickle
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.saliencies_folders import RANKING_PHASE2_FILE_PATH


def try_int(value: Any):
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return int(str(value).strip())
    except Exception:
        try:
            return int(float(str(value).strip()))
        except Exception:
            return None


def parse_lines_to_ranking(lines):
    ranking = {}
    for line in lines:
        if not isinstance(line, str):
            line = str(line)
        parts = line.strip().split(",")
        if len(parts) >= 2:
            subject = parts[0].strip()
            rank = try_int(parts[1].strip())
            if rank is not None:
                ranking[subject] = rank
    return ranking


def normalize_to_ranking(data):
    ranking = {}
    # dict-like
    if isinstance(data, dict):
        for k, v in data.items():
            subject = str(k)
            rank = try_int(v)
            if rank is not None:
                ranking[subject] = rank
        return ranking

    # sequence of pairs (list/tuple)
    if isinstance(data, (list, tuple)):
        # list of (subject, rank)
        if all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in data):
            for item in data:
                subject = str(item[0])
                rank = try_int(item[1])
                if rank is not None:
                    ranking[subject] = rank
            return ranking
        # list of strings or other items -> try parsing as lines
        return parse_lines_to_ranking([str(x) for x in data])

    # fallback: try to stringify and parse as lines
    return parse_lines_to_ranking([str(data)])


def load_ranking_file(path):
    # Try pickle first, then fallback to reading as text
    try:
        with open(path, "rb") as f:
            try:
                data = pickle.load(f)
                return normalize_to_ranking(data)
            except Exception:
                f.seek(0)
                raw = f.read()
                try:
                    text = raw.decode("utf-8")
                except Exception:
                    text = raw.decode("latin-1", errors="replace")
                return parse_lines_to_ranking(text.splitlines())
    except FileNotFoundError:
        raise
    except Exception as e:
        # As a final fallback, try opening as text
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return parse_lines_to_ranking(f.readlines())
        except Exception:
            raise e


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RANKING_PHASE2_FILE_PATH
    try:
        ranking = load_ranking_file(path)
    except FileNotFoundError:
        print(f"Ranking file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load ranking file '{path}': {e}", file=sys.stderr)
        sys.exit(1)

    if not ranking:
        print("No valid ranking entries found.", file=sys.stderr)
        sys.exit(1)

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1])
    mapping = {subject: f"subject_{i+1}" for i, (subject, _) in enumerate(sorted_ranking)}

    for subject, anonymized_name in mapping.items():
        print(f"{subject}: {anonymized_name}")