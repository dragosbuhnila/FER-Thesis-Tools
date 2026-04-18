#!/usr/bin/env python3
"""
Reorganize new machine XAI results into per-subset legacy layout.

Target (per-subset):
  saliency_maps/HEATMAPS_machines_phase2_subsets/<subset>/{Bubbles,EXTERNAL,GRADCAM}/occft_<model>/*.npy
"""
from __future__ import annotations
import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import shutil
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.filenames_utils import EMOTIONS, CANONICAL_SUBSETS
import modules.saliencies_folders as sf  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reorg")

# per-model gradcam layer to pick
GRADCAM_LAYER_MAP = {
    "convnext": "convnext_base_stage_2_block_24_depthwise_conv",
    "efficientnetb1": "block4c_project_conv",
    "inceptionv3": "conv2d_30",
    "pattlite": "conv_pw_8",
    "resnet": "conv4_block5_3_conv",
    "vgg19": "block4_conv4",
    "yolo": "module_8",
}

MODEL_PATTERNS = {
    "convnext": r"convnext",
    "efficientnetb1": r"efficientnetb1|efficientnet",
    "inceptionv3": r"inceptionv3|inception",
    "pattlite": r"pattlite",
    "resnet": r"resnet",
    "vgg19": r"vgg19|vgg",
    "yolo": r"occft_yolo",
}


def detect_model_key_from_name(name: str) -> str | None:
    ln = name.lower()
    for key, pat in MODEL_PATTERNS.items():
        if re.search(pat, ln):
            return key
    return None


def parse_subset_token(parts: tuple[str, ...]) -> str:
    # prefer canonical subset names if found in parts
    for p in parts:
        if p in CANONICAL_SUBSETS:
            return p
        low = p.lower()
        for cs in CANONICAL_SUBSETS:
            if cs.lower() == low:
                return cs
    # timestamp or match/mismatch/positive/negative detection
    for p in parts:
        if re.match(r"\d{8}-\d{6}", p):
            # keep raw timestamp folder as token (will be normalized later if mapping available)
            return p
        pl = p.lower()
        if any(k in pl for k in ("match", "mismatch", "positive", "negative")):
            # sanitize to canonical-like
            return pl
    return "unknown_subset"


def normalize_subset_name(token: str) -> str:
    # try to return exact canonical name if token maps to one; otherwise return token as-is
    low = token.lower()
    mapping = {k.lower(): k for k in CANONICAL_SUBSETS}
    if low in mapping:
        return mapping[low]
    # simple heuristics:
    if "match" in low and "mismatch" not in low:
        return "match"
    if "mismatch" in low:
        return "mismatch"
    m = re.search(r"(positive|negative)[-_ ]*([a-z]+)", low)
    if m:
        sign = m.group(1)
        emo = m.group(2).upper()
        return f"{sign}_{emo}"
    return token


def copy_tree_safe(src: Path, dst: Path, dry_run: bool):
    if not src.exists():
        return
    if dry_run:
        logger.info(f"DRY copy tree: {src} -> {dst}")
        return
    if dst.exists():
        logger.info(f"Destination exists, skipping tree copy: {dst}")
        return
    shutil.copytree(src, dst)


def copy_files(src_files: list[Path], dst_dir: Path, dry_run: bool):
    if not src_files:
        return
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_files:
        dst = dst_dir / f.name
        logger.info(f"    Copy: {f} -> {dst}")
        if not dry_run:
            shutil.copy2(f, dst)


def reorganize(raw_base_dir: Path, target_base_dir: Path, dry_run: bool = True):
    logger.info(f"Raw base: {raw_base_dir}")
    logger.info(f"Target base: {target_base_dir}")
    if not raw_base_dir.exists():
        raise FileNotFoundError(f"raw base dir not found: {raw_base_dir}")

    # walk model-level folders under raw tree and map each to a subset + method
    for root, dirs, files in os.walk(raw_base_dir):
        root_path = Path(root)
        name = root_path.name

        # skip folders that are clearly not model containers
        if not (name.lower().startswith("occft_") or any(p in name.lower() for p in ("convnext", "efficientnet", "inception", "pattlite", "resnet", "vgg", "yolo"))):
            continue

        # relative parts helps identify method and subset
        try:
            rel_parts = root_path.relative_to(raw_base_dir).parts
        except Exception:
            rel_parts = tuple(root_path.parts)

        # detect method by scanning rel parts
        method = None
        for p in rel_parts:
            pl = p.lower()
            if "bubble" in pl:
                method = "Bubbles"; break
            if "ext" in pl or "external" in pl or "extpert" in pl:
                method = "EXTERNAL"; break
            if "grad" in pl or "gradcam" in pl:
                method = "GRADCAM"; break

        subset_token = parse_subset_token(rel_parts)
        subset_name = normalize_subset_name(subset_token)

        # ensure subset_name is one of canonical names if possible
        if subset_name not in CANONICAL_SUBSETS and subset_name not in ("match", "mismatch"):
            # allow it but log
            logger.info(f"Using non-canonical subset token: {subset_name}")

        model_key = detect_model_key_from_name(name) or detect_model_key_from_name(root_path.parent.name)
        model_folder_name = f"occft_{model_key}" if model_key else "occft_unknown"
        dst_subset_base = target_base_dir / subset_name

        logger.info(f"Found model container: {root_path} -> method={method} subset={subset_name} model={model_folder_name}")

        # ------- Bubbles -------
        if method == "Bubbles" or (method is None and "bubble" in root_path.as_posix().lower()):
            # prefer HEATMAPS subfolder; else emotion subfolders or direct npy
            npy_files = []
            hp = root_path / "HEATMAPS"
            if hp.exists():
                npy_files = [p for p in hp.iterdir() if p.is_file() and p.suffix.lower() == ".npy"]
            else:
                for child in root_path.iterdir():
                    if child.is_file() and child.suffix.lower() == ".npy":
                        npy_files.append(child)
                    elif child.is_dir() and child.name.upper() in [e.upper() for e in EMOTIONS]:
                        npy_files.extend([p for p in child.iterdir() if p.is_file() and p.suffix.lower() == ".npy"])
            dst_dir = dst_subset_base / "Bubbles" / model_folder_name
            copy_files(npy_files, dst_dir, dry_run)
            src_stats = root_path / "stats_cache"
            if src_stats.exists():
                copy_tree_safe(src_stats, dst_dir / "stats_cache", dry_run)
            continue

        # ------- EXTERNAL -------
        if method == "EXTERNAL" or (method is None and any(k in root_path.as_posix().lower() for k in ("ext", "external", "extpert"))):
            hp = root_path / "HEATMAPS"
            npy_files = []
            if hp.exists():
                npy_files = [p for p in hp.iterdir() if p.is_file() and p.suffix.lower() == ".npy"]
            else:
                for child in root_path.iterdir():
                    if child.is_dir() and child.name.upper() in [e.upper() for e in EMOTIONS]:
                        npy_files.extend([p for p in child.iterdir() if p.is_file() and p.suffix.lower() == ".npy"])
            dst_dir = dst_subset_base / "EXTERNAL" / model_folder_name
            copy_files(npy_files, dst_dir, dry_run)
            src_stats = root_path / "stats_cache"
            if src_stats.exists():
                copy_tree_safe(src_stats, dst_dir / "stats_cache", dry_run)
            continue

        # ------- GRADCAM -------
        if method == "GRADCAM" or (method is None and "grad" in root_path.as_posix().lower()):
            # climb to model dir if root_path is deeper (e.g., layer folder)
            model_dir = root_path
            while model_dir != raw_base_dir and not (model_dir.name.lower().startswith("occft_") or detect_model_key_from_name(model_dir.name)):
                model_dir = model_dir.parent
            mk = detect_model_key_from_name(model_dir.name) or model_key
            if not mk:
                logger.info(f"  Could not determine model key for gradcam at {root_path}; skipping")
                continue
            desired_layer = GRADCAM_LAYER_MAP.get(mk)
            if not desired_layer:
                logger.info(f"  No gradcam layer configured for {mk}; skipping")
                continue

            # find the desired layer folder under model_dir
            candidate_layer_dir = None
            direct = model_dir / desired_layer
            if direct.exists() and direct.is_dir():
                candidate_layer_dir = direct
            else:
                for child in model_dir.iterdir():
                    if child.is_dir() and desired_layer.lower() in child.name.lower():
                        candidate_layer_dir = child
                        break
                if candidate_layer_dir is None:
                    # fallback: pick last layer-like dir
                    layer_candidates = [c for c in model_dir.iterdir() if c.is_dir() and re.search(r"block|stage|module_|conv|depthwise|layer", c.name.lower())]
                    if layer_candidates:
                        candidate_layer_dir = sorted(layer_candidates)[-1]

            if candidate_layer_dir is None:
                logger.info(f"  Layer folder for {mk} not found; skipping")
                continue

            # collect npy files from layer's HEATMAPS or emotion subfolders
            heatmaps_folder = candidate_layer_dir / "HEATMAPS"
            npy_files = []
            if heatmaps_folder.exists():
                npy_files = [p for p in heatmaps_folder.iterdir() if p.is_file() and p.suffix.lower() == ".npy"]
            else:
                for child in candidate_layer_dir.iterdir():
                    if child.is_dir() and child.name.upper() in [e.upper() for e in EMOTIONS]:
                        npy_files.extend([p for p in child.iterdir() if p.is_file() and p.suffix.lower() == ".npy"])

            # DESTINATION: put .npy directly under GRADCAM/occft_<model> (no layer subfolder)
            dst_dir = dst_subset_base / "GRADCAM" / model_folder_name
            copy_files(npy_files, dst_dir, dry_run)

            # copy stats_cache (prefer layer stats_cache, fallback to model stats_cache) into the model dst folder
            src_stats_layer = candidate_layer_dir / "stats_cache"
            if src_stats_layer.exists():
                copy_tree_safe(src_stats_layer, dst_dir / "stats_cache", dry_run)
            else:
                src_stats_model = model_dir / "stats_cache"
                if src_stats_model.exists():
                    copy_tree_safe(src_stats_model, dst_dir / "stats_cache", dry_run)
            continue

        # fallback: if HEATMAPS present, treat as EXTERNAL
        hp = root_path / "HEATMAPS"
        if hp.exists():
            npy_files = [p for p in hp.iterdir() if p.is_file() and p.suffix.lower() == ".npy"]
            dst_dir = dst_subset_base / "EXTERNAL" / model_folder_name
            copy_files(npy_files, dst_dir, dry_run)

    logger.info("Done.")

def _map_timestamped_to_subset(name: str) -> str | None:
    s = name.lower()
    if "matching" in s or ("match" in s and "mismatch" not in s):
        return "match"
    if "mismatching" in s or "mismatch" in s:
        return "mismatch"
    m = re.search(r"(positive|negative)[-_ ]*(angry|disgust|fear|happy|neutral|sad|surprise)", s)
    if m:
        sign, emo = m.group(1), m.group(2).upper()
        return f"{sign}_{emo}"
    # timestamp-only folder (no explicit token) -> None
    return None

def _merge_dir_contents(src: Path, dst: Path, dry_run: bool):
    if not src.exists():
        return
    if dry_run:
        logger.info(f"DRY merge: {src} -> {dst}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                _merge_dir_contents(item, target, dry_run)
                try:
                    item.rmdir()
                except OSError:
                    pass
            else:
                shutil.move(str(item), str(target))
        else:
            # file: overwrite if present
            if target.exists():
                shutil.copy2(str(item), str(target))
                item.unlink()
            else:
                shutil.move(str(item), str(target))
    # remove source if empty
    try:
        src.rmdir()
    except OSError:
        pass

# def normalize_subset_folders(target_base: Path, dry_run: bool):
#     wrong_subsets = os.listdir(target_base)
#     for wrong_subset in wrong_subsets:
#         wrong_subset_path = target_base / wrong_subset
#         if not wrong_subset_path.is_dir():
#             logging.info(f"Skipping non-directory item in target base: {wrong_subset_path}")
#             continue

#         keyword_to_subset_name = {
#             "occluded-matching": "match",
#             "occluded-mismatching": "mismatch",
#             "positive-angry": "positive_ANGRY",
#             "positive-disgust": "positive_DISGUST",
#             "positive-fear": "positive_FEAR",
#             "positive-happy": "positive_HAPPY",
#             "positive-neutral": "positive_NEUTRAL",
#             "positive-sad": "positive_SAD",
#             "positive-surprise": "positive_SURPRISE",
#             "negative-angry": "negative_ANGRY",
#             "negative-disgust": "negative_DISGUST",
#             "negative-fear": "negative_FEAR",
#             "negative-happy": "negative_HAPPY",
#             "negative-neutral": "negative_NEUTRAL",
#             "negative-sad": "negative_SAD",
#             "negative-surprise": "negative_SURPRISE"
#         }

#         # rename the wrong_subset_path folders to having the correct canonical subset name
#         mapped_subset_name = None
#         for keyword, subset_name in keyword_to_subset_name.items():
#             if keyword in wrong_subset.lower():
#                 mapped_subset_name = subset_name
#                 break

#         if mapped_subset_name is not None:
#             # Rename the folder
#             new_subset_path = target_base / mapped_subset_name
#             if not new_subset_path.exists():
#                 wrong_subset_path.rename(new_subset_path)
#             else:
#                 logger.warning(f"Subset {mapped_subset_name} already exists, skipping {wrong_subset}")

# & "C:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/.venv/Scripts/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/scripts/regorganize_new_machines_confronti_to_standard.py"
def main():
    parser = argparse.ArgumentParser(description="Reorganize new machine XAI heatmaps into per-subset legacy layout")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually copy files; just print actions")
    parser.add_argument("--raw", type=str, default=None, help="Raw base dir override (optional)")
    parser.add_argument("--target", type=str, default=None, help="Target subsets base dir override (optional)")
    args = parser.parse_args()

    raw_base = Path(args.raw) if args.raw else Path(sf.BASE_DIR) / sf.HEATMAPS_OCCFT_SUBSETS_RAW_DIR_BASENAME
    target_base = Path(args.target) if args.target else Path(sf.BASE_DIR) / sf.HEATMAPS_OCCFT_SUBSETS_DIR_BASENAME

    reorganize(raw_base, target_base, dry_run=args.dry_run)


if __name__ == "__main__":
    main()