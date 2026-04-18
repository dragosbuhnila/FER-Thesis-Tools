#!/usr/bin/env python3
"""
Refactor new results (raw) into a single HEATMAPS_humans_phase2_subsets
superfolder with per-subset OLD-format structure.

Usage: run from repository root; script imports constants from
modules.saliencies_folders:
- BASE_DIR
- HEATMAPS_ALE_SUBSETS_DIR_BASENAME   -> destination basename
- HEATMAPS_ALE_SUBSETS_RAW_DIR_BASENAME -> raw input basename
"""

import os
import sys
import shutil
import hashlib
import csv
import logging
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # to import from modules

from modules.saliencies_folders import (
    BASE_DIR,
    HEATMAPS_ALE_SUBSETS_DIR_BASENAME,
    HEATMAPS_ALE_SUBSETS_RAW_DIR_BASENAME,
)

# --- Configuration ---
EMOTIONS = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]
GT_MAP = {
    "ANGRY": "Angry",
    "DISGUST": "Disgust",
    "FEAR": "Fear",
    "HAPPY": "Happiness",
    "NEUTRAL": "Neutral",
    "SAD": "Sad",
    "SURPRISE": "Surprise",
}
EXPECTED_OCCLUSION_GROUPS = [f"positive_{e}" for e in EMOTIONS] + [f"negative_{e}" for e in EMOTIONS]
SUBSETS = EXPECTED_OCCLUSION_GROUPS + ["match", "mismatch"]

# File names for logs inside destination root
MAPPING_CSV = "transfer_mapping.csv"
CONFLICTS_CSV = "conflicts.csv"
UNEXPECTED_GROUPS_LOG = "unexpected_groups.txt"
SUMMARY_TXT = "summary.txt"

# --- Helpers ---
def sha256_of_file(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)

def normalize_target_name(pred: str, gt: str) -> str:
    if pred.upper() == gt.upper():
        return f"{pred}_canonical.npy"
    gt_up = gt.upper()
    mapped = GT_MAP.get(gt_up)
    if not mapped:
        mapped = gt.capitalize()
    return f"{pred}_{mapped}.npy"

def parse_pred_gt_from_basename(basename: str):
    if "_" not in basename:
        return basename, basename
    pred, rest = basename.split("_", 1)
    return pred, rest

# --- Main flow ---
def main(dry_run: bool = False):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    base = BASE_DIR  # string
    raw_root = os.path.join(base, HEATMAPS_ALE_SUBSETS_RAW_DIR_BASENAME)
    dest_root = os.path.join(base, HEATMAPS_ALE_SUBSETS_DIR_BASENAME)

    logging.info("Preparing destination subset folders")
    for subset in SUBSETS:
        ensure_dirs(os.path.join(dest_root, subset, "Results"))

    mapping_path = os.path.join(dest_root, MAPPING_CSV)
    conflicts_path = os.path.join(dest_root, CONFLICTS_CSV)
    unexpected_groups_path = os.path.join(dest_root, UNEXPECTED_GROUPS_LOG)
    summary_path = os.path.join(dest_root, SUMMARY_TXT)

    mapping_rows = []
    conflicts_rows = []
    unexpected_groups = set()

    total_copied = 0
    total_skipped_conflict = 0
    total_skipped_nonpy = 0
    total_unexpected_groups = 0

    # Process CSM occlusion groups
    csm_dir = os.path.join(raw_root, "CSM_12_GROUPS_OCCLUDED")
    if os.path.exists(csm_dir):
        logging.info("Processing CSM occlusion groups from %s", csm_dir)
        for tester_name in os.listdir(csm_dir):
            tester_dir = os.path.join(csm_dir, tester_name)
            if not os.path.isdir(tester_dir):
                continue
            occl_groups_root = os.path.join(tester_dir, "occlusion_groups")
            if not os.path.exists(occl_groups_root):
                logging.warning("No occlusion_groups subfolder for tester %s", tester_name)
                continue
            for group_name in os.listdir(occl_groups_root):
                group_dir = os.path.join(occl_groups_root, group_name)
                if not os.path.isdir(group_dir):
                    continue
                if group_name not in EXPECTED_OCCLUSION_GROUPS:
                    unexpected_groups.add(group_dir)
                    total_unexpected_groups += 1
                    continue
                cells_dir = os.path.join(group_dir, "cells")
                if not os.path.exists(cells_dir):
                    logging.warning("No cells/ subfolder in %s for tester %s", group_dir, tester_name)
                    continue
                for fname in os.listdir(cells_dir):
                    src = os.path.join(cells_dir, fname)
                    if not os.path.isfile(src):
                        continue
                    if os.path.splitext(fname)[1].lower() != ".npy":
                        total_skipped_nonpy += 1
                        continue
                    basename = os.path.splitext(fname)[0]
                    pred, gt = parse_pred_gt_from_basename(basename)
                    target_name = normalize_target_name(pred, gt)
                    target_dir = os.path.join(dest_root, group_name, "Results", tester_name, "heatmaps")
                    ensure_dirs(target_dir)
                    target_path = os.path.join(target_dir, target_name)

                    if os.path.exists(target_path):
                        src_hash = sha256_of_file(src)
                        tgt_hash = sha256_of_file(target_path)
                        if src_hash == tgt_hash:
                            mapping_rows.append([src, target_path, "identical_skip"])
                        else:
                            conflicts_rows.append([src, target_path, "conflict_skip"])
                            total_skipped_conflict += 1
                        continue

                    mapping_rows.append([src, target_path, "copy"])
                    if not dry_run:
                        shutil.copy2(src, target_path)
                    total_copied += 1
    else:
        logging.info("CSM_12_GROUPS_OCCLUDED not found under raw root %s", raw_root)

    # Process M-MM (match / mismatch)
    mmm_dir = os.path.join(raw_root, "M-MM")
    if os.path.exists(mmm_dir):
        logging.info("Processing M-MM match/mismatch from %s", mmm_dir)
        for tester_name in os.listdir(mmm_dir):
            tester_dir = os.path.join(mmm_dir, tester_name)
            if not os.path.isdir(tester_dir):
                continue
            mm_root = os.path.join(tester_dir, "match_mismatch")
            if not os.path.exists(mm_root):
                logging.warning("No match_mismatch subfolder for tester %s", tester_name)
                continue
            for sub in ("heatmaps_match", "heatmaps_mismatch"):
                subset_name = "match" if ("match" in sub and "mismatch" not in sub) else "mismatch"
                folder = os.path.join(mm_root, sub)
                if not os.path.exists(folder):
                    continue
                for fname in os.listdir(folder):
                    src = os.path.join(folder, fname)
                    if not os.path.isfile(src):
                        continue
                    if os.path.splitext(fname)[1].lower() != ".npy":
                        total_skipped_nonpy += 1
                        continue
                    name = os.path.splitext(fname)[0]
                    name = re.sub(r"_(match|mismatch)$", "", name, flags=re.IGNORECASE)
                    pred, gt = parse_pred_gt_from_basename(name)
                    target_name = normalize_target_name(pred, gt)
                    target_dir = os.path.join(dest_root, subset_name, "Results", tester_name, "heatmaps")
                    ensure_dirs(target_dir)
                    target_path = os.path.join(target_dir, target_name)

                    if os.path.exists(target_path):
                        src_hash = sha256_of_file(src)
                        tgt_hash = sha256_of_file(target_path)
                        if src_hash == tgt_hash:
                            mapping_rows.append([src, target_path, "identical_skip"])
                        else:
                            conflicts_rows.append([src, target_path, "conflict_skip"])
                            total_skipped_conflict += 1
                        continue

                    mapping_rows.append([src, target_path, "copy"])
                    if not dry_run:
                        shutil.copy2(src, target_path)
                    total_copied += 1
    else:
        logging.info("M-MM not found under raw root %s", raw_root)

    ensure_dirs(dest_root)
    if mapping_rows:
        with open(mapping_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source", "target", "action"])
            writer.writerows(mapping_rows)

    if conflicts_rows:
        with open(conflicts_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source", "target", "note"])
            writer.writerows(conflicts_rows)

    if unexpected_groups:
        with open(unexpected_groups_path, "w", encoding="utf-8") as fh:
            for g in sorted(unexpected_groups):
                fh.write(g + "\n")

    # Validation checks
    logging.info("Running validation checks")
    validation_problems = []
    for subset in SUBSETS:
        subset_path = os.path.join(dest_root, subset)
        if not os.path.exists(subset_path):
            continue
        for root, dirs, files in os.walk(subset_path):
            for f in files:
                if not f.lower().endswith(".npy"):
                    continue
                p = os.path.join(root, f)
                rel = os.path.relpath(p, subset_path)
                parts = rel.split(os.sep)
                if len(parts) < 4 or parts[0] != "Results" or parts[2] != "heatmaps":
                    validation_problems.append(p)

    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"copied:{total_copied}\n")
        fh.write(f"skipped_conflicts:{total_skipped_conflict}\n")
        fh.write(f"skipped_nonpy:{total_skipped_nonpy}\n")
        fh.write(f"unexpected_groups:{total_unexpected_groups}\n")
        fh.write(f"validation_issues:{len(validation_problems)}\n")
        if validation_problems:
            fh.write("\n--- validation problem files ---\n")
            for p in validation_problems:
                fh.write(p + "\n")

    logging.info("Done. copied=%d, conflicts=%d, non-py skipped=%d, unexpected groups=%d",
                 total_copied, total_skipped_conflict, total_skipped_nonpy, total_unexpected_groups)
    logging.info("Mapping CSV: %s", mapping_path)
    logging.info("Conflicts CSV: %s", conflicts_path if conflicts_rows else "(none)")
    logging.info("Unexpected groups log: %s", unexpected_groups_path if unexpected_groups else "(none)")
    logging.info("Summary: %s", summary_path)

if __name__ == "__main__":
    main(dry_run=False)