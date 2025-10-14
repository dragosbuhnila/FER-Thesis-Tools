import itertools
import os
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.patches import Rectangle
import numpy as np
import cv2
import pandas as pd

from modules.basefaces import get_base_face, basefaces
from modules.filenames_utils import EMOTIONS, EMOTIONS_PRED


def plot_matrix(matrix, title="No Title", cmap='turbo', vmin=None, vmax=None, block=True, background=None, alpha=0.7, save_folder=False, save_only=False):
    """
    Plots a 2D matrix as an image, optionally overlaying it on a background image.
    """
    plt.figure()
    if background is not None:
        plt.imshow(background, cmap='gray', interpolation='nearest')
        plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax, alpha=alpha)
    else:
        plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()

    # Save the plot if save_folder is specified
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        filename = title.split("(")[0].strip(" ")
        save_path = os.path.join(save_folder, f"{filename}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    # Show the plot only if save_only is False
    if not save_only:
        plt.show(block=block)
        
    plt.close()

def print_aubyau_stats(statistics, title):
    """
    Prints the statistics for each AU in a readable format.
    """
    print(f"\nStatistics for {title}:")
    for au, stat in statistics.items():
        print(f"AU: {au},", end=' ')
        for stat_name, stat_value in stat.items():
            print(f"{stat_name}: {stat_value:.4f},", end=' ')
        print()

def show_grid_tkinter(df):
    root = tk.Tk()
    root.title("Pairwise Comparison Grid")

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Add index as the first column
    columns = ["Heatmap"] + list(df.columns)
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")

    for idx, row in df.iterrows():
        tree.insert("", "end", values=[idx] + list(row))

    tree.pack(fill=tk.BOTH, expand=True)
    root.mainloop()

def make_comparison_grid_combinations(names, stats_dict, compare_func):
    """
    Create a symmetric grid DataFrame comparing all pairs in stats_dict using compare_func.
    Returns a DataFrame with formatted string values.
    """
    grid = pd.DataFrame(index=names, columns=names)
    # Fill off-diagonal
    for name1, name2 in itertools.combinations(names, 2):
        result = compare_func(stats_dict[name1], stats_dict[name2])
        val = next(iter(result.values())) if isinstance(result, dict) else result
        grid.loc[name1, name2] = f"{val:.4f}"
        grid.loc[name2, name1] = f"{val:.4f}"
    # Fill diagonal
    for name in names:
        result = compare_func(stats_dict[name], stats_dict[name])
        val = next(iter(result.values())) if isinstance(result, dict) else result
        grid.loc[name, name] = f"{val:.4f}"
    return grid



def make_comparison_grid_versus(names_1, names_2, stats_1, stats_2, compare_func):
    """
    Create a DataFrame comparing all pairs between two groups (names_1 and names_2) using compare_func. If the groups aren't exactly complementary, the missing matches will be ignored.
    Example of comparison:
        stats_dict_1       stats_dict_2     => comparison_result
        [a] [b] [ ]        [ ] [k] [ ]          [ ]         [f(b,k)]  [ ]
        [d] [ ] [ ]        [l] [m] [n]          [f(d,l)]    [ ]       [ ]
        [ ] [h] [i]        [ ] [p] [q]          [ ]         [f(h,p)]  [f(i,q)]
    Args:
        names_1 (list): List of names for the first group.
            Example: [MATVIN/NEUTRAL_Fear, FEDMAR/SAD_canonical]
        names_2 (list): List of names for the second group.
        stats_dict_1 (dict): Dictionary containing statistics for the first group.
            Example: {"MATVIN/NEUTRAL_Fear": {...}, "FEDMAR/SAD_canonical": {...}}
        stats_dict_2 (dict): Dictionary containing statistics for the second group.
        compare_func (callable): Function used to compare statistics between groups.
    Returns:
        pd.DataFrame: DataFrame containing the comparison results.
    """
    subject1 = names_1[0].split("/")[0]
    subject2 = names_2[0].split("/")[0]

    for name_1 in names_1:
        if not name_1.startswith(subject1 + "/"):
            raise ValueError(f"All names in names_1 should start with the same subject identifier '{subject1}/'. Found '{name_1}'.")

    for name_2 in names_2:
        if not name_2.startswith(subject2 + "/"):
            raise ValueError(f"All names in names_2 should start with the same subject identifier '{subject2}/'. Found '{name_2}'.")

    grid = pd.DataFrame(index=EMOTIONS, columns=EMOTIONS)
    for emotion_gt in EMOTIONS:
        for emotion_pred in EMOTIONS:
            emotion_pred_capitalized = emotion_pred.capitalize()
            emotion_pred_altname = EMOTIONS_PRED[emotion_pred]  
            if emotion_gt == emotion_pred or emotion_pred_altname == emotion_gt:
                emotion_full = f"{emotion_gt.upper()}_canonical"
            else:
                # CAPS for GT, normal case for Pred
                emotion_full = f"{emotion_gt.upper()}_{emotion_pred_altname}"

            try:
                entry_1 = f"{subject1}/{emotion_full}"
                entry_2 = f"{subject2}/{emotion_full}"
                result = compare_func(stats_1[entry_1], stats_2[entry_2])
            except KeyError:
                emotion_full = f"{emotion_gt.upper()}_{emotion_pred_capitalized}"
                try:
                    entry_1 = f"{subject1}/{emotion_full}"
                    entry_2 = f"{subject2}/{emotion_full}"
                    result = compare_func(stats_1[entry_1], stats_2[entry_2])
                except KeyError:
                    # If it still fails, set result to np.nan
                    result = np.nan

            val = next(iter(result.values())) if isinstance(result, dict) else result
            grid.loc[emotion_gt, emotion_pred] = f"{val:.4f}"

    grid = grid.rename_axis("Ground Truth", axis="index").rename_axis("Predicted", axis="columns")
    return grid

def show_grid_matplotlib(df, title, cmap='ocean_r', font_size=10, axis_names=None, save_folder=None, save_only=False, block=True, black_nan=False):
    # Convert DataFrame to a numeric matrix, replacing "-" and non-numeric with np.nan
    matrix = []
    for _, row in df.iterrows():  # No need to use the index
        row_vals = []
        for val in row:  # Iterate directly over the row values
            try:
                row_vals.append(float(val))
            except:
                row_vals.append(np.nan)
        matrix.append(row_vals)  # <- append each row inside the loop

    matrix = np.array(matrix)
    global_mean = np.nanmean(matrix)

    fig, ax = plt.subplots(figsize=(0.7 * len(df.columns), 0.7 * len(df.index)))
    # force origin and nearest interpolation for exact cell alignment, keep aspect equal
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, origin='upper', interpolation='nearest', aspect='equal')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=font_size)
    ax.set_yticklabels(df.index, fontsize=font_size)

    # ensure y coordinates align with rows (0 at top)
    ax.set_ylim(len(df.index) - 0.5, -0.5)

    # Loop over data dimensions and create text annotations.
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            is_nan = False
            try:
                fval = float(val)
                if np.isnan(fval):
                    is_nan = True
            except:
                is_nan = True

            if is_nan:
                if black_nan:
                    # draw a filled black rectangle covering the cell
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='black', edgecolor=None, zorder=3)
                    ax.add_patch(rect)
                else:
                    # leave the cell to be rendered by imshow (NaN -> colormap 'bad' color)
                    # optionally show 'nan' text instead of rectangle (comment out if not wanted)
                    # ax.text(j, i, "nan", ha="center", va="center", color="black", fontsize=font_size)
                    pass
            else:
                ax.text(j, i, f"{float(val):.3f}", ha="center", va="center", color="black", fontsize=font_size)

    # Add axis labels
    if axis_names:
        fig.text(0.04, 0.5, axis_names[0], va='center', ha='center', rotation='vertical', fontsize=font_size+4, fontweight='bold')
        fig.text(0.5, 0.92, axis_names[1], va='center', ha='center', fontsize=font_size+4, fontweight='bold')
        fig.text(0.5, 0.04, f"Global Mean: {global_mean:.4f}", va='center', ha='center', fontsize=font_size, fontstyle='italic')

    fig.suptitle(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Clean the title to make a valid filename
        filename = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        filename = filename.replace(" ", "_") + ".png"
        full_path = os.path.join(save_folder, filename)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"    >> Saved grid to {full_path}")

    # Show the plot only if save_only is False
    if not save_only:
        plt.show(block=block)
    
    plt.close()

def show_heatmaps(heatmaps, row_len=4, title_fontsize=8, alpha=0.5):
    """
    Displays the heatmaps using matplotlib, overlaying each on its base face image.
    Args:
        heatmaps (dict): Keys are filenames (should contain emotion), values are heatmap arrays.
        row_len (int): Number of heatmaps per row.
        title_fontsize (int): Font size for subplot titles.
        alpha (float): Transparency of the heatmap overlay (0=only image, 1=only heatmap).
    """
    n = len(heatmaps)
    ncols = min(row_len, n)
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axs = np.array(axs).reshape(-1)

    for i, (fname, heatmap) in enumerate(heatmaps.items()):
        # Extract emotion from fname (assumes format like 'AGAPIG/ANGRY' or 'ANGRY_canonical.npy')
        if "/" in fname:
            emotion = fname.split("/")[-1].split("_")[0]
        else:
            emotion = fname.split("_")[0]

        # Get the base face image for this emotion. If the emotion is not found, use NEUTRAL default.
        base_face = get_base_face(emotion)
        img = base_face.image
        # Convert BGR (OpenCV) to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img_rgb, interpolation='nearest')

        # Overlay the heatmap (assume heatmap is normalized 0-1, resize if needed)
        if img is not None and heatmap.shape[:2] != img.shape[:2]:
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        else:
            heatmap_resized = heatmap

        axs[i].imshow(heatmap_resized, cmap='turbo', alpha=alpha, interpolation='nearest', vmin=0, vmax=1)
        axs[i].set_title(fname, fontsize=title_fontsize)
        axs[i].axis('off')

    # Hide unused axes
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show(block=False)

def show_heatmaps_grid(heatmaps_dict, title, alpha=0.5, title_fontsize=10, save_folder=None, save_only=False, block=True):
    """
    Displays a grid of heatmaps overlayed on base faces, arranged by [Ground Truth, Predicted] emotions.
    Args:
        heatmaps_dict (dict): Keys are unique names (should contain GT and predicted emotion), values are heatmap arrays.
        alpha (float): Transparency of the heatmap overlay.
        title_fontsize (int): Font size for labels.
    """
    n = len(EMOTIONS)
    fig, axs = plt.subplots(n, n, figsize=(2.5*n, 2.5*n))

    for i, emotion_gt in enumerate(EMOTIONS):
        for j, emotion_pred in enumerate(EMOTIONS):
            if emotion_gt == emotion_pred:
                emotion_full = f"{emotion_gt.upper()}_canonical"
            else:
                emotion_full = f"{emotion_gt.upper()}_{EMOTIONS_PRED[emotion_pred]}"

            # Try to find the corresponding heatmap
            heatmap = None
            for key in heatmaps_dict:
                if key.endswith(emotion_full):
                    heatmap = heatmaps_dict[key]
                    break

            ax = axs[i, j]

            # Get base face
            base_face = get_base_face(emotion_pred)  # show predicted face in column
            img = base_face.image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb, interpolation='nearest')

            if heatmap is not None:
                if heatmap.shape[:2] != img.shape[:2]:
                    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                else:
                    heatmap_resized = heatmap
                ax.imshow(heatmap_resized, cmap='turbo', alpha=alpha, interpolation='nearest', vmin=0, vmax=1)

            ax.set_xticks([])
            ax.set_yticks([])

        # Label rows (True emotions)
        axs[i, 0].set_ylabel(emotion_gt.capitalize(), fontsize=title_fontsize, labelpad=10)

    # Label columns (Predicted emotions)
    for j, emotion_pred in enumerate(EMOTIONS):
        axs[0, j].set_title(emotion_pred.capitalize(), fontsize=title_fontsize)

    # Add big labels "True" and "Predicted"
    fig.text(0.04, 0.5, 'True', va='center', ha='center', rotation='vertical', fontsize=title_fontsize+4, fontweight='bold')
    fig.text(0.5, 0.94, 'Predicted', va='center', ha='center', fontsize=title_fontsize+4, fontweight='bold')

    fig.suptitle(title)

    plt.tight_layout(rect=[0.08, 0.08, 0.95, 0.9])

    # Save the figure if save_path is provided
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Clean the title to make a valid filename
        filename = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        filename = filename.replace(" ", "_") + ".png"
        full_path = os.path.join(save_folder, filename)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"    >> Saved grid to {full_path}")

    # Show the plot only if save_only is False
    if not save_only:
        plt.show(block=block)
        
    plt.close()
