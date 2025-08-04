from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2

from modules.basefaces import get_base_face, basefaces

def plot_matrix(matrix, title="No Title", cmap='turbo', vmin=None, vmax=None, block=True):
    """
    Plots a 2D matrix as an image.
    """
    plt.figure()
    plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.show(block=block)

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

def show_grid_matplotlib(df, cmap='plasma', font_size=6):
    # Convert DataFrame to a numeric matrix, replacing "-" and non-numeric with np.nan
    matrix = []
    for i, row in df.iterrows():
        row_vals = []
        for val in [i] + list(row):
            try:
                row_vals.append(float(val))
            except:
                row_vals.append(np.nan)
        matrix.append(row_vals[1:])  # skip the index column for the matrix

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(0.7*len(df.columns), 0.7*len(df.index)))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=font_size)
    ax.set_yticklabels(df.index, fontsize=font_size)

    # Loop over data dimensions and create text annotations.
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            if isinstance(val, str) and val == "-":
                text = "-"
            else:
                try:
                    text = f"{float(val):.3f}"
                except:
                    text = str(val)
            ax.text(j, i, text, ha="center", va="center", color="black")

    plt.title("Pairwise Comparison Grid")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show(block=False)

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