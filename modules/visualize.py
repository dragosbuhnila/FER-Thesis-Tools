from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk

import numpy as np

def plot_matrix(matrix, title="No Title", cmap='jet'):
    """
    Plots a 2D matrix as an image.
    """
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()

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

def show_grid_matplotlib(df, cmap='Reds', font_size=6):
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
    im = ax.imshow(matrix, cmap=cmap)

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

def show_heatmaps(heatmaps, row_len=4, title_fontsize=8):
    """
    Displays the heatmaps using matplotlib, putting at most n_per_line per row.
    """
    n = len(heatmaps)
    ncols = min(row_len, n)
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axs = np.array(axs).reshape(-1)  # Flatten in case nrows or ncols is 1

    for i, (fname, heatmap) in enumerate(heatmaps.items()):
        axs[i].imshow(heatmap, cmap='jet', interpolation='nearest')
        axs[i].set_title(fname, fontsize=title_fontsize)
        axs[i].axis('off')  # Hide axes

    # Hide unused axes
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show(block=False)