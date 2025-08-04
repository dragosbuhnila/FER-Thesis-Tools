import matplotlib.pyplot as plt
import numpy as np
import cv2

def generate_circle_image(label, size=50):
    """Generate a 50x50 image with a colored circle and label text."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv2.circle(img, (size//2, size//2), size//3, color, -1)
    cv2.putText(img, label[0], (10, size//2+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    return img

# Keep track of open window names
open_windows = []

def show_pair_images(img1_label, img2_label):
    global open_windows
    # Close any previously open windows
    for win in open_windows:
        cv2.destroyWindow(win)
    open_windows = []

    img1 = generate_circle_image(img1_label)
    img2 = generate_circle_image(img2_label)
    cv2.imshow("Image X", img1)
    cv2.imshow("Image Y", img2)
    open_windows = ["Image X", "Image Y"]

def on_click(event, grid, row_labels, col_labels):
    if event.inaxes:
        i = int(round(event.ydata))
        j = int(round(event.xdata))
        if 0 <= i < len(row_labels) and 0 <= j < len(col_labels):
            img1_label = row_labels[i]
            img2_label = col_labels[j]
            print(f"Clicked: {img1_label}, {img2_label}")
            show_pair_images(img1_label, img2_label)
        else:
            # Clicked outside valid grid, close any open windows
            for win in open_windows:
                cv2.destroyWindow(win)
            open_windows.clear()
    else:
        # Clicked outside axes, close any open windows
        for win in open_windows:
            cv2.destroyWindow(win)
        open_windows.clear()

# Example grid and labels
grid = np.random.rand(5, 5)
row_labels = ["A", "B", "C", "D", "E"]
col_labels = row_labels

fig, ax = plt.subplots()
im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels, rotation=45, ha="right")
ax.set_yticklabels(row_labels)

fig.canvas.mpl_connect(
    "button_press_event",
    lambda event: on_click(event, grid, row_labels, col_labels)
)

plt.show()