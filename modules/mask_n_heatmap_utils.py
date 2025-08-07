import cv2
from matplotlib import pyplot as plt
import numpy as np

from modules.visualize import plot_matrix

DISPLAY_OPTIONS = {
    "default":  {"radius": 200.0, "axis_ratio": 0.45, "gradient_exp": 1.5},
    "AU1" :     {"radius": 200.0, "axis_ratio": 0.45, "gradient_exp": 1.5},
    "AU2" :     {"radius": 200.0, "axis_ratio": 0.45, "gradient_exp": 1.5},
    "AU4" :     {"radius": 200.0, "axis_ratio": 0.45, "gradient_exp": 1.5},
    "AU5" :     {"radius": 200.0, "axis_ratio": 0.45, "gradient_exp": 1.5},
    "AU6" :     {"radius": 200.0, "axis_ratio": 0.60, "gradient_exp": 2.2},
    "AU9" :     {"radius": 200.0, "axis_ratio": 1.20, "gradient_exp": 3.0},
    "AU10":     {"radius": 200.0, "axis_ratio": 1.0 , "gradient_exp": 3.0},
    "AU12":     {"radius": 230.0, "axis_ratio": 1.0 , "gradient_exp": 3.0},
    "AU15":     {"radius": 270.0, "axis_ratio": 1.0 , "gradient_exp": 3.0},
    "AU16":     {"radius": 270.0, "axis_ratio": 0.8 , "gradient_exp": 3.0},
    "AU23":     {"radius": 270.0, "axis_ratio": 0.8 , "gradient_exp": 3.0},
    "AU24":     {"radius": 270.0, "axis_ratio": 0.8 , "gradient_exp": 3.0},
    "AU26":     {"radius": 270.0, "axis_ratio": 0.8 , "gradient_exp": 3.0},
    "AU27":     {"radius": 270.0, "axis_ratio": 0.8 , "gradient_exp": 3.0},
}

def mask_face_dots(img, landmark_coords, radius=5, color=(0, 255, 0)):
    """
    Masks the face with simple dots at the landmark coordinates.
    """
    for (x, y) in landmark_coords:
        cv2.circle(img, (x, y), radius, color, -1)  # Fill the circle
    return img

def mask_face_circles(img, landmark_coords, AU="default"):
    radius = DISPLAY_OPTIONS[AU]["radius"] if AU in DISPLAY_OPTIONS else DISPLAY_OPTIONS["default"]["radius"]
    axis_ratio = DISPLAY_OPTIONS[AU]["axis_ratio"] if AU in DISPLAY_OPTIONS else DISPLAY_OPTIONS["default"]["axis_ratio"]
    angle_deg = 0.0
    wobble_amp = 0.0
    gradient_exp = DISPLAY_OPTIONS[AU]["gradient_exp"] if AU in DISPLAY_OPTIONS else DISPLAY_OPTIONS["default"]["gradient_exp"]
    ungrad_slice = 0.15
    color = (0,0,0)
    close_size = 20   

    img = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    alpha = np.zeros((h, w), dtype=np.float32)

    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    # build soft spots
    for (cx, cy) in landmark_coords:
        x0 = max(int(cx-radius), 0)
        x1 = min(int(cx+radius+1), w)
        y0 = max(int(cy-radius), 0)
        y1 = min(int(cy+radius+1), h)

        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        dx, dy = xs - cx, ys - cy

        xr = (c*dx + s*dy) / radius
        yr = (-s*dx + c*dy) / (radius * axis_ratio)
        dist = np.sqrt(xr*xr + yr*yr)
        if wobble_amp:
            ang = np.arctan2(yr, xr)
            dist *= (1 + wobble_amp*np.sin(5*ang))

        spot = np.clip(1.0 + ungrad_slice - dist, 0.0, 1.0) ** gradient_exp
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], spot)

    # close small gaps 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

    # Create an overlay with the mask color
    overlay = np.zeros_like(img)
    overlay[:] = np.array(color[::-1], np.float32) / 255.0  # BGR → RGB

    # Blend the overlay with the original image using the alpha mask
    img = img * (1 - alpha[:, :, None]) + overlay * alpha[:, :, None]

    # Convert back to 8-bit RGB and return
    return (img * 255).astype(np.uint8)

def mask_face_lines(image, landmark_coordinates,
              expansion=37,
              blur_radius=None,
              fill=False,
              fade_start=None,
              fade_end=None,
              mask_color=(0, 0, 0)):
    """
    Draw a smoothly-connected, expanded black mask with a distance-based transparency fade.

    Parameters
    ----------
    image : np.ndarray
        BGR image to draw on.
    landmark_coordinates : list of (x, y)
        Points to connect in order.
    expansion : int, default=15
        How far the base line/polygon is dilated (thickness in px).
    blur_radius : int or None, default=None
        Radius of Gaussian blur before dilation; if None, uses max(1, expansion//2).
    fill : bool, default=False
        If True, fills the polygon defined by `landmark_coordinates` before blur/dilate.
    fade_start : int or None, default=None
        Distance (in px) from the dilated mask at which fading begins (fully opaque inside).
        If None, defaults to expansion//2.
    fade_end : int or None, default=None
        Distance (in px) from the dilated mask at which fading finishes (fully transparent outside).
        If None, defaults to expansion.

    Returns
    -------
    out : np.ndarray
        Copy of `image` with the mask region painted black, fading to transparent.
    """
    # Prepare single-channel mask
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(landmark_coordinates, dtype=np.int32).reshape(-1,1,2)

    # 1) Draw line
    cv2.polylines(mask, [pts], isClosed=False, color=255,
                  thickness=1, lineType=cv2.LINE_AA)

    # 2) Optional fill
    if fill:
        cv2.fillPoly(mask, [pts], color=255)

    # 3) Blur to smooth
    if blur_radius is None:
        blur_radius = max(1, expansion // 4)
    if blur_radius % 2 == 0:
        blur_radius += 1
    mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), sigmaX=0)

    # 4) Threshold back to binary
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # 5) Dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (expansion, expansion))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    # 6) Distance transform on the inverse
    inv = cv2.bitwise_not(dilated)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    # 7) Build alpha map
    if fade_start is None:
        fade_start = expansion // 4
    if fade_end is None:
        fade_end = expansion * 2
    # Avoid division by zero
    span = max(1, fade_end - fade_start)

    # alpha=1 where dist<=fade_start; alpha=0 where dist>=fade_end
    alpha = np.clip((fade_end - dist) / span, 0.0, 1.0)

    # Also force fully opaque inside the dilated mask
    alpha[dilated > 0] = 1.0

    # 8) Composite black over image with per-pixel alpha
    #    out = image*(1-alpha) + black*alpha  -> simply image*(1-alpha)
    # Expand alpha to 3 channels
    alpha_3ch = np.dstack([alpha]*3)
    # Prepare color layer (broadcast mask_color over the image shape)
    color_layer = np.full_like(image, mask_color, dtype=np.float32)
    out = image.astype(np.float32) * (1.0 - alpha_3ch) + color_layer * alpha_3ch

    return out.astype(np.uint8)



def apply_inverse_masks(image, list_of_au_configs, mask_color=(0, 0, 0)):
    """
    image : np.ndarray, original BGR image
    list_of_au_configs : list of dict, each with keys
        landmark_coordinates, expansion, blur_radius, fill, fade_start, fade_end
    """
    def compute_mask_alpha(landmark_coordinates,
                        image_shape,
                        expansion=37,
                        blur_radius=None,
                        fill=False,
                        fade_start=None,
                        fade_end=None):
        """
        Compute a float alpha mask (0…1) for a dilated, faded band/polygon.

        Returns
        -------
        alpha : np.ndarray, shape=(h,w), dtype=float32
            Per-pixel opacity of the mask (1=fully keep; 0=fully black).
        """
        h, w = image_shape[:2]
        # -- 1) build binary dilated mask exactly like before
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(landmark_coordinates, dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(mask, [pts], isClosed=False, color=255,
                    thickness=1, lineType=cv2.LINE_AA)
        if fill:
            cv2.fillPoly(mask, [pts], color=255)

        if blur_radius is None:
            blur_radius = max(1, expansion // 4)
        if blur_radius % 2 == 0:
            blur_radius += 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), sigmaX=0)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (expansion, expansion))
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # -- 2) distance-based fade
        inv = cv2.bitwise_not(dilated)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

        if fade_start is None:
            fade_start = expansion // 4
        if fade_end is None:
            fade_end = expansion * 2
        span = max(1, fade_end - fade_start)

        alpha = np.clip((fade_end - dist) / span, 0.0, 1.0).astype(np.float32)
        alpha[dilated > 0] = 1.0

        return alpha

    h, w = image.shape[:2]
    # start with zero alpha (fully black)
    alpha_total = np.zeros((h, w), dtype=np.float32)

    # accumulate each mask's alpha
    for cfg in list_of_au_configs:
        alpha = compute_mask_alpha(
            cfg['landmark_coordinates'],
            image.shape,
            expansion=cfg.get('expansion', 37),
            blur_radius=cfg.get('blur_radius'),
            fill=cfg.get('fill'),
            fade_start=cfg.get('fade_start'),
            fade_end=cfg.get('fade_end'),
        )
        alpha_total = np.maximum(alpha_total, alpha)

    # composite once
    alpha_3ch = np.dstack([alpha_total]*3)
    color_layer = np.full_like(image, mask_color, dtype=np.float32)
    out = image.astype(np.float32) * alpha_3ch + color_layer * (1.0 - alpha_3ch)

    return out.astype(np.uint8)

def get_roi_matrix(image_shape, landmark_coordinates, fill,
              expansion=37,
              blur_radius=None,
              fade_start=None,
              fade_end=None,
              debug=False):
    """
    Draw a smoothly-connected, expanded black mask with a distance-based transparency fade.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(landmark_coordinates, dtype=np.int32).reshape(-1,1,2)

    debug_imgs = []
    debug_titles = []

    debug = False
    if debug:
        debug_imgs.append(mask.copy())
        debug_titles.append("Mask @ step 0")

    # 1) Draw line
    cv2.polylines(mask, [pts], isClosed=False, color=255,
                  thickness=1, lineType=cv2.LINE_AA)

    if debug:
        debug_imgs.append(mask.copy())
        debug_titles.append("Mask @ step 1")

    # 2) Optional fill
    cv2.fillPoly(mask, [pts], color=255)

    if debug:
        debug_imgs.append(mask.copy())
        debug_titles.append("Mask @ step 2")

    # 3) Blur to smooth
    if blur_radius is None:
        blur_radius = max(1, expansion // 4)
    if blur_radius % 2 == 0:
        blur_radius += 1
    mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), sigmaX=0)

    if debug:
        debug_imgs.append(mask.copy())
        debug_titles.append("Mask @ step 3")

    # 4) Threshold back to binary
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    if debug:
        debug_imgs.append(mask.copy())
        debug_titles.append("Mask @ step 4")

    # 5) Dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (expansion, expansion))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    if debug:
        debug_imgs.append(dilated.copy())
        debug_titles.append("Mask @ step 5")

        # Show all steps in one figure
        n = len(debug_imgs)
        plt.figure(figsize=(3*n, 3))
        for i, (img, title) in enumerate(zip(debug_imgs, debug_titles)):
            plt.subplot(1, n, i+1)
            plt.imshow(img, cmap='hot', interpolation='nearest')
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    return dilated

def compute_pixel_repetition_heatmap(masked_heatmaps, debug=False):
    """
    Computes a heatmap showing the pixel repetition across all masked heatmaps.
    Each pixel is incremented by 1 for each AU where it is not NaN.

    Parameters
    ----------
    masked_heatmaps : dict of str -> np.ndarray
        Dictionary where keys are AU names and values are masked heatmaps.
        Example: {"AU1": np.ndarray, "AU2": np.ndarray, ...}
    debug : bool, optional
        If True, will plot the intermediate heatmaps for debugging.

    Returns
    -------
    repetition_heatmap : np.ndarray
        A heatmap showing the pixel repetition across all masked heatmaps.
    amt_of_overlapping_pixels : int
        Amount of overlapping pixels (only first overlap counted).
    amt_of_pixels_tot : int
        Total amount of mask pixels.
    """
    repetition_heatmap = None
    for au, masked_heatmap in masked_heatmaps.items():
        mask = ~np.isnan(masked_heatmap)  # True where not NaN
        if repetition_heatmap is None:
            repetition_heatmap = np.zeros_like(masked_heatmap, dtype=float)
        repetition_heatmap += mask.astype(float)  # Add 1 where not NaN, 0 where NaN
        if debug:
            plot_matrix(repetition_heatmap, title="Pixel Repetition Heatmap @ au: " + au)

    amt_of_overlapping_pixels = np.sum(repetition_heatmap > 1)
    amt_of_pixels_tot = np.sum(repetition_heatmap > 0)

    return repetition_heatmap, amt_of_overlapping_pixels, amt_of_pixels_tot

def invert_heatmap(heatmap):
    """
    Inverts a heatmap by dividing 1 by the heatmap values.
    Must be careful with NaNs and zeros.
    Args:
        heatmap (np.ndarray): The heatmap to invert.
    Returns:
        np.ndarray: The inverted heatmap.
    """
    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be a 2D array.")
    
    # Compute the inverse, but keep zeros as zero (not inf or nan)
    inverse_heatmap = np.zeros_like(heatmap, dtype=float)
    valid = (heatmap > 0) & (~np.isnan(heatmap))
    inverse_heatmap[valid] = 1.0 / heatmap[valid]
    # Zeros and NaNs remain zero

    return inverse_heatmap
