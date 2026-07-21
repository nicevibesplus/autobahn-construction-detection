import rasterio
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from skimage.morphology import skeletonize


def run_lane_marking_detection(raster_path, white_out, orange_out, white_ref_out, orange_ref_out):
    print("Detecting lane markings (white / orange)...")

    with rasterio.open(raster_path) as src:
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
        profile = src.profile

    img_bgr = np.dstack([b, g, r]).astype(np.uint8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    orange_mask = cv2.inRange(hsv, (10, 100, 100), (35, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    combined_mask = cv2.bitwise_or(white_mask, orange_mask)
    enhanced = cv2.bitwise_and(tophat, tophat, mask=combined_mask)

    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(enhanced)

    white_line_img = np.zeros_like(gray, dtype=np.uint8)
    orange_line_img = np.zeros_like(gray, dtype=np.uint8)

    if lines is not None:
        # Vectorized midpoint classification (was a per-line Python loop)
        L = lines.reshape(-1, 4).astype(int)
        mx = np.clip((L[:, 0] + L[:, 2]) // 2, 0, gray.shape[1] - 1)
        my = np.clip((L[:, 1] + L[:, 3]) // 2, 0, gray.shape[0] - 1)
        is_white = white_mask[my, mx] > 0
        is_orange = (~is_white) & (orange_mask[my, mx] > 0)

        for x1, y1, x2, y2 in L[is_white]:
            cv2.line(white_line_img, (x1, y1), (x2, y2), 255, 2)
        for x1, y1, x2, y2 in L[is_orange]:
            cv2.line(orange_line_img, (x1, y1), (x2, y2), 255, 2)

    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(white_out, "w", **profile) as dst:
        dst.write(white_line_img, 1)
    with rasterio.open(orange_out, "w", **profile) as dst:
        dst.write(orange_line_img, 1)

    refine_markings(white_out, white_ref_out)
    refine_markings(orange_out, orange_ref_out)

    print(f"White line pixels: {(white_line_img > 0).sum()}")
    print(f"Orange line pixels: {(orange_line_img > 0).sum()}")


def refine_markings(in_path, out_path, extension_pixels=20):
    print(f"Refining markings ({Path(in_path).name})...")

    MIN_LENGTH_M = 8
    with rasterio.open(in_path) as src:
        mask = src.read(1)
        profile = src.profile
        pixel_size = src.transform[0]

    if mask.sum() == 0:
        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask.astype(rasterio.uint8), 1)
        return

    binary_mask = (mask > 0).astype(np.uint8)

    # 1-pixel-wide skeletons let us analyze line paths and directions
    skeleton = skeletonize(binary_mask).astype(np.uint8)

    # Endpoints are skeleton pixels with only one 8-neighbor
    kernel_neighbors = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel_neighbors)
    endpoints = (skeleton == 1) & (neighbor_count == 1)
    end_y, end_x = np.where(endpoints)

    # Extend outwards from each endpoint along its local orientation
    extension_mask = np.zeros_like(binary_mask)
    h, w = skeleton.shape
    for ey, ex in tqdm(list(zip(end_y, end_x)), desc="Extending endpoints"):
        y0, y1 = max(0, ey - 3), min(h, ey + 4)
        x0, x1 = max(0, ex - 3), min(w, ex + 4)
        window = skeleton[y0:y1, x0:x1]
        wy, wx = np.where(window == 1)

        if len(wy) > 1:
            pts = np.column_stack((wx + x0, wy + y0))
            distances = np.sum((pts - np.array([ex, ey])) ** 2, axis=1)
            far_pt = pts[np.argmax(distances)]

            dx = far_pt[0] - ex
            dy = far_pt[1] - ey
            length = np.hypot(dx, dy)

            if length > 0:
                dx = dx / length
                dy = dy / length
                end_extended_x = int(round(ex + dx * extension_pixels))
                end_extended_y = int(round(ey + dy * extension_pixels))
                cv2.line(extension_mask, (ex, ey), (end_extended_x, end_extended_y), 1, thickness=1)

    extended_mask = np.maximum(binary_mask, extension_mask) * 255

    # Capture stats so we can use per-component bounding boxes
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        extended_mask.astype(np.uint8), connectivity=8)

    keep_mask = np.zeros_like(extended_mask)
    for label_id in tqdm(range(1, n_labels), desc=f"Filtering components ({Path(in_path).name})"):
        x, y, bw, bh, area = stats[label_id]

        # Cheap reject: bbox diagonal is an upper bound on true length
        if np.hypot(bw, bh) * pixel_size < MIN_LENGTH_M:
            continue

        # Work only inside the component's bounding box (was full-image scans)
        sub_labels = labels[y:y + bh, x:x + bw]
        component = (sub_labels == label_id).astype(np.uint8) * 255

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(contours[0])
        length_m = max(rw, rh) * pixel_size
        if length_m >= MIN_LENGTH_M:
            keep_mask[y:y + bh, x:x + bw][sub_labels == label_id] = 255

    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(keep_mask.astype(rasterio.uint8), 1)