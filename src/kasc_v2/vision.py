"""Best-effort local OpenCV image analysis for authorized demonstration photos.

Not a surveyed stadium map, gate detector, or validated routing system.
Images are processed in memory and are not saved by this function.
"""
from __future__ import annotations

import base64


def analyze_photo(image_bytes: bytes) -> dict:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('OpenCV is optional; install with pip install -e ".[vision]"') from exc

    if not image_bytes or len(image_bytes) > 8 * 1024 * 1024:
        raise ValueError("Image must contain between 1 byte and 8 MB")

    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or min(img.shape[:2]) < 120:
        raise ValueError("Invalid or too-small image: use JPEG, PNG or WEBP at least 120 px wide")

    h, w = img.shape[:2]
    if max(h, w) > 1400:
        scale = 1400 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (29, 55, 28), (94, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count_components, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    best = None
    for index in range(1, count_components):
        x, y, bw, bh, area = map(int, stats[index])
        ratio = bw / max(1, bh)
        fill_ratio = bw * bh / max(1, area)
        if area >= h * w * 0.018 and 0.23 < ratio < 4 and 0.5 < fill_ratio < 4.4:
            if best is None or area > best[-1]:
                best = (x, y, bw, bh, area)

    overlay = img.copy()
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.3)
    edges = cv2.Canny(gray, 75, 160)
    edge_count = int(cv2.countNonZero(edges))
    tint = np.zeros_like(img)
    tint[:, :, 0] = 180
    tint[:, :, 1] = 135
    tint[:, :, 2] = 60
    pixels = edges > 0
    overlay[pixels] = np.clip(0.6 * overlay[pixels] + 0.4 * tint[pixels], 0, 255).astype("uint8")

    bbox = None
    fraction = 0.0
    if best:
        x, y, bw, bh, area = best
        bbox = [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)]
        fraction = round(area / (w * h), 5)
        color = (192, 250, 81)
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, max(2, int(w / 450)))
        cv2.putText(overlay, "PITCH COLOR REGION - ESTIMATE",
                    (max(0, x), max(16, y - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.38, w / 1750), color, 2, cv2.LINE_AA)

    cv2.rectangle(overlay, (0, 0), (min(w, 620), 39), (8, 23, 31), -1)
    cv2.putText(overlay, "CV EXPERIMENT - NOT A CERTIFIED STADIUM MAP",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (203, 248, 221), 1, cv2.LINE_AA)
    ok, buffer = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Could not encode annotated image")

    return {
        "algorithm": "HSV green segmentation + morphology + Canny edge detection",
        "image_shape": [int(w), int(h)],
        "pitch_bbox": bbox,
        "pitch_fraction": fraction,
        "edge_pixel_count": edge_count,
        "status": "illustrative segmentation; not a venue survey or route geometry",
        "overlay_png_base64": base64.b64encode(buffer.tobytes()).decode("ascii"),
    }
