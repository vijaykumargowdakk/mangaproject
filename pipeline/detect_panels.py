"""Stage 2 — Panel Detection & Cropping: Detect and crop individual manga panels."""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from pipeline import load_config, get_job_dir


def detect_panels(
    pages_dir: str | Path,
    job_id: str,
    config: dict | None = None,
) -> list[dict]:
    """Detect and crop individual panels from manga page images.

    Args:
        pages_dir: Directory containing page PNG images.
        job_id: Unique job identifier.
        config: Configuration dict. Loaded from config.yaml if None.

    Returns:
        List of panel info dicts with keys: panel_id, image_path,
        page_index, bbox (x, y, w, h).
    """
    pages_dir = Path(pages_dir)
    if not pages_dir.exists():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")

    if config is None:
        config = load_config()

    det_cfg = config.get("panel_detection", {})
    direction = det_cfg.get("reading_direction", "rtl")
    min_area_pct = det_cfg.get("min_panel_area_pct", 0.04)
    padding = det_cfg.get("panel_padding_px", 12)
    row_overlap = det_cfg.get("row_overlap_threshold", 0.5)

    # Set up output
    job_dir = get_job_dir(job_id)
    panels_dir = job_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # Collect page images sorted by name
    page_files = sorted(
        [f for f in pages_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda p: p.name,
    )

    if not page_files:
        raise ValueError(f"No page images found in {pages_dir}")

    all_panels = []
    panel_counter = 0

    for page_idx, page_path in enumerate(page_files):
        img = cv2.imread(str(page_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        page_area = h * w
        min_area = page_area * min_area_pct

        # Detect panels using OpenCV contour detection
        bboxes = _detect_panels_opencv(img, min_area)

        # If no panels found, treat entire page as one panel
        if not bboxes:
            bboxes = [(0, 0, w, h)]

        # Sort panels in reading order
        bboxes = _sort_reading_order(bboxes, direction, row_overlap)

        # Crop and save each panel
        for bbox in bboxes:
            x, y, bw, bh = bbox

            # Apply padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bw + padding)
            y2 = min(h, y + bh + padding)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            panel_id = f"panel_{panel_counter:04d}"
            panel_path = panels_dir / f"{panel_id}.png"
            cv2.imwrite(str(panel_path), crop)

            all_panels.append({
                "panel_id": panel_id,
                "image_path": str(panel_path),
                "page_index": page_idx,
                "page_file": page_path.name,
                "bbox": {"x": x, "y": y, "w": bw, "h": bh},
            })
            panel_counter += 1

    # Write manifest
    manifest_path = job_dir / "panels_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_panels, f, indent=2, default=str)

    return all_panels


def _detect_panels_opencv(img: np.ndarray, min_area: float) -> list[tuple]:
    """Detect panel bounding boxes using OpenCV contour detection.

    Args:
        img: Input image as numpy array (BGR).
        min_area: Minimum area threshold for valid panels.

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary threshold — invert so panel borders (dark lines) become white
    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Invert: we want the panel interiors as white regions
    inverted = cv2.bitwise_not(binary)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find external contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Filter by minimum area
        if area < min_area:
            continue

        # Filter out very thin slices (gutters)
        aspect = w / h if h > 0 else 0
        if aspect > 15 or aspect < 1/15:
            continue

        bboxes.append((x, y, w, h))

    return bboxes


def _sort_reading_order(
    bboxes: list[tuple],
    direction: str = "rtl",
    row_overlap_threshold: float = 0.5,
) -> list[tuple]:
    """Sort bounding boxes in manga reading order.

    Groups panels into rows based on vertical overlap, then sorts:
    - RTL: right-to-left within each row (Japanese manga)
    - LTR: left-to-right within each row (Western comics)

    Args:
        bboxes: List of (x, y, w, h) bounding boxes.
        direction: 'rtl' or 'ltr'.
        row_overlap_threshold: Minimum vertical overlap ratio to group into same row.

    Returns:
        Sorted list of bounding boxes.
    """
    if not bboxes:
        return []

    # Sort by y coordinate first
    sorted_by_y = sorted(bboxes, key=lambda b: b[1])

    # Group into rows
    rows = []
    current_row = [sorted_by_y[0]]

    for bbox in sorted_by_y[1:]:
        # Check vertical overlap with the first box in current row
        ref = current_row[0]
        ref_top, ref_bottom = ref[1], ref[1] + ref[3]
        cur_top, cur_bottom = bbox[1], bbox[1] + bbox[3]

        overlap_start = max(ref_top, cur_top)
        overlap_end = min(ref_bottom, cur_bottom)
        overlap = max(0, overlap_end - overlap_start)

        ref_height = ref_bottom - ref_top
        overlap_ratio = overlap / ref_height if ref_height > 0 else 0

        if overlap_ratio >= row_overlap_threshold:
            current_row.append(bbox)
        else:
            rows.append(current_row)
            current_row = [bbox]

    rows.append(current_row)

    # Sort within each row by x coordinate
    result = []
    for row in rows:
        if direction == "rtl":
            row_sorted = sorted(row, key=lambda b: b[0], reverse=True)
        else:
            row_sorted = sorted(row, key=lambda b: b[0])
        result.extend(row_sorted)

    return result


if __name__ == "__main__":
    import sys
    import uuid

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.detect_panels <pages_dir>")
        sys.exit(1)

    pages = sys.argv[1]
    jid = str(uuid.uuid4())[:8]
    print(f"Detecting panels in: {pages}")
    print(f"Job ID: {jid}")

    try:
        panels = detect_panels(pages, jid)
        print(f"[SUCCESS] Detected {len(panels)} panels in work/{jid}/panels/")
        for p in panels:
            print(f"  - {p['panel_id']} from {p['page_file']} at ({p['bbox']['x']},{p['bbox']['y']})")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
