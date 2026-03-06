"""Test Stage 2 -- Panel Detection."""

import sys
import json
import uuid
import shutil
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_detect_panels():
    """Test panel detection on synthetic manga page."""
    from test_data.generate_test_data import generate_sample_manga_page, generate_sample_manga_pdf
    from pipeline.extract import extract_pages
    from pipeline.detect_panels import detect_panels

    test_dir = ROOT / "test_data"
    png_path = test_dir / "sample_manga_page.png"
    pdf_path = test_dir / "sample_manga.pdf"

    # Ensure test data exists
    if not png_path.exists():
        generate_sample_manga_page(png_path)
        generate_sample_manga_pdf(png_path, pdf_path)

    # Extract pages first
    job_id = f"test_{uuid.uuid4().hex[:8]}"
    pages = extract_pages(pdf_path, job_id)
    assert len(pages) > 0, "No pages extracted"

    # Detect panels
    pages_dir = ROOT / "work" / job_id / "pages"
    panels = detect_panels(pages_dir, job_id)

    # Validate
    assert len(panels) > 0, "No panels detected"

    # Check panel files exist
    for panel in panels:
        panel_path = Path(panel["image_path"])
        assert panel_path.exists(), f"Panel file missing: {panel_path}"
        assert panel_path.stat().st_size > 0, f"Panel file is empty: {panel_path}"

    # Check manifest
    manifest_path = ROOT / "work" / job_id / "panels_manifest.json"
    assert manifest_path.exists(), "Manifest file missing"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    assert len(manifest) == len(panels), "Manifest panel count mismatch"

    print(f"  Detected {len(panels)} panels:")
    for p in panels:
        bbox = p["bbox"]
        print(f"    - {p['panel_id']}: page={p['page_index']}, "
              f"bbox=({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']})")

    # Cleanup
    job_dir = ROOT / "work" / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("TEST: Stage 2 -- Panel Detection")
    print("=" * 50)
    try:
        result = test_detect_panels()
        print("[PASS] Stage 2 panel detection works correctly.")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
