"""Test Stage 1 — PDF Extraction."""

import sys
import uuid
import shutil
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_extract():
    """Test PDF page extraction end-to-end."""
    from test_data.generate_test_data import generate_sample_manga_page, generate_sample_manga_pdf
    from pipeline.extract import extract_pages

    test_dir = ROOT / "test_data"
    png_path = test_dir / "sample_manga_page.png"
    pdf_path = test_dir / "sample_manga.pdf"

    # Step 1: Generate test data if it doesn't exist
    if not png_path.exists():
        print("  Generating test data...")
        generate_sample_manga_page(png_path)
        generate_sample_manga_pdf(png_path, pdf_path)

    # Step 2: Extract pages
    job_id = f"test_{uuid.uuid4().hex[:8]}"
    pages = extract_pages(pdf_path, job_id)

    # Step 3: Validate
    assert len(pages) > 0, "No pages extracted"
    for p in pages:
        assert p.exists(), f"Page file missing: {p}"
        assert p.stat().st_size > 0, f"Page file is empty: {p}"
        assert p.suffix == ".png", f"Unexpected format: {p.suffix}"

    print(f"  Extracted {len(pages)} page(s):")
    for p in pages:
        print(f"    - {p.name} ({p.stat().st_size:,} bytes)")

    # Cleanup
    job_dir = ROOT / "work" / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("TEST: Stage 1 — PDF Extraction")
    print("=" * 50)
    try:
        result = test_extract()
        print("[PASS] Stage 1 extraction works correctly.")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
