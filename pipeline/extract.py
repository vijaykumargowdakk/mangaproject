"""Stage 1 — PDF Extraction: Convert manga PDF pages to PNG images."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional

from pipeline import load_config, get_job_dir


def extract_pages(
    pdf_path: str | Path,
    job_id: str,
    config: dict | None = None,
    password: str | None = None,
) -> list[Path]:
    """Extract each page of a manga PDF into high-resolution PNG images.

    Args:
        pdf_path: Path to the input PDF file.
        job_id: Unique job identifier for organizing output.
        config: Configuration dict. Loaded from config.yaml if None.
        password: Optional password for DRM-protected PDFs.

    Returns:
        List of paths to extracted page PNG files, in page order.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If the PDF is encrypted and no/wrong password is given.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if config is None:
        config = load_config()

    ext_cfg = config.get("pdf_extraction", {})
    dpi_scale = ext_cfg.get("dpi_scale", 2)
    colorspace = ext_cfg.get("colorspace", "rgb")
    output_format = ext_cfg.get("output_format", "png")

    # Set up output directory
    job_dir = get_job_dir(job_id)
    pages_dir = job_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Open PDF
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    # Handle password-protected PDFs
    if doc.is_encrypted:
        if password:
            if not doc.authenticate(password):
                raise RuntimeError("Incorrect PDF password.")
        else:
            raise RuntimeError(
                "PDF is password-protected. Please provide a password."
            )

    # Choose colorspace
    cs = fitz.csRGB if colorspace == "rgb" else fitz.csGRAY

    # Extract pages
    output_paths = []
    for i, page in enumerate(doc):
        # Handle rotated pages
        rotation = page.rotation
        mat = fitz.Matrix(dpi_scale, dpi_scale)
        if rotation:
            mat = mat.prerotate(-rotation)

        pix = page.get_pixmap(matrix=mat, colorspace=cs)

        # Detect double-page spreads (aspect ratio > 1.5)
        aspect = pix.width / pix.height if pix.height > 0 else 1
        is_spread = aspect > 1.5

        # Save with zero-padded filename
        ext = "png" if output_format == "png" else "jpg"
        filename = f"page_{i:04d}.{ext}"
        if is_spread:
            filename = f"page_{i:04d}_spread.{ext}"

        output_path = pages_dir / filename
        pix.save(str(output_path))
        output_paths.append(output_path)

    doc.close()
    return output_paths


if __name__ == "__main__":
    import sys
    import uuid

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.extract <pdf_path>")
        sys.exit(1)

    pdf = sys.argv[1]
    jid = str(uuid.uuid4())[:8]
    print(f"Extracting pages from: {pdf}")
    print(f"Job ID: {jid}")

    try:
        pages = extract_pages(pdf, jid)
        print(f"[SUCCESS] Extracted {len(pages)} pages to work/{jid}/pages/")
        for p in pages:
            print(f"  → {p.name}")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
