"""Generate sample manga test data — a synthetic manga page and PDF."""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def generate_sample_manga_page(output_path: Path) -> Path:
    """Create a synthetic manga page PNG with clear panel borders.
    
    The page has a 2x3 grid of panels with simple geometric shapes
    and text, suitable for testing panel detection.
    """
    from PIL import Image, ImageDraw, ImageFont

    # Page dimensions (standard manga ~5x7 inches at 300dpi)
    width, height = 1500, 2100
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Draw page border
    border = 60
    draw.rectangle(
        [border, border, width - border, height - border],
        outline="black",
        width=3,
    )

    # Define 6 panels in a 2x3 grid with gutters
    gutter = 30
    panel_w = (width - 2 * border - gutter) // 2
    panel_h = (height - 2 * border - 2 * gutter) // 3

    panels = []
    for row in range(3):
        for col in range(2):
            x1 = border + col * (panel_w + gutter)
            y1 = border + row * (panel_h + gutter)
            x2 = x1 + panel_w
            y2 = y1 + panel_h
            panels.append((x1, y1, x2, y2))

    # Draw each panel with border and content
    contents = [
        ("Action!", "circle", "#FFE0E0"),
        ("Dramatic!", "star", "#E0FFE0"),
        ("Surprise!", "triangle", "#E0E0FF"),
        ("Tension!", "rectangle", "#FFFFE0"),
        ("Calm...", "circle", "#E0FFFF"),
        ("Finale!", "star", "#FFE0FF"),
    ]

    for i, (x1, y1, x2, y2) in enumerate(panels):
        text, shape, bg_color = contents[i]
        
        # Fill panel background
        draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline="black", width=4)

        # Draw shape in center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = min(panel_w, panel_h) // 6

        if shape == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=3)
        elif shape == "triangle":
            draw.polygon(
                [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)],
                outline="black",
                width=3,
            )
        elif shape == "rectangle":
            draw.rectangle(
                [cx - r, cy - r // 2, cx + r, cy + r // 2],
                outline="black",
                width=3,
            )
        elif shape == "star":
            import math
            points = []
            for j in range(10):
                angle = math.pi / 2 + j * math.pi / 5
                rad = r if j % 2 == 0 else r // 2
                points.append((cx + rad * math.cos(angle), cy - rad * math.sin(angle)))
            draw.polygon(points, outline="black", width=2)

        # Draw speech bubble
        bx, by = x1 + 40, y1 + 30
        bw, bh = len(text) * 12 + 30, 50
        draw.rounded_rectangle(
            [bx, by, bx + bw, by + bh], radius=15, fill="white", outline="black", width=2
        )
        
        # Draw text in bubble
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((bx + 15, by + 12), text, fill="black", font=font)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    return output_path


def generate_sample_manga_pdf(png_path: Path, pdf_path: Path) -> Path:
    """Wrap a manga page PNG into a single-page PDF using PyMuPDF."""
    import fitz

    doc = fitz.open()
    # A4-ish page
    page = doc.new_page(width=595, height=842)
    rect = page.rect
    page.insert_image(rect, filename=str(png_path))
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


if __name__ == "__main__":
    test_data_dir = ROOT / "test_data"
    
    png_path = test_data_dir / "sample_manga_page.png"
    pdf_path = test_data_dir / "sample_manga.pdf"

    print("Generating sample manga page...")
    generate_sample_manga_page(png_path)
    print(f"  → {png_path}")

    print("Generating sample manga PDF...")
    generate_sample_manga_pdf(png_path, pdf_path)
    print(f"  → {pdf_path}")

    print("[SUCCESS] Test data generated.")
