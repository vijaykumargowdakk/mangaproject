"""Test Stage 5 -- Video Assembly."""

import sys
import uuid
import shutil
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_fit_to_resolution():
    """Test image resize and letterboxing."""
    from PIL import Image
    from pipeline.assemble import _fit_to_resolution

    # Test landscape image
    img = Image.new("RGB", (800, 400), "red")
    fitted = _fit_to_resolution(img, 1920, 1080)
    assert fitted.size == (1920, 1080), f"Expected 1920x1080, got {fitted.size}"

    # Test portrait image
    img = Image.new("RGB", (400, 800), "blue")
    fitted = _fit_to_resolution(img, 1920, 1080)
    assert fitted.size == (1920, 1080)

    # Test exact ratio
    img = Image.new("RGB", (1920, 1080), "green")
    fitted = _fit_to_resolution(img, 1920, 1080)
    assert fitted.size == (1920, 1080)

    print("  - Image resize and letterboxing: OK")


def test_collect_panel_data():
    """Test panel data collection and audio matching."""
    from pipeline.assemble import _collect_panel_data
    from PIL import Image

    job_id = f"test_{uuid.uuid4().hex[:8]}"
    job_dir = ROOT / "work" / job_id
    panels_dir = job_dir / "panels"
    audio_dir = job_dir / "audio"
    panels_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy panels
    for i in range(3):
        img = Image.new("RGB", (200, 300), "white")
        img.save(str(panels_dir / f"panel_{i:04d}.png"))

    # Create matching audio for first two panels
    for i in range(2):
        (audio_dir / f"panel_{i:04d}.mp3").write_bytes(b"\x00" * 100)

    panels_data = [
        {"panel_id": "panel_0000", "dialogue": "Hello!", "is_splash": False, "mood": "calm"},
        {"panel_id": "panel_0001", "dialogue": "", "is_splash": True, "mood": "action"},
        {"panel_id": "panel_0002", "dialogue": "Goodbye!", "is_splash": False, "mood": "dramatic"},
    ]

    result = _collect_panel_data(panels_dir, audio_dir, panels_data)
    assert len(result) == 3, f"Expected 3 panels, got {len(result)}"
    assert result[0]["audio_path"] is not None, "First panel should have audio"
    assert result[2]["audio_path"] is None, "Third panel should not have audio"
    assert result[0]["dialogue"] == "Hello!"
    assert result[1]["is_splash"] is True

    # Cleanup
    shutil.rmtree(job_dir)
    print("  - Panel data collection: OK")


def test_video_assembly_basic():
    """Test basic video assembly (without audio, simplified)."""
    from PIL import Image
    from pipeline.assemble import assemble_video
    import json

    job_id = f"test_{uuid.uuid4().hex[:8]}"
    job_dir = ROOT / "work" / job_id
    panels_dir = job_dir / "panels"
    audio_dir = job_dir / "audio"
    video_dir = job_dir / "video"

    for d in [panels_dir, audio_dir, video_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create 2 dummy panels
    for i in range(2):
        img = Image.new("RGB", (800, 600), ["red", "blue"][i])
        img.save(str(panels_dir / f"panel_{i:04d}.png"))

    # Create panels.json
    panels_data = [
        {"panel_id": "panel_0000", "dialogue": "", "is_splash": False, "mood": "calm",
         "action": "Test scene 1", "image_path": str(panels_dir / "panel_0000.png")},
        {"panel_id": "panel_0001", "dialogue": "Test!", "is_splash": False, "mood": "action",
         "action": "Test scene 2", "image_path": str(panels_dir / "panel_0001.png")},
    ]
    with open(job_dir / "panels.json", "w") as f:
        json.dump(panels_data, f)

    # Use smaller resolution for test speed
    config = {
        "video": {
            "resolution": "640x360",
            "fps": 12,
            "codec": "libx264",
            "crf": 28,
            "ken_burns": False,  # Disable for speed
            "background_music": False,
            "dialogue_overlay": False,
        }
    }

    try:
        output = assemble_video(job_id, config)
        assert output.exists(), "Output video not created"
        assert output.stat().st_size > 0, "Output video is empty"
        print(f"  - Video assembly: OK ({output.stat().st_size:,} bytes)")
    except ImportError as e:
        print(f"  - Video assembly: SKIPPED (moviepy not installed: {e})")
    except Exception as e:
        if "ffmpeg" in str(e).lower() or "ImageMagick" in str(e):
            print(f"  - Video assembly: SKIPPED (ffmpeg/ImageMagick not available: {e})")
        else:
            raise
    finally:
        # Cleanup
        if job_dir.exists():
            shutil.rmtree(job_dir)


if __name__ == "__main__":
    print("=" * 50)
    print("TEST: Stage 5 -- Video Assembly")
    print("=" * 50)
    try:
        test_fit_to_resolution()
        test_collect_panel_data()
        test_video_assembly_basic()
        print("[PASS] Stage 5 assembly works correctly.")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
