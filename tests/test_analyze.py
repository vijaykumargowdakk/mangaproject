"""Test Stage 3 -- AI Scene Analysis."""

import sys
import json
import uuid
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_ollama_check():
    """Test Ollama availability check (mocked)."""
    from pipeline.analyze import check_ollama_available

    # Test with mocked unavailable server
    with patch("pipeline.analyze.requests.get", side_effect=Exception("Connection refused")):
        assert not check_ollama_available("http://localhost:11434"), \
            "Should return False when Ollama is down"

    # Test with mocked available server
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("pipeline.analyze.requests.get", return_value=mock_resp):
        assert check_ollama_available("http://localhost:11434"), \
            "Should return True when Ollama is up"

    print("  - Ollama availability check: OK")


def test_json_parsing():
    """Test JSON response parsing with various formats."""
    from pipeline.analyze import _parse_json_response

    # Clean JSON
    result = _parse_json_response('{"characters": ["Hero"], "mood": "action"}')
    assert result is not None and result["mood"] == "action"

    # JSON in markdown fences
    result = _parse_json_response('```json\n{"characters": [], "mood": "calm"}\n```')
    assert result is not None and result["mood"] == "calm"

    # JSON with extra text
    result = _parse_json_response('Here is the result: {"mood": "tense", "action": "fighting"}')
    assert result is not None and result["mood"] == "tense"

    # Malformed text
    result = _parse_json_response("This is not JSON at all")
    assert result is None

    print("  - JSON parsing with various formats: OK")


def test_analyze_with_mock():
    """Test full analysis pipeline with mocked Ollama."""
    from test_data.generate_test_data import generate_sample_manga_page
    from pipeline.analyze import analyze_panels

    test_dir = ROOT / "test_data"
    png_path = test_dir / "sample_manga_page.png"
    if not png_path.exists():
        generate_sample_manga_page(png_path)

    job_id = f"test_{uuid.uuid4().hex[:8]}"

    # Create a panels dir with the test image
    job_dir = ROOT / "work" / job_id / "panels"
    job_dir.mkdir(parents=True, exist_ok=True)
    import shutil as sh
    sh.copy2(png_path, job_dir / "panel_0000.png")

    # Mock Ollama responses
    mock_ollama_response = {
        "response": json.dumps({
            "characters": ["Hero"],
            "action": "Standing dramatically",
            "emotion": "determined",
            "dialogue": "Let's go!",
            "mood": "action",
            "is_splash": False,
        })
    }

    mock_tags = MagicMock()
    mock_tags.status_code = 200

    mock_generate = MagicMock()
    mock_generate.status_code = 200
    mock_generate.json.return_value = mock_ollama_response
    mock_generate.raise_for_status = MagicMock()

    def mock_get(url, **kwargs):
        return mock_tags

    def mock_post(url, **kwargs):
        return mock_generate

    with patch("pipeline.analyze.requests.get", side_effect=mock_get):
        with patch("pipeline.analyze.requests.post", side_effect=mock_post):
            results = analyze_panels(job_dir, job_id)

    assert len(results) > 0, "No results returned"
    assert results[0]["mood"] == "action", f"Expected 'action', got '{results[0]['mood']}'"
    assert results[0]["dialogue"] == "Let's go!"

    # Check panels.json was written
    panels_json = ROOT / "work" / job_id / "panels.json"
    assert panels_json.exists(), "panels.json not created"

    print(f"  - Analyzed {len(results)} panel(s) with mocked Ollama: OK")

    # Cleanup
    full_job_dir = ROOT / "work" / job_id
    if full_job_dir.exists():
        shutil.rmtree(full_job_dir)


def test_error_message():
    """Test that clear error message is shown when Ollama is down."""
    from pipeline.analyze import analyze_panels

    job_id = f"test_{uuid.uuid4().hex[:8]}"
    job_dir = ROOT / "work" / job_id / "panels"
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy panel
    from PIL import Image
    img = Image.new("RGB", (100, 100), "white")
    img.save(str(job_dir / "panel_0000.png"))

    with patch("pipeline.analyze.requests.get", side_effect=Exception("refused")):
        try:
            analyze_panels(job_dir, job_id)
            assert False, "Should have raised ConnectionError"
        except ConnectionError as e:
            assert "Ollama is not running" in str(e)
            print("  - Ollama-not-running error message: OK")

    # Cleanup
    full_job_dir = ROOT / "work" / job_id
    if full_job_dir.exists():
        shutil.rmtree(full_job_dir)


if __name__ == "__main__":
    print("=" * 50)
    print("TEST: Stage 3 -- AI Scene Analysis")
    print("=" * 50)
    try:
        test_ollama_check()
        test_json_parsing()
        test_analyze_with_mock()
        test_error_message()
        print("[PASS] Stage 3 analysis works correctly.")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
