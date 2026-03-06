"""Test Stage 4 -- Narration & TTS."""

import sys
import json
import uuid
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_fallback_narration():
    """Test fallback narration generation (no LLM needed)."""
    from pipeline.narrate import _fallback_narration

    panel = {
        "action": "Hero charges forward",
        "emotion": "determination",
        "dialogue": "I won't give up!",
        "mood": "action",
    }

    narration = _fallback_narration(panel)
    assert "Hero charges forward" in narration
    assert "I won't give up!" in narration
    print("  - Fallback narration generation: OK")


def test_script_generation_with_mock():
    """Test script generation with mocked Ollama."""
    from pipeline.narrate import _generate_scripts

    panels = [
        {"action": "Walking", "emotion": "calm", "dialogue": "", "mood": "calm", "characters": ["Hero"]},
        {"action": "Fighting", "emotion": "intense", "dialogue": "Take this!", "mood": "action", "characters": ["Hero", "Villain"]},
    ]

    # Mock Ollama to return narrations
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "The hero walks silently through the shadowy corridor."}
    mock_resp.raise_for_status = MagicMock()

    with patch("pipeline.narrate.requests.post", return_value=mock_resp):
        scripts = _generate_scripts(panels, "http://localhost:11434", "mistral:7b", 2)

    assert len(scripts) == 2, f"Expected 2 scripts, got {len(scripts)}"
    assert len(scripts[0]) > 0, "Script should not be empty"
    print(f"  - Script generation (mocked LLM): OK ({len(scripts)} scripts)")


def test_silent_audio_creation():
    """Test silent audio placeholder creation."""
    from pipeline.narrate import _create_silent_audio

    test_path = ROOT / "work" / "test_silent.mp3"
    test_path.parent.mkdir(parents=True, exist_ok=True)

    _create_silent_audio(test_path, duration_ms=1000)
    assert test_path.exists(), "Silent audio not created"
    assert test_path.stat().st_size > 0, "Silent audio is empty"

    test_path.unlink(missing_ok=True)
    print("  - Silent audio creation: OK")


def test_full_narration_pipeline():
    """Test the full narration pipeline with mocked LLM and pyttsx3 fallback."""
    from pipeline.narrate import generate_narration

    job_id = f"test_{uuid.uuid4().hex[:8]}"
    job_dir = ROOT / "work" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create a test panels.json
    panels = [
        {
            "panel_id": "panel_0000",
            "image_path": "panels/panel_0000.png",
            "action": "A warrior stands ready",
            "emotion": "determined",
            "dialogue": "Let's begin!",
            "mood": "action",
            "characters": ["Warrior"],
            "is_splash": False,
        }
    ]
    panels_json = job_dir / "panels.json"
    with open(panels_json, "w") as f:
        json.dump(panels, f)

    # Use pyttsx3 config to avoid network dependency
    config = {
        "scene_analysis": {"ollama_host": "http://localhost:11434"},
        "narration": {
            "script_model": "mistral:7b",
            "context_window": 2,
            "tts_engine": "pyttsx3",
            "silence_pad_ms": 100,
        },
    }

    # Mock Ollama text call to avoid needing Ollama running
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "The warrior stands tall, eyes blazing with determination."}
    mock_resp.raise_for_status = MagicMock()

    with patch("pipeline.narrate.requests.post", return_value=mock_resp):
        audio_paths = generate_narration(panels_json, job_id, config)

    assert len(audio_paths) == 1, f"Expected 1 audio file, got {len(audio_paths)}"

    # Check narration scripts JSON was saved
    scripts_path = job_dir / "narration_scripts.json"
    assert scripts_path.exists(), "narration_scripts.json not created"

    print(f"  - Full narration pipeline: OK ({len(audio_paths)} audio file(s))")

    # Cleanup
    if job_dir.exists():
        shutil.rmtree(job_dir)


if __name__ == "__main__":
    print("=" * 50)
    print("TEST: Stage 4 -- Narration & TTS")
    print("=" * 50)
    try:
        test_fallback_narration()
        test_script_generation_with_mock()
        test_silent_audio_creation()
        test_full_narration_pipeline()
        print("[PASS] Stage 4 narration works correctly.")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
