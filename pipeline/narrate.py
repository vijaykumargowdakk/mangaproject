"""Stage 4 -- Script Writing & Voiceover: Generate narration scripts and synthesize audio."""

import asyncio
import json
import os
import requests
from pathlib import Path

from pipeline import load_config, get_job_dir


# Narrator prompt template for the text LLM
'''
NARRATOR_PROMPT = """You are narrating a manga. Given the scene description below, write
1-3 sentences of cinematic narration. Be evocative, not just descriptive.
Match the mood. If there is dialogue, quote it naturally.

Scene: {scene}
Previous context: {context}

Write ONLY the narration text, no extra commentary."""

'''

# Narrator prompt template for the text LLM
NARRATOR_PROMPT = """You are a professional voiceover narrator for a cinematic manga video.
Read the scene description and write EXACTLY 1 to 3 sentences of immersive narration.
Blend the character actions, emotions, and dialogue into a smooth, evocative read.

Scene: {scene}
Previous context: {context}

CRITICAL INSTRUCTION: Return ONLY the raw narration text. Do NOT include phrases like "Here is the narration". Do NOT wrap the text in quotes. Start the story immediately."""

def generate_narration(
    panels_json_path: str | Path,
    job_id: str,
    config: dict | None = None,
) -> list[Path]:
    """Generate narration scripts and synthesize TTS audio for each panel.

    Args:
        panels_json_path: Path to the panels.json from Stage 3.
        job_id: Unique job identifier.
        config: Configuration dict. Loaded from config.yaml if None.

    Returns:
        List of paths to generated audio files.

    Raises:
        FileNotFoundError: If panels.json does not exist.
        ConnectionError: If Ollama is needed but not available.
    """
    panels_json_path = Path(panels_json_path)
    if not panels_json_path.exists():
        raise FileNotFoundError(f"Panels JSON not found: {panels_json_path}")

    if config is None:
        config = load_config()

    narr_cfg = config.get("narration", {})
    analysis_cfg = config.get("scene_analysis", {})
    host = analysis_cfg.get("ollama_host", "http://localhost:11434")
    script_model = narr_cfg.get("script_model", "mistral:7b")
    context_window = narr_cfg.get("context_window", 2)
    tts_engine = narr_cfg.get("tts_engine", "edge")
    silence_pad_ms = narr_cfg.get("silence_pad_ms", 300)

    # Load panels data
    with open(panels_json_path, "r", encoding="utf-8") as f:
        panels = json.load(f)

    if not panels:
        raise ValueError("No panels found in panels.json")

    # Set up output directory
    job_dir = get_job_dir(job_id)
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Generate narration scripts
    narrations = _generate_scripts(panels, host, script_model, context_window)

    # Synthesize audio
    audio_paths = []
    for i, (panel, narration) in enumerate(zip(panels, narrations)):
        panel_id = panel.get("panel_id", f"panel_{i:04d}")
        audio_path = audio_dir / f"{panel_id}.mp3"

        _synthesize_audio(narration, audio_path, tts_engine, narr_cfg, silence_pad_ms)
        audio_paths.append(audio_path)

    # Save narration scripts alongside audio
    scripts_path = job_dir / "narration_scripts.json"
    script_data = [
        {"panel_id": p.get("panel_id", f"panel_{i:04d}"), "narration": n}
        for i, (p, n) in enumerate(zip(panels, narrations))
    ]
    with open(scripts_path, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=2)

    return audio_paths


def _generate_scripts(
    panels: list[dict],
    host: str,
    model: str,
    context_window: int,
) -> list[str]:
    """Generate narrator scripts for each panel using an LLM.

    Args:
        panels: List of panel analysis dicts from Stage 3.
        host: Ollama server URL.
        model: Text model name for script generation.
        context_window: Number of previous narrations to include as context.

    Returns:
        List of narration strings.
    """
    narrations = []

    for i, panel in enumerate(panels):
        # Build context from previous narrations
        start = max(0, i - context_window)
        context = " ".join(narrations[start:i]) if narrations[start:i] else "This is the beginning."

        # Build scene description
        scene = json.dumps({
            k: panel.get(k, "")
            for k in ["action", "emotion", "dialogue", "mood", "characters"]
        })

        prompt = NARRATOR_PROMPT.format(scene=scene, context=context)

        # Try Ollama first
        '''
        try:
            narration = _call_ollama_text(host, model, prompt)
            if narration and narration.strip():
                narrations.append(narration.strip())
                continue
        except Exception:
            pass
        '''
        # Try Ollama first
        try:
            narration = _call_ollama_text(host, model, prompt)
            if narration and narration.strip():
                # --- NEW: Anti-Chatter Filter for Llama 3 ---
                clean_narration = narration.strip()
                # Remove common AI intro phrases
                bad_prefixes = ["Here is the narration:", "Here's the narration:", "Narration:"]
                for prefix in bad_prefixes:
                    if clean_narration.lower().startswith(prefix.lower()):
                        clean_narration = clean_narration[len(prefix):].strip()
                # Remove surrounding quotes if Llama 3 added them
                if clean_narration.startswith('"') and clean_narration.endswith('"'):
                    clean_narration = clean_narration[1:-1].strip()
                # --------------------------------------------
                
                if clean_narration:
                    narrations.append(clean_narration)
                    continue
        except Exception:
            pass

        # Fallback: generate a simple narration from the panel data
        narration = _fallback_narration(panel)
        narrations.append(narration)

    return narrations


def _call_ollama_text(host: str, model: str, prompt: str) -> str:
    """Call Ollama's text generation API.

    Args:
        host: Ollama server URL.
        model: Model name.
        prompt: Text prompt.

    Returns:
        Generated text response.
    """
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _fallback_narration(panel: dict) -> str:
    """Generate a simple narration when LLM is unavailable.

    Args:
        panel: Panel analysis dict.

    Returns:
        Simple narration string.
    """
    action = panel.get("action", "A scene unfolds")
    emotion = panel.get("emotion", "")
    dialogue = panel.get("dialogue", "")
    mood = panel.get("mood", "calm")

    parts = []
    if action:
        parts.append(action + ".")
    if emotion:
        parts.append(f"The mood is {emotion}.")
    if dialogue:
        parts.append(f'"{dialogue}"')

    return " ".join(parts) if parts else "The scene continues."


def _synthesize_audio(
    text: str,
    output_path: Path,
    engine: str,
    config: dict,
    silence_pad_ms: int = 300,
) -> Path:
    """Synthesize text to speech using the configured engine.

    Args:
        text: Narration text to synthesize.
        output_path: Path for the output audio file.
        engine: TTS engine name: 'edge', 'coqui', or 'pyttsx3'.
        config: Narration config dict.
        silence_pad_ms: Silence padding to add at end of audio (ms).

    Returns:
        Path to the output audio file.
    """
    if engine == "edge":
        _tts_edge(text, output_path)
    elif engine == "coqui":
        _tts_coqui(text, output_path, config)
    elif engine == "pyttsx3":
        _tts_pyttsx3(text, output_path)
    else:
        # Default to pyttsx3 as universal fallback
        _tts_pyttsx3(text, output_path)

    # Add silence padding if pydub is available
    _add_silence_padding(output_path, silence_pad_ms)

    return output_path


def _tts_edge(text: str, output_path: Path):
    """Synthesize using Microsoft Edge TTS (free, no API key)."""
    try:
        import edge_tts
    except ImportError:
        print("  Warning: edge-tts not installed. Falling back to pyttsx3.")
        _tts_pyttsx3(text, output_path)
        return

    async def _run():
        communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
        await communicate.save(str(output_path))

    asyncio.run(_run())


def _tts_coqui(text: str, output_path: Path, config: dict):
    """Synthesize using Coqui TTS (local, free)."""
    try:
        from TTS.api import TTS
    except ImportError:
        print("  Warning: Coqui TTS not installed. Falling back to pyttsx3.")
        _tts_pyttsx3(text, output_path)
        return

    model_name = config.get("tts_model", "tts_models/en/ljspeech/tacotron2-DDC")
    tts = TTS(model_name)
    # Coqui outputs WAV, so save as WAV then convert
    wav_path = output_path.with_suffix(".wav")
    tts.tts_to_file(text=text, file_path=str(wav_path))

    # Convert to MP3
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(str(wav_path))
        audio.export(str(output_path), format="mp3")
        wav_path.unlink(missing_ok=True)
    except Exception:
        # If conversion fails, keep the WAV
        import shutil
        shutil.move(str(wav_path), str(output_path))


def _tts_pyttsx3(text: str, output_path: Path):
    """Synthesize using pyttsx3 (fully offline, universal fallback)."""
    try:
        import pyttsx3
    except ImportError:
        # Ultimate fallback: create a silent audio file
        print("  Warning: pyttsx3 not installed. Creating silent audio placeholder.")
        _create_silent_audio(output_path, duration_ms=2000)
        return

    engine = pyttsx3.init()
    wav_path = output_path.with_suffix(".wav")
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()

    # Try to convert to MP3, otherwise keep as WAV
    try:
        from pydub import AudioSegment
        if wav_path.exists() and wav_path.stat().st_size > 0:
            try:
                audio = AudioSegment.from_wav(str(wav_path))
                audio.export(str(output_path), format="mp3")
                wav_path.unlink(missing_ok=True)
            except (FileNotFoundError, OSError):
                # FFmpeg not available, keep WAV
                import shutil
                shutil.move(str(wav_path), str(output_path))
        else:
            _create_silent_audio(output_path)
    except Exception:
        if wav_path.exists():
            import shutil
            shutil.move(str(wav_path), str(output_path))
        else:
            _create_silent_audio(output_path)


def _create_silent_audio(output_path: Path, duration_ms: int = 2000):
    """Create a silent audio file as a placeholder."""
    try:
        from pydub import AudioSegment

        # Generate silence
        silent = AudioSegment.silent(duration=duration_ms)
        # Try MP3 first, fall back to WAV if ffmpeg is missing
        try:
            silent.export(str(output_path), format="mp3")
        except (FileNotFoundError, OSError):
            wav_path = output_path.with_suffix(".wav")
            silent.export(str(wav_path), format="wav")
            if wav_path != output_path:
                import shutil
                shutil.move(str(wav_path), str(output_path))
    except ImportError:
        # Create a minimal WAV file header (silence)
        import struct
        sample_rate = 22050
        num_samples = int(sample_rate * duration_ms / 1000)
        data_size = num_samples * 2  # 16-bit mono
        with open(output_path, "wb") as f:
            # WAV header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))  # chunk size
            f.write(struct.pack("<H", 1))   # PCM
            f.write(struct.pack("<H", 1))   # mono
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", sample_rate * 2))
            f.write(struct.pack("<H", 2))   # block align
            f.write(struct.pack("<H", 16))  # bits per sample
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(b"\x00" * data_size)


def _add_silence_padding(audio_path: Path, pad_ms: int = 300):
    """Add silence padding to the end of an audio file."""
    if pad_ms <= 0:
        return

    try:
        from pydub import AudioSegment

        if not audio_path.exists() or audio_path.stat().st_size < 100:
            return

        audio = AudioSegment.from_file(str(audio_path))
        silence = AudioSegment.silent(duration=pad_ms)
        padded = audio + silence
        # Try MP3 first, fall back to WAV
        try:
            padded.export(str(audio_path), format="mp3")
        except (FileNotFoundError, OSError):
            padded.export(str(audio_path), format="wav")
    except Exception:
        pass  # Don't fail if padding can't be added


if __name__ == "__main__":
    import sys
    import uuid

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.narrate <panels_json_path>")
        sys.exit(1)

    panels_json = sys.argv[1]
    jid = str(uuid.uuid4())[:8]
    print(f"Generating narration from: {panels_json}")
    print(f"Job ID: {jid}")

    try:
        audio_files = generate_narration(panels_json, jid)
        print(f"[SUCCESS] Generated {len(audio_files)} audio files in work/{jid}/audio/")
        for a in audio_files:
            print(f"  - {a.name}")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
