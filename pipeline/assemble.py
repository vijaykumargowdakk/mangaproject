"""Stage 5 -- Video Assembly: Combine panels and audio into a cinematic MP4 video."""

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline import load_config, get_job_dir


def _mp_set_duration(clip, duration):
    """Set clip duration, compatible with MoviePy v1 and v2."""
    try:
        return clip.with_duration(duration)
    except AttributeError:
        return clip.set_duration(duration)


def _mp_set_audio(clip, audio):
    """Set clip audio, compatible with MoviePy v1 and v2."""
    try:
        return clip.with_audio(audio)
    except AttributeError:
        return clip.set_audio(audio)


def _mp_set_position(clip, pos):
    """Set clip position, compatible with MoviePy v1 and v2."""
    try:
        return clip.with_position(pos)
    except AttributeError:
        return clip.set_position(pos)


def assemble_video(
    job_id: str,
    config: dict | None = None,
) -> Path:
    """Assemble panel images and audio into a final MP4 video.

    Args:
        job_id: Unique job identifier (must have panels/ and audio/ in work dir).
        config: Configuration dict. Loaded from config.yaml if None.

    Returns:
        Path to the output MP4 file.

    Raises:
        FileNotFoundError: If required files are missing.
        RuntimeError: If video assembly fails.
    """
    if config is None:
        config = load_config()

    vid_cfg = config.get("video", {})
    resolution_str = vid_cfg.get("resolution", "1920x1080")
    fps = vid_cfg.get("fps", 24)
    codec = vid_cfg.get("codec", "libx264")
    crf = vid_cfg.get("crf", 20)
    use_ken_burns = vid_cfg.get("ken_burns", True)
    kb_zoom = vid_cfg.get("ken_burns_zoom", 1.15)
    use_bg_music = vid_cfg.get("background_music", False)
    music_vol_db = vid_cfg.get("music_volume_db", -15)
    use_dialogue_overlay = vid_cfg.get("dialogue_overlay", True)

    # Parse resolution
    try:
        out_w, out_h = map(int, resolution_str.split("x"))
    except ValueError:
        out_w, out_h = 1920, 1080

    job_dir = get_job_dir(job_id)
    panels_dir = job_dir / "panels"
    audio_dir = job_dir / "audio"
    video_dir = job_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Load panels manifest/json
    panels_json_path = job_dir / "panels.json"
    panels_manifest_path = job_dir / "panels_manifest.json"

    panels_data = []
    if panels_json_path.exists():
        with open(panels_json_path, "r", encoding="utf-8") as f:
            panels_data = json.load(f)
    elif panels_manifest_path.exists():
        with open(panels_manifest_path, "r", encoding="utf-8") as f:
            panels_data = json.load(f)

    # Collect panel images and matching audio
    panel_clips_data = _collect_panel_data(panels_dir, audio_dir, panels_data)

    if not panel_clips_data:
        raise ValueError("No panel images found for video assembly")

    # Import moviepy (handle both v1 and v2)
    try:
        try:
            from moviepy.editor import (
                ImageClip, AudioFileClip, CompositeVideoClip,
                concatenate_videoclips, TextClip, ImageSequenceClip,
            )
        except ImportError:
            from moviepy import (
                ImageClip, AudioFileClip, CompositeVideoClip,
                concatenate_videoclips, TextClip, ImageSequenceClip,
            )
    except ImportError:
        raise ImportError(
            "MoviePy is required for video assembly.\n"
            "Install it with: pip install moviepy"
        )

    clips = []
    total_panels = len(panel_clips_data)
    for idx, panel_info in enumerate(panel_clips_data):
        img_path = panel_info["image_path"]
        audio_path = panel_info.get("audio_path")
        dialogue = panel_info.get("dialogue", "")
        is_splash = panel_info.get("is_splash", False)

        # Load image and resize to output resolution
        print(f"  Assembling panel {idx + 1}/{total_panels}: {panel_info['panel_id']}")
        img = Image.open(img_path).convert("RGB")
        img = _fit_to_resolution(img, out_w, out_h)

        # Determine clip duration
        duration = 3.0  # default duration
        audio_clip = None
        if audio_path and Path(audio_path).exists() and Path(audio_path).stat().st_size > 100:
            try:
                audio_clip = AudioFileClip(str(audio_path))
                duration = audio_clip.duration
                if duration < 1.0:
                    duration = 3.0
                    audio_clip = None
            except Exception:
                duration = 3.0

        # Splash panels get extra time
        if is_splash:
            duration = max(duration, 5.0)

        # Create video clip
        if use_ken_burns:
            clip = _create_ken_burns_clip(img, duration, fps, out_w, out_h, kb_zoom, is_splash)
        else:
            img_array = np.array(img)
            clip = _mp_set_duration(ImageClip(img_array), duration)

        # Add audio
        if audio_clip:
            clip = _mp_set_audio(clip, audio_clip)

        # Add dialogue overlay
        if use_dialogue_overlay and dialogue:
            try:
                txt_clip = _create_dialogue_overlay(dialogue, out_w, out_h, duration, vid_cfg)
                clip = CompositeVideoClip([clip, txt_clip])
            except Exception:
                pass  # Skip overlay if it fails (font issues etc.)

        clips.append(clip)

    if not clips:
        raise RuntimeError("No video clips were created")

    # Concatenate all clips
    final = concatenate_videoclips(clips, method="compose")

    # Export
    output_path = video_dir / "output_final.mp4"
    print(f"  Encoding video ({total_panels} clips)...")
    final.write_videofile(
        str(output_path),
        fps=fps,
        codec=codec,
        audio_codec="aac",
        bitrate="5000k",
        preset="medium",
        ffmpeg_params=["-crf", str(crf)],
        logger="bar",  # Show encoding progress bar
    )

    # Cleanup moviepy resources
    final.close()
    for clip in clips:
        clip.close()

    return output_path


def _collect_panel_data(
    panels_dir: Path,
    audio_dir: Path,
    panels_data: list[dict],
) -> list[dict]:
    """Collect panel images and match with audio and metadata.

    Args:
        panels_dir: Directory with panel PNG images.
        audio_dir: Directory with audio files.
        panels_data: Panel analysis data from panels.json.

    Returns:
        List of dicts with image_path, audio_path, dialogue, is_splash.
    """
    # Get panel image files
    panel_files = sorted(
        [f for f in panels_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    ) if panels_dir.exists() else []

    if not panel_files:
        return []

    # Build lookup from panels_data
    data_lookup = {}
    for pd in panels_data:
        pid = pd.get("panel_id", "")
        data_lookup[pid] = pd

    result = []
    for img_file in panel_files:
        panel_id = img_file.stem  # e.g., "panel_0000"

        # Find matching audio
        audio_path = None
        for ext in [".mp3", ".wav", ".ogg"]:
            candidate = audio_dir / f"{panel_id}{ext}"
            if candidate.exists():
                audio_path = str(candidate)
                break

        # Get metadata
        meta = data_lookup.get(panel_id, {})

        result.append({
            "panel_id": panel_id,
            "image_path": str(img_file),
            "audio_path": audio_path,
            "dialogue": meta.get("dialogue", ""),
            "is_splash": meta.get("is_splash", False),
            "mood": meta.get("mood", "calm"),
        })

    return result


def _fit_to_resolution(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize and pad image to fit target resolution while maintaining aspect ratio.

    Uses letterboxing (black bars) for non-matching aspect ratios.

    Args:
        img: Input PIL Image.
        target_w: Target width.
        target_h: Target height.

    Returns:
        Resized PIL Image at exactly target_w x target_h.
    """
    img_w, img_h = img.size
    target_ratio = target_w / target_h
    img_ratio = img_w / img_h

    if img_ratio > target_ratio:
        # Image is wider - fit width, pad top/bottom
        new_w = target_w
        new_h = int(target_w / img_ratio)
    else:
        # Image is taller - fit height, pad left/right
        new_h = target_h
        new_w = int(target_h * img_ratio)

    # Resize with high quality
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create black canvas and paste centered
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def _create_ken_burns_clip(
    img: Image.Image,
    duration: float,
    fps: int,
    out_w: int,
    out_h: int,
    max_zoom: float = 1.15,
    is_splash: bool = False,
):
    """Create a Ken Burns effect clip from a static image.

    Uses lazy frame generation (VideoClip.make_frame) to avoid storing
    all frames in memory at once.

    Args:
        img: Input PIL Image (already fitted to resolution).
        duration: Clip duration in seconds.
        fps: Frames per second.
        out_w: Output width.
        out_h: Output height.
        max_zoom: Maximum zoom level.
        is_splash: If True, use slower zoom.

    Returns:
        MoviePy VideoClip with Ken Burns effect.
    """
    try:
        from moviepy.editor import VideoClip
    except ImportError:
        from moviepy import VideoClip

    # For Ken Burns, we need the image larger than output to allow for zoom/pan
    pad_factor = max_zoom + 0.05
    padded_w = int(out_w * pad_factor)
    padded_h = int(out_h * pad_factor)

    # Resize image to padded size (stored once)
    img_large = img.resize((padded_w, padded_h), Image.LANCZOS)
    img_array = np.array(img_large)

    # Choose random effect
    effects = ["zoom_in", "zoom_out", "pan_left", "pan_right"]
    effect = random.choice(effects)

    if is_splash:
        max_zoom = 1.0 + (max_zoom - 1.0) * 0.5

    # Lazy frame generator — creates ONE frame per call, no memory buildup
    def make_frame(t_sec):
        t = t_sec / max(duration, 0.001)  # normalized 0..1

        if effect == "zoom_in":
            scale = 1.0 + t * (max_zoom - 1.0)
            cx, cy = padded_w // 2, padded_h // 2
        elif effect == "zoom_out":
            scale = max_zoom - t * (max_zoom - 1.0)
            cx, cy = padded_w // 2, padded_h // 2
        elif effect == "pan_left":
            scale = 1.0
            cx = int(padded_w * 0.55 - t * padded_w * 0.1)
            cy = padded_h // 2
        else:  # pan_right
            scale = 1.0
            cx = int(padded_w * 0.45 + t * padded_w * 0.1)
            cy = padded_h // 2

        crop_w = int(out_w / scale)
        crop_h = int(out_h / scale)

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(padded_w, x1 + crop_w)
        y2 = min(padded_h, y1 + crop_h)

        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)

        cropped = img_array[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = img_array

        frame_img = Image.fromarray(cropped)
        frame_img = frame_img.resize((out_w, out_h), Image.LANCZOS)
        return np.array(frame_img)

    clip = VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip


def _create_dialogue_overlay(
    text: str,
    width: int,
    height: int,
    duration: float,
    vid_cfg: dict,
) -> "TextClip":
    """Create a dialogue text overlay clip.

    Args:
        text: Dialogue text to display.
        width: Video width.
        height: Video height.
        duration: Clip duration.
        vid_cfg: Video configuration dict.

    Returns:
        MoviePy TextClip positioned at the bottom of the frame.
    """
    try:
        from moviepy.editor import TextClip
    except ImportError:
        from moviepy import TextClip

    font = vid_cfg.get("dialogue_font", "Arial")
    font_size = max(24, width // 40)

    # Truncate very long dialogue
    if len(text) > 200:
        text = text[:197] + "..."

    txt = _mp_set_position(
        _mp_set_duration(
            TextClip(
                text,
                font_size=font_size,
                font=font,
                color="white",
                bg_color="rgba(0,0,0,0.6)",
                size=(width - 100, None),
                method="caption",
            ),
            duration,
        ),
        ("center", height - 120),
    )

    return txt


if __name__ == "__main__":
    import sys
    import uuid

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.assemble <job_id>")
        sys.exit(1)

    jid = sys.argv[1]
    print(f"Assembling video for job: {jid}")

    try:
        output = assemble_video(jid)
        print(f"[SUCCESS] Video assembled: {output}")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
