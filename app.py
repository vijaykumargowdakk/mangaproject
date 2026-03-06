"""MangaMotion — Gradio Web UI for manga-to-video conversion."""

import sys
import uuid
import traceback
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr
from pipeline import load_config, get_job_dir
from pipeline.extract import extract_pages
from pipeline.detect_panels import detect_panels
from pipeline.analyze import analyze_panels, check_ollama_available
from pipeline.narrate import generate_narration
from pipeline.assemble import assemble_video


def check_dependencies() -> list[str]:
    """Check for required dependencies and return list of issues."""
    issues = []

    # Check FFmpeg
    import shutil
    if not shutil.which("ffmpeg"):
        issues.append(
            "FFmpeg not found in PATH. Video assembly requires FFmpeg.\n"
            "Install from: https://ffmpeg.org/download.html"
        )

    return issues


def run_pipeline(
    pdf_file,
    reading_direction: str,
    tts_engine: str,
    bg_music: bool,
    resolution: str,
    progress=gr.Progress(),
):
    """Run the full MangaMotion pipeline.

    Args:
        pdf_file: Uploaded PDF file path.
        reading_direction: 'rtl' or 'ltr'.
        tts_engine: TTS engine name.
        bg_music: Whether to add background music.
        resolution: Output resolution string.
        progress: Gradio progress tracker.

    Returns:
        Tuple of (video_path, log_text).
    """
    if pdf_file is None:
        return None, "Error: Please upload a manga PDF file."

    log_lines: list[str] = []

    def log(msg):
        log_lines.append(msg)
        print(msg)

    try:
        config = load_config()

        # Override config with UI settings
        config["panel_detection"]["reading_direction"] = reading_direction
        config["narration"]["tts_engine"] = tts_engine
        config["video"]["background_music"] = bg_music

        # Parse resolution
        if resolution == "720p":
            config["video"]["resolution"] = "1280x720"
        else:
            config["video"]["resolution"] = "1920x1080"

        job_id = uuid.uuid4().hex[:8]
        job_dir = get_job_dir(job_id)
        log(f"Job ID: {job_id}")

        # ── Stage 1: Extract Pages ──────────────────────
        progress(0.0, desc="Stage 1/5: Extracting pages...")
        log("Stage 1: Extracting pages from PDF...")

        pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        pages = extract_pages(pdf_path, job_id, config)
        log(f"  Extracted {len(pages)} page(s)")

        # ── Stage 2: Detect Panels ──────────────────────
        progress(0.2, desc="Stage 2/5: Detecting panels...")
        log("Stage 2: Detecting panels...")

        pages_dir = job_dir / "pages"
        panels = detect_panels(pages_dir, job_id, config)
        log(f"  Detected {len(panels)} panel(s)")

        # ── Stage 3: Analyze Scenes ─────────────────────
        progress(0.4, desc="Stage 3/5: Analyzing scenes with AI...")
        log("Stage 3: Analyzing scenes with AI vision model...")

        # Check Ollama availability
        host = config.get("scene_analysis", {}).get("ollama_host", "http://localhost:11434")
        if not check_ollama_available(host):
            log("  WARNING: Ollama is not running!")
            log("  Please start Ollama: 'ollama serve'")
            log("  Pull a model: 'ollama pull moondream'")
            log("  Continuing with fallback analysis...")

        panels_dir = job_dir / "panels"
        try:
            analysis = analyze_panels(panels_dir, job_id, config, panels)
        except ConnectionError as e:
            log(f"  Ollama not available: Using fallback analysis")
            # Create fallback panels.json
            import json
            fallback = []
            for p in panels:
                fallback.append({
                    "panel_id": p["panel_id"],
                    "image_path": p["image_path"],
                    "characters": [],
                    "action": "A scene from the manga",
                    "emotion": "neutral",
                    "dialogue": "",
                    "mood": "calm",
                    "is_splash": False,
                })
            panels_json = job_dir / "panels.json"
            with open(panels_json, "w") as f:
                json.dump(fallback, f, indent=2, default=str)
            analysis = fallback

        log(f"  Analyzed {len(analysis)} panel(s)")

        # ── Stage 4: Generate Narration ─────────────────
        progress(0.6, desc="Stage 4/5: Generating narration & audio...")
        log("Stage 4: Generating narration scripts & TTS audio...")

        panels_json = job_dir / "panels.json"
        audio_files = generate_narration(panels_json, job_id, config)
        log(f"  Generated {len(audio_files)} audio file(s)")

        # ── Stage 5: Assemble Video ─────────────────────
        progress(0.8, desc="Stage 5/5: Assembling video...")
        log("Stage 5: Assembling final video...")

        output_path = assemble_video(job_id, config)
        log(f"  Video saved: {output_path}")

        progress(1.0, desc="Done!")
        log("Pipeline complete!")

        return str(output_path), "\n".join(log_lines)

    except Exception as e:
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        log_lines.append(error_msg)
        return None, "\n".join(log_lines)


def build_ui():
    """Build and return the Gradio Blocks interface."""

    # Check dependencies on startup
    dep_issues = check_dependencies()

    with gr.Blocks(
        title="MangaMotion",
    ) as app:
        gr.Markdown("# MangaMotion", elem_classes=["main-title"])
        gr.Markdown("*Manga PDF to Panel-by-Panel Narrated Video*", elem_classes=["subtitle"])

        # Show dependency warnings
        if dep_issues:
            for issue in dep_issues:
                gr.Markdown(f"> **Warning:** {issue}")

        with gr.Row():
            # ── Left Panel: Input & Settings ──
            with gr.Column(scale=1):
                gr.Markdown("### Input & Settings")

                pdf_input = gr.File(
                    label="Upload Manga PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )

                reading_dir = gr.Radio(
                    choices=["rtl", "ltr"],
                    value="rtl",
                    label="Reading Direction",
                    info="RTL for Japanese manga, LTR for Western comics",
                )

                tts_voice = gr.Dropdown(
                    choices=["edge", "pyttsx3", "coqui"],
                    value="edge",
                    label="TTS Voice Engine",
                    info="edge = best quality (needs internet), pyttsx3 = offline",
                )

                bg_music = gr.Checkbox(
                    value=False,
                    label="Background Music",
                    info="Add background music to the video",
                )

                resolution = gr.Radio(
                    choices=["720p", "1080p"],
                    value="1080p",
                    label="Output Resolution",
                )

                generate_btn = gr.Button(
                    "Generate Video",
                    variant="primary",
                    size="lg",
                )

            # ── Right Panel: Output & Progress ──
            with gr.Column(scale=2):
                gr.Markdown("### Output & Progress")

                video_output = gr.Video(
                    label="Generated Video",
                    interactive=False,
                )

                log_output = gr.Textbox(
                    label="Pipeline Log",
                    lines=15,
                    max_lines=30,
                    interactive=False,
                )

        # Wire up the pipeline
        generate_btn.click(
            fn=run_pipeline,
            inputs=[pdf_input, reading_dir, tts_voice, bg_music, resolution],
            outputs=[video_output, log_output],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="orange",
            secondary_hue="blue",
        ),
        css="""
        .main-title {
            text-align: center;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-top: 0;
        }
        """,
    )
