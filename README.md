# MangaMotion

**Manga PDF to Panel-by-Panel Narrated Video**

MangaMotion is a local-first, open-source application that converts manga PDF files into narrated, panel-by-panel cinematic videos. Upload a manga PDF and the system automatically detects panels, uses AI vision to understand scenes, generates narration, synthesises voiceover audio, and assembles a final MP4 video.

---

## Features

- **5-stage pipeline**: PDF extraction -> panel detection -> AI scene analysis -> narration/TTS -> video assembly
- **Gradio web UI** with real-time progress tracking
- **Multiple TTS engines**: Edge TTS (best quality), pyttsx3 (offline), Coqui TTS (optional)
- **Ken Burns effect** for cinematic pan & zoom on static panels
- **RTL/LTR reading order** support (Japanese manga & Western comics)
- **Dialogue text overlays** on video
- **Configurable** via `config.yaml`

---

## Prerequisites

| Dependency | Required | Purpose |
|-----------|----------|---------|
| **Python 3.10+** | Yes | Runtime |
| **FFmpeg** | Yes | Video encoding & audio conversion |
| **Ollama** | Recommended | AI scene analysis & narration generation |

---

## Setup Instructions

### 1. Clone / Download

```bash
cd d:\mangaproject
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

#### Windows
```powershell
# Option A: Using winget
winget install FFmpeg

# Option B: Using chocolatey
choco install ffmpeg

# Option C: Manual download from https://ffmpeg.org/download.html
# Extract and add to PATH
```

#### macOS
```bash
brew install ffmpeg
```

#### Ubuntu/Debian
```bash
sudo apt update && sudo apt install ffmpeg
```

### 4. Install & Configure Ollama (Recommended)

Ollama is used for AI scene analysis (Stage 3) and narration script generation (Stage 4). Without it, the pipeline uses fallback descriptions.

#### Install Ollama

- **Windows/macOS**: Download from [ollama.ai](https://ollama.ai/download)
- **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`

#### Pull Models

```bash
# Vision model for scene analysis (smallest, CPU-friendly)
ollama pull moondream

# Text model for narration scripts
ollama pull mistral:7b
```

#### Start Ollama

```bash
ollama serve
```

> **Note**: If Ollama is not running, the pipeline will still work using fallback descriptions and narration. The UI will show a clear warning message. For vision models, we recommend `moondream`.

---

## Usage

### Web UI (Recommended)

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

1. Upload a manga PDF
2. Choose reading direction (RTL for Japanese, LTR for Western)
3. Select TTS engine
4. Click **Generate Video**
5. Watch progress and download the MP4

### CLI (Individual Stages)

Each pipeline stage can be run independently:

```bash
# Stage 1: Extract PDF pages
python -m pipeline.extract path/to/manga.pdf

# Stage 2: Detect panels (provide pages directory)
python -m pipeline.detect_panels work/<job_id>/pages

# Stage 3: Analyze panels with AI
python -m pipeline.analyze work/<job_id>/panels

# Stage 4: Generate narration & audio
python -m pipeline.narrate work/<job_id>/panels.json

# Stage 5: Assemble final video
python -m pipeline.assemble <job_id>
```

### Running Tests

```bash
python tests/test_extract.py
python tests/test_detect_panels.py
python tests/test_analyze.py
python tests/test_narrate.py
python tests/test_assemble.py
```

---

## Configuration

All settings are in `config.yaml`:

| Section | Key Settings |
|---------|-------------|
| `pdf_extraction` | DPI scale, colorspace (rgb/gray), output format |
| `panel_detection` | Reading direction (rtl/ltr), min panel area, padding |
| `scene_analysis` | Ollama host, vision model, retry count |
| `narration` | Script model, TTS engine (edge/coqui/pyttsx3), silence padding |
| `video` | Resolution, FPS, Ken Burns toggle, dialogue overlays, codec |

---

## Project Structure

```
manga-motion/
├── app.py                  # Gradio web UI entry point
├── config.yaml             # All tuneable settings
├── pipeline/
│   ├── __init__.py         # Config loader & job dir manager
│   ├── extract.py          # Stage 1: PDF -> page PNGs
│   ├── detect_panels.py    # Stage 2: Panel detection & cropping
│   ├── analyze.py          # Stage 3: AI scene analysis via Ollama
│   ├── narrate.py          # Stage 4: Script generation & TTS
│   └── assemble.py         # Stage 5: Video assembly with MoviePy
├── tests/                  # Per-stage test scripts
├── test_data/              # Sample manga page for testing
├── models/                 # Downloaded model weights
├── work/                   # Temporary per-job files
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ollama is not running` | Start Ollama: `ollama serve`, then `ollama pull moondream` |
| `FFmpeg not found` | Install FFmpeg and add to PATH |
| `pyttsx3 not installed` | `pip install pyttsx3` for offline TTS |
| Video has no audio | Ensure FFmpeg is installed for audio encoding |
| Panel detection finds 1 panel | Adjust `min_panel_area_pct` in config.yaml |

---

## License

Open source. See individual library licenses for dependencies.
