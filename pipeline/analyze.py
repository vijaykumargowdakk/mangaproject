"""Stage 3 -- AI Scene Analysis: Analyze manga panels with a vision LLM via Ollama."""

import base64
import json
import re
import requests
from pathlib import Path

from pipeline import load_config, get_job_dir


# Prompt sent to the vision model for each panel
ANALYSIS_PROMPT = """You are a manga scene analyst. Given a manga panel image, return ONLY
valid JSON with these fields:
{
  "characters": ["list of characters visible"],
  "action": "what is happening in the panel",
  "emotion": "dominant emotion/tone of the scene",
  "dialogue": "text visible in speech bubbles (empty string if none)",
  "mood": "one of: tense, calm, comedic, dramatic, romantic, action",
  "is_splash": false
}

Rules:
- Return ONLY the JSON object, no markdown fences, no explanation.
- If you cannot determine a field, use a reasonable default.
- "is_splash" should be true only for full-page dramatic panels.
- "dialogue" should be empty string if no speech bubbles are visible.
"""


def check_ollama_available(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running and reachable.

    Args:
        host: Ollama server URL.

    Returns:
        True if Ollama is reachable, False otherwise.
    """
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def analyze_panels(
    panels_dir: str | Path,
    job_id: str,
    config: dict | None = None,
    panel_list: list[dict] | None = None,
) -> list[dict]:
    """Analyze each panel image using a vision LLM via Ollama.

    Args:
        panels_dir: Directory containing panel PNG images.
        job_id: Unique job identifier.
        config: Configuration dict. Loaded from config.yaml if None.
        panel_list: Optional pre-existing panel manifest list.

    Returns:
        List of panel analysis dicts with scene descriptions.

    Raises:
        ConnectionError: If Ollama is not running.
    """
    panels_dir = Path(panels_dir)
    if config is None:
        config = load_config()

    analysis_cfg = config.get("scene_analysis", {})
    host = analysis_cfg.get("ollama_host", "http://localhost:11434")
    model = analysis_cfg.get("vision_model", "moondream")
    max_retries = analysis_cfg.get("max_retries", 3)

    # Check Ollama availability
    if not check_ollama_available(host):
        raise ConnectionError(
            "Ollama is not running!\n"
            "Please start Ollama first:\n"
            "  1. Install: https://ollama.ai/download\n"
            "  2. Start: run 'ollama serve' in a terminal\n"
            "  3. Pull a vision model: 'ollama pull moondream'\n"
            "Then try again."
        )

    # Get panel files
    if panel_list:
        panel_files = [(p["panel_id"], Path(p["image_path"])) for p in panel_list]
    else:
        panel_files = [
            (f.stem, f)
            for f in sorted(panels_dir.iterdir())
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]

    if not panel_files:
        raise ValueError(f"No panel images found in {panels_dir}")

    job_dir = get_job_dir(job_id)
    results = []

    for panel_id, panel_path in panel_files:
        analysis = _analyze_single_panel(
            panel_path, panel_id, host, model, max_retries
        )
        results.append(analysis)

    # Save panels.json
    output_path = job_dir / "panels.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def _analyze_single_panel(
    panel_path: Path,
    panel_id: str,
    host: str,
    model: str,
    max_retries: int,
) -> dict:
    """Analyze a single panel image via Ollama vision API.

    Args:
        panel_path: Path to the panel image.
        panel_id: Panel identifier.
        host: Ollama server URL.
        model: Vision model name.
        max_retries: Maximum retry attempts for malformed JSON.

    Returns:
        Dict with panel analysis results.
    """
    # Read and encode image
    with open(panel_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Default result in case all retries fail
    default_result = {
        "panel_id": panel_id,
        "image_path": str(panel_path),
        "characters": [],
        "action": "Scene depicted in manga panel",
        "emotion": "neutral",
        "dialogue": "",
        "mood": "calm",
        "is_splash": False,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": ANALYSIS_PROMPT,
                    "images": [image_b64],
                    "stream": False,
                },
                timeout=300,
            )
            resp.raise_for_status()

            response_text = resp.json().get("response", "")
            parsed = _parse_json_response(response_text)

            if parsed:
                result = {**default_result, **parsed}
                result["panel_id"] = panel_id
                result["image_path"] = str(panel_path)
                return result

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"  Warning: Failed to analyze {panel_id} after {max_retries} attempts: {e}")
            continue
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Warning: Error analyzing {panel_id}: {e}")
            continue

    return default_result


def _parse_json_response(text: str) -> dict | None:
    """Parse JSON from LLM response, handling common formatting issues.

    Args:
        text: Raw text response from the LLM.

    Returns:
        Parsed dict or None if parsing fails.
    """
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code fences
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{.*\}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pattern else match.group(0))
            except json.JSONDecodeError:
                continue

    # Try to fix common issues: single quotes -> double quotes
    try:
        fixed = text.strip().replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None


if __name__ == "__main__":
    import sys
    import uuid

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.analyze <panels_dir>")
        sys.exit(1)

    panels = sys.argv[1]
    jid = str(uuid.uuid4())[:8]
    print(f"Analyzing panels in: {panels}")
    print(f"Job ID: {jid}")

    try:
        results = analyze_panels(panels, jid)
        print(f"[SUCCESS] Analyzed {len(results)} panels")
        for r in results:
            print(f"  - {r['panel_id']}: mood={r['mood']}, action={r['action'][:50]}")
    except ConnectionError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
