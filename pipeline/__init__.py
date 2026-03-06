"""MangaMotion Pipeline — Manga PDF → Narrated Video."""

import os
import yaml
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load configuration from config.yaml.
    
    Args:
        config_path: Optional path to config file. Defaults to project root config.yaml.
    
    Returns:
        Dictionary of configuration settings.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_job_dir(job_id: str) -> Path:
    """Get or create the working directory for a job.
    
    Args:
        job_id: Unique job identifier.
    
    Returns:
        Path to the job's working directory.
    """
    job_dir = ROOT_DIR / "work" / job_id
    for subdir in ["pages", "panels", "audio", "video"]:
        (job_dir / subdir).mkdir(parents=True, exist_ok=True)
    return job_dir
