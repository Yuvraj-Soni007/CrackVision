"""Utility helpers used across the project."""

import os
import yaml
import logging
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(name: str = "crackvision", level: int = logging.INFO) -> logging.Logger:
    """Create a formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()

# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

_CONFIG_CACHE: dict | None = None


def load_config(config_path: str | None = None) -> dict:
    """Load YAML configuration with caching."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = str(Path(__file__).resolve().parents[1] / "config" / "config.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    _CONFIG_CACHE = cfg
    return cfg


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def list_images(directory: str, extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) -> list[str]:
    """List all image files in a directory."""
    images = []
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(extensions):
            images.append(os.path.join(directory, f))
    return images
